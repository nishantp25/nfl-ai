"""Source-conflict resolution.

The brief specifies a priority ladder and says to follow it *unless there is a
documented reason otherwise*. That last clause is the important one: an override
is legitimate, an undocumented override is not. So :func:`resolve` takes the
ladder from the manifest, and the only way to depart from it is to supply an
:class:`Override` carrying a reason and a source -- which then shows up in the
output, in the reconciliation log, and in the audit trail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import polars as pl

from .config import REPORT_DIR
from .manifest import get_manifest

log = logging.getLogger(__name__)

RECONCILIATION_LOG = REPORT_DIR / "reconciliations.jsonl"


@dataclass(frozen=True)
class Candidate:
    """One source's answer for a field."""

    value: Any
    tier: str
    dataset: str
    note: str = ""


@dataclass(frozen=True)
class Override:
    """A documented departure from the priority ladder."""

    key: str            # e.g. "game_id=2013_01_DAL_NYG:surface"
    prefer_dataset: str
    reason: str
    source_url: str
    recorded_by: str
    recorded_on: str

    def __post_init__(self) -> None:
        for f in ("reason", "source_url", "recorded_by"):
            if not getattr(self, f).strip():
                raise ValueError(f"override for '{self.key}' needs a non-empty {f}")


@dataclass
class Resolution:
    key: str
    chosen: Candidate
    rejected: list[Candidate] = field(default_factory=list)
    rule: str = "priority_ladder"
    override: Override | None = None
    disagreement: bool = False


def resolve(
    key: str,
    candidates: Sequence[Candidate],
    *,
    overrides: dict[str, Override] | None = None,
    log_to_file: bool = True,
) -> Resolution:
    """Pick a winner among competing values for one field.

    Lower manifest rank wins. Ties inside a tier are broken by candidate order,
    which is deterministic and therefore reproducible.
    """
    if not candidates:
        raise ValueError(f"no candidates supplied for '{key}'")

    mf = get_manifest()
    live = [c for c in candidates if c.value is not None]
    if not live:
        return Resolution(key=key, chosen=candidates[0], rejected=list(candidates[1:]))

    overrides = overrides or {}
    ov = overrides.get(key)
    if ov is not None:
        match = next((c for c in live if c.dataset == ov.prefer_dataset), None)
        if match is None:
            raise ValueError(
                f"override for '{key}' prefers '{ov.prefer_dataset}', which supplied no value"
            )
        res = Resolution(
            key=key,
            chosen=match,
            rejected=[c for c in live if c is not match],
            rule="documented_override",
            override=ov,
            disagreement=len({c.value for c in live}) > 1,
        )
    else:
        ranked = sorted(live, key=lambda c: (mf.tier_rank(c.tier), candidates.index(c)))
        res = Resolution(
            key=key,
            chosen=ranked[0],
            rejected=ranked[1:],
            disagreement=len({str(c.value) for c in live}) > 1,
        )

    if log_to_file and res.disagreement:
        _log(res)
    return res


def _log(res: Resolution) -> None:
    RECONCILIATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "at": datetime.now(timezone.utc).isoformat(),
        "key": res.key,
        "rule": res.rule,
        "chosen": asdict(res.chosen),
        "rejected": [asdict(c) for c in res.rejected],
        "override": asdict(res.override) if res.override else None,
    }
    with RECONCILIATION_LOG.open("a") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def coalesce_by_priority(
    df: pl.DataFrame,
    output: str,
    sources: dict[str, str],
) -> pl.DataFrame:
    """Vectorised ladder over columns: ``{column_name: tier}``.

    The winning tier is recorded per row in ``{output}_source_tier``, and rows
    where the sources disagree are flagged in ``{output}_disagreement`` rather
    than silently smoothed over.
    """
    mf = get_manifest()
    ordered = sorted(sources.items(), key=lambda kv: mf.tier_rank(kv[1]))
    present = [(col, tier) for col, tier in ordered if col in df.columns]
    if not present:
        raise ValueError(f"none of {list(sources)} are present in the frame")

    value = pl.coalesce([pl.col(c) for c, _ in present]).alias(output)

    tier_expr = pl.lit(None, dtype=pl.String)
    for col, tier in reversed(present):
        tier_expr = pl.when(pl.col(col).is_not_null()).then(pl.lit(tier)).otherwise(tier_expr)

    non_null = [pl.col(c).cast(pl.String) for c, _ in present]
    disagreement = (
        pl.concat_list(non_null).list.drop_nulls().list.n_unique() > 1
    ).alias(f"{output}_disagreement")

    return df.with_columns(value, tier_expr.alias(f"{output}_source_tier"), disagreement)
