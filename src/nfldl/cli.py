"""Command line interface.

    nfldl verify                     probe what actually exists
    nfldl verify --write-manifest    write discovered coverage back into the YAML
    nfldl ingest                     fetch everything in the target window
    nfldl ingest play_by_play -s 2020 2021
    nfldl validate                   run quality checks on what is on disk
    nfldl manifest                   print the declared source table
    nfldl inventory                  what is in the lake right now
    nfldl doctor                     end-to-end health check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

from . import ingest as ingest_mod
from . import quality, verify
from .config import COVERAGE_REPORT_PATH, get_settings, nfl_season_of
from .manifest import get_manifest
from .storage import inventory


def _setup_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )


def _seasons(args: argparse.Namespace) -> list[int] | None:
    if args.seasons:
        return sorted(args.seasons)
    if args.season_range:
        lo, hi = args.season_range
        return list(range(lo, hi + 1))
    return None


def cmd_manifest(args: argparse.Namespace) -> int:
    mf = get_manifest()
    if args.json:
        print(json.dumps(
            {n: {"provider": s.provider, "tier": s.tier, "access": s.access,
                 "declared": str(s.declared), "verified": str(s.verified),
                 "required": s.required, "primary_key": list(s.primary_key),
                 "leakage_risk": s.leakage_risk}
             for n, s in mf.datasets.items()},
            indent=2,
        ))
        return 0

    print(f"manifest v{mf.version}  verified_on={mf.verified_on}  ({mf.path})")
    print(f"{'dataset':20s} {'tier':18s} {'declared':14s} {'verified':14s} req leak")
    print("-" * 82)
    for name in mf.names():
        s = mf[name]
        print(
            f"{name:20s} {s.tier:18s} {str(s.declared):14s} {str(s.verified):14s} "
            f"{'Y' if s.required else 'n':3s} {s.leakage_risk}"
        )
    print()
    print(f"forbidden (must be derived, never downloaded): {len(mf.forbidden_downloads)} features")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify.verify_all(args.datasets or None, sample=args.sample)
    print(verify.format_report(report))
    if args.write_manifest:
        verify.write_back_to_manifest(report)
        print(f"\nmanifest updated with discovered coverage")
    return 1 if report["failures"] else 0


def cmd_ingest(args: argparse.Namespace) -> int:
    if get_settings().strict_coverage and not COVERAGE_REPORT_PATH.exists():
        print(
            "no coverage report found. Run `nfldl verify` first, or set "
            "NFLDL_STRICT_COVERAGE=0 to ingest against declared coverage.",
            file=sys.stderr,
        )
        return 2
    report = ingest_mod.ingest(
        args.datasets or None,
        seasons=_seasons(args),
        force=args.force,
        dry_run=args.dry_run,
        required_only=args.required_only,
    )
    print(ingest_mod.format_report(report))
    return 1 if report.failures else 0


def cmd_validate(args: argparse.Namespace) -> int:
    report = quality.validate_all(args.datasets or None)
    print(quality.format_report(report))
    return 1 if report["n_errors"] else 0


def cmd_inventory(_: argparse.Namespace) -> int:
    inv = inventory()
    if inv.is_empty():
        print("lake is empty -- run `nfldl ingest`")
        return 0
    summary = (
        inv.group_by("dataset")
        .agg([
            __import__("polars").len().alias("partitions"),
            __import__("polars").col("rows").sum().alias("rows"),
        ])
        .sort("dataset")
    )
    print(summary)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"current season (nfldl): {nfl_season_of()}")
    print(f"target window:          {get_settings().target_seasons[0]}-{get_settings().target_seasons[-1]}")
    print()
    mf = get_manifest()
    print(f"manifest:  {len(mf.datasets)} datasets, {len(mf.required_names())} required")
    print(f"           verified_on={mf.verified_on}")
    print()
    if COVERAGE_REPORT_PATH.exists():
        report = json.loads(COVERAGE_REPORT_PATH.read_text())
        print(verify.format_report(report))
    else:
        print("no coverage report yet -- run `nfldl verify`")
    print()
    inv = inventory()
    print(f"lake:      {inv.height} partitions on disk")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nfldl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument("-q", "--quiet", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add_datasets(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("datasets", nargs="*", help="dataset names (default: all)")

    def add_seasons(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-s", "--seasons", nargs="+", type=int)
        sp.add_argument("-r", "--season-range", nargs=2, type=int, metavar=("FROM", "TO"))

    sp = sub.add_parser("manifest", help="print the declared source table")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_manifest)

    sp = sub.add_parser("verify", help="probe the provider for real coverage")
    add_datasets(sp)
    sp.add_argument("--sample", action="store_true",
                    help="also download one season per dataset to check schema and key uniqueness")
    sp.add_argument("--write-manifest", action="store_true",
                    help="write discovered coverage back into configs/data_sources.yaml")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("ingest", help="fetch and store raw data")
    add_datasets(sp)
    add_seasons(sp)
    sp.add_argument("--force", action="store_true", help="re-fetch partitions that already exist")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--required-only", action="store_true")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("validate", help="run quality checks against the lake")
    add_datasets(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("inventory", help="what is stored locally")
    sp.set_defaults(func=cmd_inventory)

    sp = sub.add_parser("doctor", help="end-to-end health check")
    sp.set_defaults(func=cmd_doctor)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(0 if args.quiet else args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
