"""Source adapters. One module per provider family."""

from . import coaching, nflverse, stadiums, weather  # noqa: F401

__all__ = ["nflverse", "stadiums", "coaching", "weather"]
