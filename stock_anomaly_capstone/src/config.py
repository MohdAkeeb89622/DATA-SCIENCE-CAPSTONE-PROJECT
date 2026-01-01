from dataclasses import dataclass

@dataclass(frozen=True)
class Windows:
    w_return: int = 63
    w_volume: int = 21
    w_range: int = 63

@dataclass(frozen=True)
class Thresholds:
    ret_z: float = 2.5
    volz: float = 2.5
    range_pct: float = 95.0
    market_breadth: float = 0.30
    market_ret_pct: float = 95.0  # percentile of |market_ret| over rolling history

DEFAULT_UNIVERSE = ["QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META"]
