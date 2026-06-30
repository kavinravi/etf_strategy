SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]

# First full month each ETF traded (used to build the expanding universe).
SECTOR_INCEPTION = {
    "XLB": "1998-12-31", "XLE": "1998-12-31", "XLF": "1998-12-31",
    "XLI": "1998-12-31", "XLK": "1998-12-31", "XLP": "1998-12-31",
    "XLU": "1998-12-31", "XLV": "1998-12-31", "XLY": "1998-12-31",
    "XLRE": "2015-10-31", "XLC": "2018-06-30",
}
BENCHMARK = "SPY"

MAX_WEIGHT = 0.18
MIN_WEIGHT = 0.02
NO_TRADE_BAND = 0.02
TILT_SCALE = 0.5
COST_BPS = 5.0
PERIODS_PER_YEAR = 12
