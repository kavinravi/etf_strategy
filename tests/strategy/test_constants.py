from etf.strategy import constants as C

def test_universe_is_eleven_sectors():
    assert len(C.SECTOR_ETFS) == 11
    assert set(C.SECTOR_INCEPTION) == set(C.SECTOR_ETFS)

def test_cap_floor_are_feasible():
    n = len(C.SECTOR_ETFS)
    assert C.MIN_WEIGHT * n <= 1.0 <= C.MAX_WEIGHT * n  # a valid simplex exists
