import pytest
import pandas as pd

from etf.data.ingest_macro import align_macro_monthly, download_macro


def test_align_takes_last_value_at_or_before_month_end():
    daily = pd.DataFrame(
        {"mac_vixcls": [15.0, 16.0, 30.0]},
        index=pd.to_datetime(["2020-01-15", "2020-01-31", "2020-02-20"]),
    )
    month_ends = pd.to_datetime(["2020-01-31", "2020-02-29"])
    out = align_macro_monthly(daily, month_ends)
    assert list(out.index) == list(month_ends)
    assert out["mac_vixcls"].iloc[0] == 16.0  # value on Jan-31
    assert out["mac_vixcls"].iloc[1] == 30.0  # last <= Feb-29 is Feb-20


@pytest.mark.live
def test_download_macro_live():
    df = download_macro(start="2020-01-01")
    assert "mac_vixcls" in df.columns and len(df) > 200
