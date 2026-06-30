from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Walk:
    train_end_year: int
    val_year: int
    test_year: int
    def masks(self, dates: pd.Series):
        y = pd.to_datetime(dates).dt.year
        return (y <= self.train_end_year, y == self.val_year, y == self.test_year)

def expanding_walks(first_test_year: int, last_test_year: int) -> list[Walk]:
    return [Walk(ty - 2, ty - 1, ty) for ty in range(first_test_year, last_test_year + 1)]
