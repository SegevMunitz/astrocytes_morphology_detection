"""Feature-table aggregation helpers."""

from collections.abc import Sequence

import pandas as pd


def aggregate_feature_table(table: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Average numeric fields by explicit experimental grouping columns."""
    missing = [column for column in group_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Grouping columns are missing: {missing}")
    return table.groupby(list(group_columns), dropna=False).mean(numeric_only=True).reset_index()

