"""Feature-table aggregation helpers."""

from collections.abc import Sequence

import pandas as pd


def aggregate_feature_table(table: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Average numeric features by explicit experimental grouping columns.

    Group names must already exist in the input table; the function never infers
    wells or replicates from filenames. One mean row is returned per group.
    """
    missing = [column for column in group_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Grouping columns are missing: {missing}")
    return table.groupby(list(group_columns), dropna=False).mean(numeric_only=True).reset_index()
