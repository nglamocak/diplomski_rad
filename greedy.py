"""Greedy approach for the 0-1 knapsack problem (women)."""
from collections.abc import Hashable
from typing import List, Tuple
import pandas as pd


def greedy_approach(
    df: pd.DataFrame,
    max_weight: int,
) -> Tuple[List[Hashable], float, int]:
    """Greedy approach for the 0-1 knapsack problem.
    Select items based on the highest ratio of risk_score to weight.
    """
    temp = df.copy()
    temp["ratio"] = temp["risk_score"] / temp["weight"]

    temp = temp.sort_values(
        by=["ratio", "risk_score", "weight"],
        ascending=[False, False, True],
    )

    choice: List[Hashable] = []
    total_w = 0
    total_s = 0.0

    for idx, row in temp.iterrows():
        w = int(row["weight"])
        r = float(row["risk_score"])
        if total_w + w <= max_weight:
            choice.append(idx)
            total_w += w
            total_s += r

    return choice, total_s, total_w
