""" GRASP algorithm """
from collections.abc import Hashable
from typing import List, Tuple
import random
import pandas as pd
from local_search import local_search_first_improvement


def ratio_val(rj: float, wj: int, alpha: float, lambda_w: float) -> float:
    """Ratio value for sorting items."""
    return rj / ((max(wj, 1) + lambda_w) ** alpha)


def grasp_construct(
    df: pd.DataFrame,
    max_weight: int,
    rn: random.Random,
    *,
    rcl_size: int = 20,
    alpha: float = 0.9,
    lambda_w: float = 0.5,
) -> tuple[list[Hashable], float, int]:
    """GRASP construction phase."""
    w = df["weight"].astype(int).to_dict()
    r = df["risk_score"].astype(float).to_dict()
    ids_all: list[Hashable] = list(df.index)

    chosen: list[Hashable] = []
    tw, ts = 0, 0.0
    remaining = set(ids_all)

    while True:
        feasible = [j for j in remaining if r[j] > 0
                    and (tw + w[j] <= max_weight)]
        if not feasible:
            break

        feasible.sort(
            key=lambda j: ratio_val(r[j], w[j], alpha, lambda_w),
            reverse=True,
        )
        rcl = feasible[: max(1, rcl_size)]
        j = rn.choice(rcl)

        chosen.append(j)
        tw += w[j]
        ts += r[j]
        remaining.remove(j)

    return chosen, ts, tw


def grasp(
    df: pd.DataFrame,
    max_weight: int,
    *,
    iterations: int = 50,
    seed: int = 42,
    rcl_size: int = 20,
    alpha: float = 0.9,
    lambda_w: float = 0.5,
    ls_imp: int = 2,
) -> Tuple[List[Hashable], float, int]:
    """GRASP algorithm with local search.Uses first-improvement local search.
    Returns the best solution found (choice, score, weight)."""

    rn = random.Random(seed)

    best_choice: list[Hashable] = []
    best_s: float = -1.0
    best_w: int = 10**9

    for _ in range(iterations):
        c0, _, _ = grasp_construct(
            df, max_weight, rn,
            rcl_size=rcl_size, alpha=alpha, lambda_w=lambda_w
        )

        c, s, w = local_search_first_improvement(
            df, max_weight,
            start_choice=c0,
            max_no_improve=ls_imp,
            alpha=alpha, lambda_w=lambda_w,
            rn=rn,
        )

        if (s > best_s) or (s == best_s and w < best_w):
            best_choice, best_s, best_w = c, s, w

    return best_choice, best_s, best_w
