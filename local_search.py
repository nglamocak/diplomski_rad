"""Local search algorithms for the knapsack problem (women)."""
from collections.abc import Hashable
from itertools import combinations
from typing import List
import random
import pandas as pd


def ratio(rj: float, wj: int, alpha: float, lambda_w: float) -> float:
    """Ratio value for sorting items."""
    return rj / ((max(wj, 1) + lambda_w) ** alpha)  # alpha=kazna
# lambda pomak da ne uzme premale tezine uvijek


def first_improvement_step(
    chosen: set[Hashable],
    total_w: int,
    total_s: float,
    max_weight: int,
    weights: dict[Hashable, int],
    scores: dict[Hashable, float],
    in_order: List[Hashable],
    out_order: List[Hashable],
    alpha: float,
    lambda_w: float,
    top_out: int = 40,
) -> tuple[set[Hashable], int, float, bool]:
    """One step of first-improvement local search with moves +,
    1 swap 1, 1 swap 2, 2 swap 1."""
    w, r = weights, scores

    out_sorted = sorted(
        out_order,
        key=lambda j: ratio(r[j], w[j], alpha, lambda_w),
        reverse=True,
    )[:top_out]

    for j in out_sorted:
        wj, rj = w[j], r[j]
        if rj <= 0:
            continue
        if total_w + wj <= max_weight:
            chosen.add(j)
            return chosen, total_w + wj, total_s + rj, True
    # 1 swap 1
    for i in in_order:
        wi, ri = w[i], r[i]
        for j in out_sorted:
            wj, rj = w[j], r[j]
            if rj <= ri:
                continue
            if total_w - wi + wj <= max_weight:
                chosen.remove(i)
                chosen.add(j)
                return chosen, total_w - wi + wj, total_s - ri + rj, True
    # 1 swap 2
    for i in in_order:
        wi, ri = w[i], r[i]
        for j, k in combinations(out_sorted, 2):
            wjk = w[j] + w[k]
            rjk = r[j] + r[k]
            if rjk <= ri:
                continue
            if total_w - wi + wjk <= max_weight:
                chosen.remove(i)
                chosen.update([j, k])
                return chosen, total_w - wi + wjk, total_s - ri + rjk, True

    in_list = list(in_order)  # korisno kad želiš više prolaza, indeksiranje
    # ili kombinacije nad istim skupom elemenata.
    # 2 swap 1
    for i, u in combinations(in_list, 2):
        wiu = w[i] + w[u]
        riu = r[i] + r[u]
        for j in out_sorted:
            wj, rj = w[j], r[j]
            if rj <= riu:
                continue
            if total_w - wiu + wj <= max_weight:
                chosen.remove(i)
                chosen.remove(u)
                chosen.add(j)
                return chosen, total_w - wiu + wj, total_s - riu + rj, True

    return chosen, total_w, total_s, False


def local_search_first_improvement(
    df: pd.DataFrame,
    max_weight: int,
    start_choice: list[Hashable] | None = None,
    max_no_improve: int = 3,
    alpha: float = 0.9,
    lambda_w: float = 0.5,
    top_out: int = 40,
    rn: random.Random | None = None,
) -> tuple[list[Hashable], float, int]:
    """Local search with first-improvement strategy.
    Uses moves +, 1 swap 1, 1 swap 2, 2 swap 1.
    If start_choice is None, a greedy solution is used
    as the starting point."""
    if rn is None:
        rn = random.Random(0)

    w = df["weight"].astype(int).to_dict()
    r = df["risk_score"].astype(float).to_dict()
    all_ids: list[Hashable] = list(df.index)

    if start_choice is None:
        ids = sorted(
            all_ids,
            key=lambda j: ratio(r[j], w[j], alpha, lambda_w),
            reverse=True,
        )
        chosen: set[Hashable] = set()
        tw, ts = 0, 0.0
        for j in ids:
            if r[j] <= 0:
                continue
            if tw + w[j] <= max_weight:
                chosen.add(j)
                tw += w[j]
                ts += r[j]
    else:
        chosen = set(start_choice)
        tw = sum(w[i] for i in chosen)
        ts = sum(r[i] for i in chosen)

        if tw > max_weight:
            ids = sorted(
                chosen,
                key=lambda j: ratio(r[j], w[j], alpha, lambda_w),
                reverse=True,
            )
            chosen, tw, ts = set(), 0, 0.0
            for j in ids:
                if tw + w[j] <= max_weight:
                    chosen.add(j)
                    tw += w[j]
                    ts += r[j]

    no_improve = 0
    while no_improve < max_no_improve:
        improved_ = False

        inside = list(chosen)
        outside = [i for i in all_ids if i not in chosen]
        rn.shuffle(inside)  # random pretraga susjedstva
        rn.shuffle(outside)

        while True:
            chosen, tw, ts, improved = first_improvement_step(
                chosen, tw, ts, max_weight, w, r,
                inside, outside, alpha, lambda_w, top_out=top_out
            )
            if not improved:
                break
            improved_ = True

            inside = list(chosen)
            outside = [i for i in all_ids if i not in chosen]
            rn.shuffle(inside)
            rn.shuffle(outside)

        if improved_:
            no_improve = 0
        else:
            no_improve += 1

    return list(chosen), ts, tw
