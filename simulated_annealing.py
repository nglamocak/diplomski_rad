"""Simulated Annealing for 0-1 knapsack (women)."""
from collections.abc import Hashable
from typing import Iterable, Tuple, List
import random
import math
import pandas as pd


def ratio(r: dict, w: dict, j: Hashable) -> float:
    """Risk/weight ratio, with weight at least 1 to avoid div by 0."""
    return float(r[j]) / max(int(w[j]), 1)


def greedy_like_outside(outside: list[Hashable], r: dict,
                        w: dict, k: int) -> list[Hashable]:
    """Top-k outside by ratio."""
    return sorted(outside, key=lambda j: ratio(r, w, j), reverse=True)[:k]


def worst_inside(chosen: list[Hashable], r: dict, w: dict) -> Hashable | None:
    """Worst inside by ratio (or None if empty)."""
    if not chosen:
        return None
    return min(chosen, key=lambda i: ratio(r, w, i))


def simulated_annealing(
    df: pd.DataFrame,
    max_weight: int,
    *,
    start_choice: Iterable[Hashable] | None = None,
    seed: int = 42,
    T0: float = 10.0,
    Tmin: float = 1e-3,
    alpha: float = 0.97,
    iters_per_T: int = 120,
    top_k: int = 40,
    patience_temps: int = 3,
) -> Tuple[List[Hashable], float, int]:
    """Simulated Annealing for 0-1 knapsack problem (women).
    Uses moves + and 1 swap 1.
    If start_choice is None, a greedy solution is used as the starting point.
    Returns (choice, score, weight)."""

    rn = random.Random(seed)
    w = df["weight"].astype(int).to_dict()
    r = df["risk_score"].astype(float).to_dict()
    all_ids = list(df.index)

    if start_choice is None:
        ids = sorted(all_ids, key=lambda j: ratio(r, w, j), reverse=True)
        chosen: set[Hashable] = set()
        tw, ts = 0, 0.0
        for j in ids:
            wj, rj = w[j], r[j]
            if rj <= 0:
                continue
            if tw + wj <= max_weight:
                chosen.add(j)
                tw += wj
                ts += rj
    else:
        chosen = set(start_choice)
        tw = sum(w[i] for i in chosen)
        ts = sum(r[i] for i in chosen)
        if tw > max_weight:
            for i in sorted(list(chosen), key=lambda j: ratio(r, w, j)):
                if tw <= max_weight:
                    break
                chosen.discard(i)
                tw -= w[i]
                ts -= r[i]

    best = list(chosen)
    best_w, best_s = tw, ts

    temp = T0
    temps_no_accept = 0

    while temp > Tmin:
        accept_temp = False
        for _ in range(iters_per_T):
            inside = list(chosen)
            outside = [j for j in all_ids if j not in chosen and r[j] > 0]
            outside_top = greedy_like_outside(outside, r, w, top_k)
            worst_in = worst_inside(inside, r, w)

            do_add = (rn.random() < 0.5) or (worst_in is None)

            cand_choice: set[Hashable] | None = None
            cand_s = ts
            cand_w = tw

            if do_add and outside_top:
                for j in outside_top:
                    wj, rj = w[j], r[j]
                    if tw + wj <= max_weight:
                        cand_choice = set(chosen)
                        cand_choice.add(j)
                        cand_s = ts + rj
                        cand_w = tw + wj
                        break
            elif (not do_add) and worst_in is not None and outside_top:
                i = worst_in
                wi, ri = w[i], r[i]
                for j in outside_top:
                    wj, rj = w[j], r[j]
                    new_w = tw - wi + wj
                    if new_w <= max_weight:
                        cand_choice = set(chosen)
                        cand_choice.discard(i)
                        cand_choice.add(j)
                        cand_s = ts - ri + rj
                        cand_w = new_w
                        break

            if cand_choice is None:
                continue

            dE = ts - cand_s
            if dE <= 0 or rn.random() < math.exp(-dE / temp):
                chosen = cand_choice
                ts, tw = cand_s, cand_w
                accept_temp = True

                if (ts > best_s) or (ts == best_s and tw < best_w):
                    best, best_s, best_w = list(chosen), ts, tw
        temp *= alpha
        if accept_temp:
            temps_no_accept = 0
        else:
            temps_no_accept += 1
            if temps_no_accept >= patience_temps:
                break

    return list(best), float(best_s), int(best_w)
