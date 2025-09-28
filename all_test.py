"""Testing all algorithms on the dataset."""
from collections.abc import Sequence, Hashable
import pandas as pd
from data import prepare_dataset
from greedy import greedy_approach
from local_search import local_search_first_improvement
from simulated_annealing import simulated_annealing
from grasp import grasp


def evaluate_selection(data: pd.DataFrame, result: Sequence[Hashable]) -> dict:
    """Evaluate selection against 'Violence' (just for reporting)."""
    if "Violence" not in data.columns:
        return {"note": "Violence column not found."}

    idx = list(result)
    if len(idx) == 0:
        return {
            "selected": 0, "selected_yes": 0,
            "base_yes":
                int((data["Violence"].astype(str).str.lower() == "yes").sum()),
            "precision": 0.0, "recall": 0.0, "lift": 0.0,
            "base_rate":
                float((
                    data["Violence"].astype(str).str.lower() == "yes").mean()),
        }

    sel = data.loc[idx]
    sel_yes = int((sel["Violence"].astype(str).str.lower() == "yes").sum())
    base_yes = int((data["Violence"].astype(str).str.lower() == "yes").sum())

    precision = sel_yes / len(sel) if len(sel) else 0.0
    recall = sel_yes / base_yes if base_yes else 0.0
    base_rate = base_yes / len(data) if len(data) else 0.0
    lift = (precision / base_rate) if base_rate > 0 else 0.0

    return {
        "selected": int(len(sel)),
        "selected_yes": sel_yes,
        "base_yes": base_yes,
        "precision": precision,
        "recall": recall,
        "lift": lift,
        "base_rate": base_rate,
    }


def yes_at_k(data: pd.DataFrame, choice, ks=(10, 15, 25)) -> dict:
    """Count 'Violence' = 'yes' in top-k selected."""
    out = {}
    y = data["Violence"].astype(str).str.lower().eq("yes")
    idx = list(choice)
    for k in ks:
        topk = idx[:k]
        if not topk:
            out[k] = {"yes": 0, "rate": 0.0}
            continue
        cnt = int(y.loc[pd.Index(topk)].sum())
        out[k] = {"yes": cnt, "rate": cnt / len(topk)}
    return out


def print_top(
    data: pd.DataFrame,
    choice: Sequence[Hashable],
    n: int,
    title: str,
) -> None:
    """Print top-n selected individuals."""
    cols = [
        "Age",
        "Education",
        "Employment",
        "Income",
        "Marital status",
        "risk_score",
        "weight",
        "Violence",
    ]
    cols = [c for c in cols if c in data.columns]

    ids = list(choice)[:n]
    if not ids:
        print(f"\nFirst {n} selected ({title}): <empty>")
        return

    print(f"\nFirst {n} selected ({title}):")
    print(data.loc[pd.Index(ids), cols].to_string(index=True))


def print_summary(
    name: str,
    weight: int,
    score: float,
    selected: int,
    evald: dict,
) -> None:
    """Print summary of the results."""
    score_out: float | int = int(score) if score.is_integer() else score

    summary = {
        "weight": weight,
        "score": score_out,
        "selected": selected,
    }

    eval_short = {
        "selected": int(evald["selected"]),
        "selected_yes": int(evald["selected_yes"]),
        "precision": round(float(evald["precision"]), 3),
        "recall": round(float(evald["recall"]), 3),
        "lift": round(float(evald["lift"]), 3),
    }

    print(f"{name:6s}: {summary}")
    print(f"Evaluation ({name}): {eval_short}")


if __name__ == "__main__":

    W = 51
    N = 25
    SHOW_YES = (15, 35, 55)

    df = prepare_dataset("data.csv")

    # Greedy
    g_choice, g_score, g_weight = greedy_approach(df, W)
    g_eval = evaluate_selection(df, g_choice)
    print_summary("Greedy", g_weight, g_score, len(g_choice), g_eval)
    print_top(df, g_choice, N, "Greedy")
    print("Greedy Yes", yes_at_k(df, g_choice, SHOW_YES))
    print("\n")

    # Local search (start = Greedy)
    ls_choice, ls_score, ls_weight = local_search_first_improvement(
        df, W, start_choice=g_choice,
        max_no_improve=3,
        alpha=0.9, lambda_w=0.5
    )
    ls_eval = evaluate_selection(df, ls_choice)
    print_summary("LS", ls_weight, ls_score, len(ls_choice), ls_eval)
    print_top(df, ls_choice, N, "Local Search")
    print("LS Yes", yes_at_k(df, ls_choice, SHOW_YES))
    print("\n")

    # Simulated annealing (start = Greedy)
    sa_choice, sa_score, sa_weight = simulated_annealing(
        df, W, start_choice=g_choice,
        T0=10.0, Tmin=1e-3, alpha=0.97, iters_per_T=120, seed=0
    )
    sa_eval = evaluate_selection(df, sa_choice)
    print_summary("SA", sa_weight, sa_score, len(sa_choice), sa_eval)
    print_top(df, sa_choice, N, "Simulated Annealing")
    print("SA Yes", yes_at_k(df, sa_choice, SHOW_YES))
    print("\n")

    # GRASP
    gr_choice, gr_score, gr_weight = grasp(
        df, W,
        iterations=100,
        rcl_size=25,
        alpha=0.9, lambda_w=0.5
    )

    gr_eval = evaluate_selection(df, gr_choice)
    print_summary("GRASP", gr_weight, gr_score, len(gr_choice), gr_eval)
    print_top(df, gr_choice, N, "GRASP")
    print("GRASP Yes", yes_at_k(df, gr_choice, SHOW_YES))
