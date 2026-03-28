# evaluation.py
from typing import List, Tuple, Set
from graph_structures import SolutionGraph


def f1_pr(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return f1, p, r


def edges_from_sequence(actions: List[str]) -> Set[Tuple[str, str]]:
    """
    Gold edges in MultiWOZ are not explicitly annotated.
    We assume within-turn action order is a sequence.
    """
    e = set()
    for i in range(len(actions) - 1):
        e.add((actions[i], actions[i + 1]))
    return e


def eval_solution_graph(pred_g: SolutionGraph, gold_actions_ordered: List[str]) -> dict:
    """
    Evaluate both:
      - Node accuracy (action set F1)
      - Edge accuracy (sequence edges F1), based on within-turn ordering assumption
    """
    pred_nodes = set(pred_g.node_labels())
    gold_nodes = set(gold_actions_ordered)
    node_f1, node_p, node_r = f1_pr(pred_nodes, gold_nodes)

    pred_edges = set((a, b) for (a, b, _) in pred_g.edges_labeled())
    gold_edges = edges_from_sequence(gold_actions_ordered)
    edge_f1, edge_p, edge_r = f1_pr(pred_edges, gold_edges)

    return {
        "node_f1": node_f1, "node_p": node_p, "node_r": node_r,
        "edge_f1": edge_f1, "edge_p": edge_p, "edge_r": edge_r,
        "pred_nodes": len(pred_nodes), "gold_nodes": len(gold_nodes),
        "pred_edges": len(pred_edges), "gold_edges": len(gold_edges),
    }