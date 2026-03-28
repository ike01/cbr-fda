# merge_graph.py
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
import networkx as nx

from graph_structures import Case, SolutionGraph
from retrieval import Retrieved

logger = logging.getLogger("merge_graph")


def build_action_sequence_graph(actions_ordered: List[str]) -> SolutionGraph:
    """
    Build a simple sequence graph from an ordered list of action labels.
    """
    sg = SolutionGraph()
    for i, lab in enumerate(actions_ordered):
        sg.add_node(i, lab)
    for i in range(len(actions_ordered) - 1):
        sg.add_edge(i, i + 1, label="sequence")
    return sg


def merge_matched_actions_into_graph(
    matched_actions: List[str],
    retrieved: List[Retrieved],
    case_base: List[Case],
    debug: bool = False
) -> Tuple[SolutionGraph, Dict[str, float], List[str]]:
    """
    Graph merge idea (MultiWOZ-friendly and domain-light):

    1) We choose an ordering for matched actions using similarity-weighted
       average position across retrieved neighbours (that contain the action).
    2) We create a sequence graph using that merged ordering.
    3) We also compute pairwise order support scores (debuggable),
       but we keep the final graph simple (sequence).

    Returns:
      - merged SolutionGraph
      - action_avg_pos: similarity-weighted avg position per action
      - merged_order: ordered action labels
    """
    if not matched_actions:
        return build_action_sequence_graph([]), {}, []

    # Compute similarity-weighted average position for each action
    pos_sum: Dict[str, float] = {a: 0.0 for a in matched_actions}
    pos_w: Dict[str, float] = {a: 0.0 for a in matched_actions}

    # Pairwise order support: support[a,b] = weighted count where a appears before b
    support: Dict[Tuple[str, str], float] = {}

    for r in retrieved:
        c = case_base[r.idx]
        w = max(r.score, 0.0)
        seq = c.solution_actions
        idx_map = {lab: i for i, lab in enumerate(seq)}

        # Only consider actions that appear in this neighbour
        present = [a for a in matched_actions if a in idx_map]
        for a in present:
            pos_sum[a] += w * idx_map[a]
            pos_w[a] += w

        # pairwise support from this neighbour's ordering
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a = present[i]
                b = present[j]
                if idx_map[a] < idx_map[b]:
                    support[(a, b)] = support.get((a, b), 0.0) + w
                else:
                    support[(b, a)] = support.get((b, a), 0.0) + w

    action_avg_pos: Dict[str, float] = {}
    for a in matched_actions:
        if pos_w[a] > 0:
            action_avg_pos[a] = pos_sum[a] / pos_w[a]
        else:
            action_avg_pos[a] = 1e9  # if unseen, push to end

    # Merge ordering: sort by avg position
    merged_order = sorted(matched_actions, key=lambda a: action_avg_pos[a])

    # Build a simple sequence graph
    sg = build_action_sequence_graph(merged_order)

    if debug:
        logger.info("Merged order: %s", merged_order)
        # show top pairwise supports
        top_edges = sorted(support.items(), key=lambda kv: -kv[1])[:10]
        for (a, b), s in top_edges:
            logger.info("Order support: %s  ->  %s   weight=%.3f", a, b, s)

    return sg, action_avg_pos, merged_order