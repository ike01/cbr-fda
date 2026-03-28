# graph_structures.py
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import networkx as nx


@dataclass
class Case:
    """
    Generic CBR case.

    - problem_text: used for retrieval (TF-IDF or SBERT).
    - needs: decomposed problem elements used for stable matching.
    - solution_actions: ordered list of atomic solution components (nodes).
    """
    case_id: str
    problem_text: str
    needs: List[str]
    service: str
    solution_actions: List[str]


class SolutionGraph:
    """
    Thin wrapper around a directed graph with node/edge labels.
    """
    def __init__(self):
        self.G = nx.DiGraph()

    def add_node(self, node_id: int, label: str, **attrs):
        self.G.add_node(node_id, label=label, **attrs)

    def add_edge(self, u: int, v: int, label: str = "sequence", **attrs):
        self.G.add_edge(u, v, label=label, **attrs)

    def node_labels(self) -> List[str]:
        return [self.G.nodes[n].get("label", "") for n in self.G.nodes]

    def edges_labeled(self) -> List[Tuple[str, str, str]]:
        out = []
        for u, v, d in self.G.edges(data=True):
            lu = self.G.nodes[u].get("label", "")
            lv = self.G.nodes[v].get("label", "")
            out.append((lu, lv, d.get("label", "relation")))
        return out

    def __len__(self):
        return self.G.number_of_nodes()