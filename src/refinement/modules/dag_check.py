"""
DAG checking utilities for variable-level directed graphs.

We intentionally keep this dependency-free (no networkx) and O(V+E).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


Edge = Tuple[str, str]


def is_dag_kahn(nodes: Iterable[str], edges: Iterable[Edge]) -> bool:
    """
    Returns True iff the directed graph is acyclic (DAG), using Kahn's algorithm.
    Complexity: O(V+E).
    """
    nodes_set: Set[str] = set(nodes)
    g: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = {n: 0 for n in nodes_set}

    for u, v in edges:
        nodes_set.add(u)
        nodes_set.add(v)
        g[u].append(v)
        indeg[v] = indeg.get(v, 0) + 1
        indeg.setdefault(u, 0)

    q = deque([n for n, d in indeg.items() if d == 0])
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in g.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == len(indeg)


def find_one_cycle(nodes: Iterable[str], edges: Iterable[Edge]) -> Optional[List[str]]:
    """
    If graph has a directed cycle, returns one cycle as a list of nodes like:
      [A, B, C, A]
    Else returns None.
    """
    nodes_set: Set[str] = set(nodes)
    g: Dict[str, List[str]] = defaultdict(list)
    for u, v in edges:
        nodes_set.add(u)
        nodes_set.add(v)
        g[u].append(v)

    # 0=unseen, 1=visiting, 2=done
    color: Dict[str, int] = {n: 0 for n in nodes_set}
    parent: Dict[str, Optional[str]] = {}

    def dfs(u: str) -> Optional[List[str]]:
        color[u] = 1
        for v in g.get(u, []):
            if color.get(v, 0) == 0:
                parent[v] = u
                cyc = dfs(v)
                if cyc:
                    return cyc
            elif color.get(v) == 1:
                # back edge u->v
                path = [v]
                cur = u
                while cur != v and cur in parent and parent[cur] is not None:
                    path.append(cur)
                    cur = parent[cur]  # type: ignore[assignment]
                path.append(v)
                path.reverse()
                return path
        color[u] = 2
        return None

    for n in nodes_set:
        if color.get(n, 0) == 0:
            parent[n] = None
            cyc = dfs(n)
            if cyc:
                return cyc
    return None


def cycle_to_string(cycle: Sequence[str]) -> str:
    return " -> ".join(cycle)


@dataclass(frozen=True)
class DagProjectionCut:
    u: str
    v: str
    strength: float
    cycle: List[str]  # e.g. [A,B,C,A]


@dataclass(frozen=True)
class DagProjectionResult:
    is_dag: bool
    cuts: List[DagProjectionCut]
    num_nodes: int
    num_edges_in: int
    num_edges_out: int


def project_to_dag_cut_weakest_in_cycle(
    *,
    edges: Set[Edge],
    edge_strength: Callable[[str, str], float],
    nodes: Optional[Set[str]] = None,
    max_iters: int = 10_000,
) -> Tuple[Set[Edge], DagProjectionResult]:
    """
    Greedy DAG projection:
      while there exists a directed cycle:
        find one cycle
        cut the weakest edge (min strength) on that cycle

    Complexity is usually small in sparse graphs; each iteration removes one edge.
    """
    edges_cur: Set[Edge] = set(edges)
    if nodes is None:
        nodes_set: Set[str] = set()
        for u, v in edges_cur:
            nodes_set.add(u)
            nodes_set.add(v)
    else:
        nodes_set = set(nodes)

    cuts: List[DagProjectionCut] = []
    for _ in range(int(max_iters)):
        if is_dag_kahn(nodes_set, edges_cur):
            res = DagProjectionResult(
                is_dag=True,
                cuts=cuts,
                num_nodes=len(nodes_set),
                num_edges_in=len(edges),
                num_edges_out=len(edges_cur),
            )
            return edges_cur, res

        cyc = find_one_cycle(nodes_set, edges_cur)
        if not cyc or len(cyc) < 2:
            break

        # cycle nodes list ends with start node: [A,B,C,A]
        cyc_edges: List[Edge] = [(cyc[i], cyc[i + 1]) for i in range(len(cyc) - 1)]
        weakest = None
        for u, v in cyc_edges:
            s = float(edge_strength(u, v))
            if weakest is None or s < weakest[2]:
                weakest = (u, v, s)
        if weakest is None:
            break
        u, v, s = weakest
        if (u, v) not in edges_cur:
            break
        edges_cur.remove((u, v))
        cuts.append(DagProjectionCut(u=u, v=v, strength=float(s), cycle=list(cyc)))

    # Failed to reach DAG within max_iters
    res = DagProjectionResult(
        is_dag=is_dag_kahn(nodes_set, edges_cur),
        cuts=cuts,
        num_nodes=len(nodes_set),
        num_edges_in=len(edges),
        num_edges_out=len(edges_cur),
    )
    return edges_cur, res

