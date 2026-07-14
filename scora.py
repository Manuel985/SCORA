from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pulp

from utilities import (
    AttackGraph,
    NodeId,
    EdgeId,
    CountermeasureCatalog,
    OptimizationConfig,
    ModelInput,
)


def worst_residual_path(
    graph: AttackGraph,
    log_p_edge: Dict[EdgeId, float],
    log_I: Dict[NodeId, float],
) -> Tuple[List[NodeId], List[EdgeId], float]:

    nodes = graph.nodes
    sources = graph.sources
    targets = graph.targets

    adjacency: Dict[NodeId, List[Tuple[NodeId, EdgeId, float]]] = {n: [] for n in nodes}

    for e_id, (u, v) in graph.edges.items():
        cost = -log_p_edge[e_id]
        if cost < 0.0:
            raise ValueError(
                f"Edge {e_id} has negative cost ({cost}); Dijkstra requires "
                "non-negative weights. Check p_edge / effectiveness bounds."
            )
        adjacency[u].append((v, e_id, cost))

    dist: Dict[NodeId, float] = {n: float("inf") for n in nodes}
    parent: Dict[NodeId, Optional[NodeId]] = {n: None for n in nodes}
    parent_edge: Dict[NodeId, Optional[EdgeId]] = {n: None for n in nodes}

    pq: List[Tuple[float, NodeId]] = []
    for s in sources:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, s))

    visited: Set[NodeId] = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v, e_id, w in adjacency[u]:
            if v in visited:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                parent_edge[v] = e_id
                heapq.heappush(pq, (nd, v))

    best_target = min(targets, key=lambda t: dist[t] - log_I[t])
    best_score = -(dist[best_target] - log_I[best_target])

    path_nodes: List[NodeId] = []
    path_edges: List[EdgeId] = []
    cur: Optional[NodeId] = best_target

    while cur is not None:
        path_nodes.append(cur)
        prev = parent[cur]
        if prev is not None:
            e_id = parent_edge[cur]
            assert e_id is not None
            path_edges.append(e_id)
        cur = prev

    path_nodes.reverse()
    path_edges.reverse()

    return path_nodes, path_edges, best_score


def compute_log_p_and_log_I(
    graph: AttackGraph,
    catalog: CountermeasureCatalog,
    x_vals: Dict[str, float],
) -> Tuple[Dict[EdgeId, float], Dict[NodeId, float]]:

    log_p_edge = graph.log_p_edge_base.copy()
    log_I = graph.log_I_base.copy()

    for cm in catalog.items:

        x = x_vals.get(cm.id, 0.0)
        if x <= 0.0:
            continue

        log_factor = cm.log_effectiveness

        if cm.scope.issubset(graph.edges):
            for e in cm.scope:
                log_p_edge[e] += x * log_factor

        else:
            for t in cm.scope:
                log_I[t] += x * log_factor

    return log_p_edge, log_I


def build_path_constraint_coeffs(
    path_nodes: Sequence[NodeId],
    path_edges: Sequence[EdgeId],
    target: NodeId,
    graph: AttackGraph,
    catalog: CountermeasureCatalog,
) -> Tuple[float, Dict[str, float]]:

    const_term = sum(graph.log_p_edge_base[e] for e in path_edges)
    const_term += graph.log_I_base[target]

    coeffs: Dict[str, float] = {}

    for e in path_edges:

        for cm in catalog.edge_to_cm.get(e, []):

            log_factor = cm.log_effectiveness
            coeffs[cm.id] = coeffs.get(cm.id, 0.0) + log_factor

    for cm in catalog.target_to_cm.get(target, []):

        log_factor = cm.log_effectiveness
        coeffs[cm.id] = coeffs.get(cm.id, 0.0) + log_factor

    return const_term, coeffs


def _apply_warm_start(
    x_vars: Dict[str, pulp.LpVariable],
    warm_start_x: Optional[Dict[str, float]],
) -> bool:

    if not warm_start_x:
        return False

    applied = False
    for cid, var in x_vars.items():
        val = warm_start_x.get(cid)
        if val is not None:
            var.setInitialValue(1 if val > 0.5 else 0)
            applied = True

    return applied


def _solve(prob: pulp.LpProblem, warm: bool) -> None:
    solver = pulp.PULP_CBC_CMD(msg=False, gapRel=0, gapAbs=0, warmStart=warm)
    prob.solve(solver)


def _cut_key(path_edges: Sequence[EdgeId]) -> Tuple[EdgeId, ...]:
    return tuple(path_edges)


def _add_cap_cut(prob, x_vars, cut, log_cap) -> None:
    coeffs = cut["coeffs"]
    prob += (
        cut["const_term"] + pulp.lpSum(coeffs[c] * x_vars[c] for c in coeffs) <= log_cap
    )


def _add_risk_cut(prob, x_vars, z, cut) -> None:
    coeffs = cut["coeffs"]
    prob += z >= cut["const_term"] + pulp.lpSum(coeffs[c] * x_vars[c] for c in coeffs)


def _solve_min_cost_given_risk_cap_and_budget(
    graph: AttackGraph,
    catalog: CountermeasureCatalog,
    budget: float,
    risk_cap: float,
    iter_max: int = 50,
    tol: float = 1e-6,
    seed_cap_cuts: Optional[List[Dict[str, object]]] = None,
    warm_start_x: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:

    log_R_cap = math.log(risk_cap)

    prob = pulp.LpProblem("MinCostGivenRiskCapAndBudget", pulp.LpMinimize)

    x_vars = {
        cm.id: pulp.LpVariable(
            f"x_{cm.id}", lowBound=1 if cm.implemented else 0, upBound=1, cat="Binary"
        )
        for cm in catalog.items
    }

    prob += pulp.lpSum(cm.cost * x_vars[cm.id] for cm in catalog.items)

    prob += pulp.lpSum(cm.cost * x_vars[cm.id] for cm in catalog.items) <= budget

    cap_cuts: List[Dict[str, object]] = list(seed_cap_cuts) if seed_cap_cuts else []
    for cut in cap_cuts:
        _add_cap_cut(prob, x_vars, cut, log_R_cap)

    active_paths: List[List[NodeId]] = []
    best_solution = None
    current_warm = warm_start_x

    for it in range(iter_max):

        warm_applied = _apply_warm_start(x_vars, current_warm)
        _solve(prob, warm_applied)

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        x_vals = {cid: var.value() or 0.0 for cid, var in x_vars.items()}
        current_warm = x_vals

        log_p, log_I = compute_log_p_and_log_I(graph, catalog, x_vals)
        path, path_edges, log_risk = worst_residual_path(graph, log_p, log_I)

        worst_risk = math.exp(log_risk)

        if worst_risk <= risk_cap + tol:

            best_solution = {
                "status": "Optimal",
                "selected_cms": [cm.id for cm in catalog.items if x_vals[cm.id] > 0.5],
                "x_values": x_vals,
                "total_cost": sum(cm.cost * x_vals[cm.id] for cm in catalog.items),
                "worst_path": path,
                "worst_path_risk": worst_risk,
                "active_paths": active_paths,
                "cap_cuts": cap_cuts,
            }
            break

        target = path[-1]

        const_term, coeffs = build_path_constraint_coeffs(
            path, path_edges, target, graph, catalog
        )

        key = _cut_key(path_edges)
        if key not in {c["key"] for c in cap_cuts}:
            cap_cuts.append({"key": key, "const_term": const_term, "coeffs": coeffs})

        prob += (
            const_term + pulp.lpSum(coeffs[c] * x_vars[c] for c in coeffs) <= log_R_cap
        )

        active_paths.append(list(path))

    if best_solution is None:

        best_solution = {
            "status": "IterLimit",
            "selected_cms": [],
            "x_values": {},
            "total_cost": None,
            "worst_path": None,
            "worst_path_risk": None,
            "active_paths": active_paths,
            "cap_cuts": cap_cuts,
        }

    return best_solution


def solve_min_cost_given_risk_threshold(
    model_input: ModelInput,
    iter_max: int = 50,
    tol: float = 1e-6,
    seed_cap_cuts: Optional[List[Dict[str, object]]] = None,
    seed_risk_cuts: Optional[List[Dict[str, object]]] = None,
    warm_start_x: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:

    graph = model_input.graph
    catalog = model_input.catalog
    config = model_input.config

    if config.risk_threshold is None:
        raise ValueError("risk_threshold not set")

    R_max = config.risk_threshold
    log_R_max = math.log(R_max)

    prob = pulp.LpProblem("MinCostUnderRiskThreshold", pulp.LpMinimize)

    x_vars = {
        cm.id: pulp.LpVariable(
            f"x_{cm.id}", lowBound=1 if cm.implemented else 0, upBound=1, cat="Binary"
        )
        for cm in catalog.items
    }

    prob += pulp.lpSum(cm.cost * x_vars[cm.id] for cm in catalog.items)

    cap_cuts: List[Dict[str, object]] = list(seed_cap_cuts) if seed_cap_cuts else []
    for cut in cap_cuts:
        _add_cap_cut(prob, x_vars, cut, log_R_max)

    active_paths: List[List[NodeId]] = []
    best_solution = None
    current_warm = warm_start_x

    for it in range(iter_max):

        warm_applied = _apply_warm_start(x_vars, current_warm)
        _solve(prob, warm_applied)

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        x_vals = {cid: var.value() or 0.0 for cid, var in x_vars.items()}
        current_warm = x_vals

        log_p, log_I = compute_log_p_and_log_I(graph, catalog, x_vals)
        path, path_edges, log_risk = worst_residual_path(graph, log_p, log_I)

        worst_risk = math.exp(log_risk)

        if worst_risk <= R_max + tol:

            best_solution = {
                "status": "Optimal",
                "selected_cms": [cm.id for cm in catalog.items if x_vals[cm.id] > 0.5],
                "x_values": x_vals,
                "total_cost": sum(cm.cost * x_vals[cm.id] for cm in catalog.items),
                "worst_path": path,
                "worst_path_risk": worst_risk,
                "active_paths": active_paths,
                "cap_cuts": cap_cuts,
            }
            break

        target = path[-1]

        const_term, coeffs = build_path_constraint_coeffs(
            path, path_edges, target, graph, catalog
        )

        key = _cut_key(path_edges)
        if key not in {c["key"] for c in cap_cuts}:
            cap_cuts.append({"key": key, "const_term": const_term, "coeffs": coeffs})

        prob += (
            const_term + pulp.lpSum(coeffs[c] * x_vars[c] for c in coeffs) <= log_R_max
        )

        active_paths.append(list(path))

    if best_solution is None:
        return {
            "status": "IterLimit",
            "selected_cms": [],
            "x_values": {},
            "total_cost": None,
            "worst_path": None,
            "worst_path_risk": None,
            "active_paths": active_paths,
            "cap_cuts": cap_cuts,
            "risk_cuts": list(seed_risk_cuts) if seed_risk_cuts else [],
        }

    cost_star = float(best_solution["total_cost"])

    lex_config = OptimizationConfig(risk_threshold=None, budget=cost_star)
    lex_input = ModelInput(graph=graph, catalog=catalog, config=lex_config)

    return solve_min_risk_given_budget(
        lex_input,
        iter_max,
        tol,
        seed_risk_cuts=seed_risk_cuts,
        seed_cap_cuts=cap_cuts,
        warm_start_x=best_solution["x_values"],
    )


def solve_min_risk_given_budget(
    model_input: ModelInput,
    iter_max: int = 50,
    tol: float = 1e-6,
    seed_risk_cuts: Optional[List[Dict[str, object]]] = None,
    seed_cap_cuts: Optional[List[Dict[str, object]]] = None,
    warm_start_x: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:

    graph = model_input.graph
    catalog = model_input.catalog
    config = model_input.config

    if config.budget is None:
        raise ValueError("budget not set")

    budget = config.budget

    prob = pulp.LpProblem("MinRiskGivenBudget", pulp.LpMinimize)

    x_vars = {
        cm.id: pulp.LpVariable(
            f"x_{cm.id}", lowBound=1 if cm.implemented else 0, upBound=1, cat="Binary"
        )
        for cm in catalog.items
    }

    z = pulp.LpVariable("z", lowBound=math.log(1e-12))

    prob += z

    prob += pulp.lpSum(cm.cost * x_vars[cm.id] for cm in catalog.items) <= budget

    risk_cuts: List[Dict[str, object]] = list(seed_risk_cuts) if seed_risk_cuts else []
    for cut in risk_cuts:
        _add_risk_cut(prob, x_vars, z, cut)

    active_paths: List[List[NodeId]] = []
    best_solution = None
    risk_cap = None
    current_warm = warm_start_x

    for it in range(iter_max):

        warm_applied = _apply_warm_start(x_vars, current_warm)
        _solve(prob, warm_applied)

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        x_vals = {cid: var.value() or 0.0 for cid, var in x_vars.items()}
        current_warm = x_vals

        log_p, log_I = compute_log_p_and_log_I(graph, catalog, x_vals)
        path, path_edges, log_risk = worst_residual_path(graph, log_p, log_I)

        worst_risk = math.exp(log_risk)

        z_val = z.value()
        if z_val is not None and z_val + tol >= log_risk:

            best_solution = {
                "status": "Optimal",
                "selected_cms": [cm.id for cm in catalog.items if x_vals[cm.id] > 0.5],
                "x_values": x_vals,
                "total_cost": sum(cm.cost * x_vals[cm.id] for cm in catalog.items),
                "worst_path": path,
                "worst_path_risk": worst_risk,
                "active_paths": active_paths,
                "risk_cuts": risk_cuts,
            }

            risk_cap = worst_risk
            break

        target = path[-1]

        const_term, coeffs = build_path_constraint_coeffs(
            path, path_edges, target, graph, catalog
        )

        key = _cut_key(path_edges)
        if key not in {c["key"] for c in risk_cuts}:
            risk_cuts.append({"key": key, "const_term": const_term, "coeffs": coeffs})

        prob += z >= const_term + pulp.lpSum(coeffs[c] * x_vars[c] for c in coeffs)

        active_paths.append(list(path))

    if best_solution is None:
        return {
            "status": "IterLimit",
            "selected_cms": [],
            "x_values": {},
            "total_cost": None,
            "worst_path": None,
            "worst_path_risk": None,
            "active_paths": active_paths,
            "risk_cuts": risk_cuts,
            "cap_cuts": list(seed_cap_cuts) if seed_cap_cuts else [],
        }

    lex = _solve_min_cost_given_risk_cap_and_budget(
        graph,
        catalog,
        budget,
        risk_cap,
        iter_max,
        tol,
        seed_cap_cuts=seed_cap_cuts,
        warm_start_x=best_solution["x_values"],
    )
    lex["risk_cuts"] = risk_cuts

    return lex
