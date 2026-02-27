"""
V-structure hard-mask postprocessing (scheme A)

This module provides a small, reusable interface to:
  1) infer collider (v-structure) constraints from a PAG-style CSV (FCI/RFCI outputs)
  2) apply them as a hard mask on a learned *state-level* adjacency matrix
     by zeroing the entire reverse block (child -> parent) for forced edges.

It is intentionally conservative:
  - It does NOT boost the forward direction weights.
  - It only forbids the reverse direction for inferred collider parents.

This is designed to be plugged into evaluation or as an optional postprocess step
after training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch

# Local imports are optional; keep module usable in analysis scripts.
try:
    from modules.data_loader import CausalDataLoader
    from modules.evaluator import CausalGraphEvaluator
except Exception:  # pragma: no cover
    CausalDataLoader = None
    CausalGraphEvaluator = None


ARROW_IN_TYPES = {"partial", "directed", "bidirected"}


@dataclass(frozen=True)
class VStructureMaskStats:
    """Lightweight stats for debugging/reporting."""

    fci_csv_path: str
    forced_edges_total: int
    forced_edges_in_structure: int
    blocks_zeroed: int
    # Optional: when applied to skeleton mask we track how many directed blocks were forbidden.
    skeleton_blocks_forbidden: int = 0


def load_pag_edges_csv(fci_csv_path: str) -> pd.DataFrame:
    """
    Load a PAG-style edges CSV with at least: source, target, edge_type.
    """
    p = Path(fci_csv_path)
    if not p.exists():
        raise FileNotFoundError(fci_csv_path)

    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    if "source" not in cols or "target" not in cols:
        raise ValueError(f"FCI/RFCI CSV missing source/target columns: {list(df.columns)}")
    if "edge_type" not in cols:
        raise ValueError(
            f"FCI/RFCI CSV missing edge_type column: {list(df.columns)}. "
            f"V-structure inference requires PAG edge types."
        )

    df = df.rename(
        columns={
            cols["source"]: "source",
            cols["target"]: "target",
            cols["edge_type"]: "edge_type",
        }
    )
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["edge_type"] = df["edge_type"].astype(str)
    return df[["source", "target", "edge_type"]]


def build_skeleton_pairs(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for s, t in zip(df["source"].tolist(), df["target"].tolist()):
        a, b = (s, t) if s < t else (t, s)
        pairs.add((a, b))
    return pairs


def build_arrow_in_map(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    target -> set(sources) where row encodes an arrowhead into target.

    We treat any edge_type in ARROW_IN_TYPES as implying an arrowhead at the target end.
    """
    incoming: Dict[str, Set[str]] = {}
    for s, t, et in zip(df["source"].tolist(), df["target"].tolist(), df["edge_type"].tolist()):
        et = et.lower().strip()
        if et in ARROW_IN_TYPES:
            incoming.setdefault(t, set()).add(s)
    return incoming


def infer_vstructure_forced_edges(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    """
    Infer forced directed edges (parent, child) from unshielded colliders.

    If we see two distinct incoming arrowheads into z: x *-> z and y *-> z
    and x,y are NOT adjacent in the skeleton, then enforce x->z and y->z.
    """
    skeleton = build_skeleton_pairs(df)
    incoming = build_arrow_in_map(df)

    forced: Set[Tuple[str, str]] = set()
    for z, pars in incoming.items():
        if len(pars) < 2:
            continue
        pars_list = sorted(list(pars))
        for i in range(len(pars_list)):
            for j in range(i + 1, len(pars_list)):
                x = pars_list[i]
                y = pars_list[j]
                a, b = (x, y) if x < y else (y, x)
                # Unshielded triple requirement: x and y not adjacent in skeleton
                if (a, b) in skeleton:
                    continue
                forced.add((x, z))
                forced.add((y, z))
    return forced


def apply_hard_mask(
    adjacency: torch.Tensor,
    var_to_states: Dict[str, List[int]],
    forced_edges: Iterable[Tuple[str, str]],
) -> Tuple[torch.Tensor, int]:
    """
    For each forced edge (x->z), set reverse block (z->x) to 0 at the STATE level.

    Returns:
        (masked_adjacency, blocks_zeroed)
    """
    out = adjacency.clone()
    blocks_zeroed = 0
    for x, z in forced_edges:
        if x not in var_to_states or z not in var_to_states:
            continue
        idx_x = var_to_states[x]
        idx_z = var_to_states[z]

        # IMPORTANT: avoid chained advanced indexing (it writes into a temporary).
        iz = torch.as_tensor(idx_z, dtype=torch.long, device=out.device)
        ix = torch.as_tensor(idx_x, dtype=torch.long, device=out.device)
        out[iz[:, None], ix[None, :]] = 0.0
        blocks_zeroed += 1
    return out, blocks_zeroed


def vstructure_hard_mask_postprocess(
    *,
    adjacency: torch.Tensor,
    var_structure: Dict,
    fci_csv_path: str,
) -> Tuple[torch.Tensor, VStructureMaskStats]:
    """
    High-level helper: infer forced collider edges from a PAG CSV and hard-mask adjacency.
    """
    df = load_pag_edges_csv(fci_csv_path)
    forced = infer_vstructure_forced_edges(df)

    var_to_states = var_structure["var_to_states"]
    masked, blocks_zeroed = apply_hard_mask(adjacency, var_to_states, forced)

    forced_in_structure = sum(1 for (x, z) in forced if x in var_to_states and z in var_to_states)
    stats = VStructureMaskStats(
        fci_csv_path=str(fci_csv_path),
        forced_edges_total=len(forced),
        forced_edges_in_structure=int(forced_in_structure),
        blocks_zeroed=int(blocks_zeroed),
    )
    return masked, stats


def enforce_vstructure_on_skeleton_mask(
    *,
    skeleton_mask: torch.Tensor,
    var_structure: Dict,
    pag_csv_path: str,
) -> Tuple[torch.Tensor, VStructureMaskStats]:
    """
    Training-time hard constraint: enforce inferred v-structures directly on the *skeleton mask*.

    For each forced collider edge (x->z), we forbid the reverse direction (z->x) by setting the
    entire state-level block to 0 in skeleton_mask. This prevents the model from ever learning z->x.
    """
    df = load_pag_edges_csv(pag_csv_path)
    forced = infer_vstructure_forced_edges(df)
    vt = var_structure["var_to_states"]

    out = skeleton_mask.clone()
    forbidden = 0
    for x, z in forced:
        if x not in vt or z not in vt:
            continue
        idx_x = vt[x]
        idx_z = vt[z]
        iz = torch.as_tensor(idx_z, dtype=torch.long, device=out.device)
        ix = torch.as_tensor(idx_x, dtype=torch.long, device=out.device)
        out[iz[:, None], ix[None, :]] = 0.0
        forbidden += 1

    forced_in_structure = sum(1 for (x, z) in forced if x in vt and z in vt)
    stats = VStructureMaskStats(
        fci_csv_path=str(pag_csv_path),
        forced_edges_total=len(forced),
        forced_edges_in_structure=int(forced_in_structure),
        blocks_zeroed=0,
        skeleton_blocks_forbidden=int(forbidden),
    )
    return out, stats


def postprocess_vstructure_on_run_dir(
    *,
    run_dir: str,
    data_path: str,
    metadata_path: str,
    ground_truth_path: str,
    ground_truth_type: str,
    pag_csv_path: str,
    edge_threshold: float,
    force: bool = False,
) -> Optional[Dict]:
    """
    Scheme A (post-training): run v-structure hard mask postprocess on a saved run directory.

    This is the "glue" wrapper so callers (e.g., experiment scripts) only need one call.

    It will:
      - load adjacency from run_dir (complete_adjacency.pt or adjacency.pt)
      - infer v-structures from pag_csv_path (must contain edge_type)
      - apply the hard mask and save:
          complete_adjacency_vstructure_masked.pt
          complete_metrics_vstructure.json
          complete_vstructure_stats.json
      - evaluate masked adjacency at variable level using edge_threshold

    If cached JSONs exist and force=False, it returns cached results.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "complete_metrics_vstructure.json"
    stats_path = out / "complete_vstructure_stats.json"
    masked_adj_path = out / "complete_adjacency_vstructure_masked.pt"

    if not force and metrics_path.exists() and stats_path.exists() and masked_adj_path.exists():
        try:
            metrics = metrics_path.read_text(encoding="utf-8")
            stats = stats_path.read_text(encoding="utf-8")
            return {
                "cached": True,
                "metrics": __import__("json").loads(metrics),
                "stats": __import__("json").loads(stats),
                "adjacency_path_out": str(masked_adj_path),
            }
        except Exception:
            # fall through to recompute
            pass

    if CausalDataLoader is None or CausalGraphEvaluator is None:
        raise RuntimeError(
            "postprocess_vstructure_on_run_dir requires modules.data_loader and modules.evaluator "
            "to be importable. Check PYTHONPATH/sys.path."
        )

    pag_csv = Path(pag_csv_path)
    if not pag_csv.exists():
        print(f"[WARN] V-structure postprocess skipped: PAG CSV not found: {pag_csv_path}")
        return None

    # Load adjacency (state-level)
    adjacency_path = out / "complete_adjacency.pt"
    if not adjacency_path.exists():
        alt = out / "adjacency.pt"
        adjacency_path = alt if alt.exists() else adjacency_path
    if not adjacency_path.exists():
        print(f"[WARN] V-structure postprocess skipped: adjacency file not found under {out}")
        return None

    adjacency = torch.load(adjacency_path, map_location="cpu")
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.tensor(adjacency)

    # Build var_structure (metadata-only; don't call load_data)
    loader = CausalDataLoader(data_path=str(data_path), metadata_path=str(metadata_path))
    var_structure = loader.get_variable_structure()

    # Apply mask
    masked, stats = vstructure_hard_mask_postprocess(
        adjacency=adjacency,
        var_structure=var_structure,
        fci_csv_path=str(pag_csv),
    )

    # Save masked adjacency + stats
    torch.save(masked, masked_adj_path)

    import json

    stats_dict = {
        "fci_csv_path": stats.fci_csv_path,
        "forced_edges_total": stats.forced_edges_total,
        "forced_edges_in_structure": stats.forced_edges_in_structure,
        "blocks_zeroed": stats.blocks_zeroed,
        "adjacency_in": str(adjacency_path),
        "adjacency_out": str(masked_adj_path),
    }
    stats_path.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")

    # Evaluate
    evaluator = CausalGraphEvaluator(
        ground_truth_path=str(ground_truth_path),
        var_structure=var_structure,
        ground_truth_type=str(ground_truth_type),
    )
    learned_edges = evaluator.extract_learned_edges(masked, threshold=float(edge_threshold))
    metrics = evaluator.evaluate(learned_edges)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "cached": False,
        "metrics": metrics,
        "stats": stats_dict,
        "adjacency_path_out": str(masked_adj_path),
    }

