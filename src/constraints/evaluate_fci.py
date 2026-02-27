"""
Evaluate FCI/RFCI outputs against ground truth.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import config


def parse_ground_truth(gt_path):
    ground_truth_edges = set()
    gt_path = Path(gt_path)

    with open(gt_path, "r", encoding="utf-8") as f:
        content = f.read()

    prob_pattern = r"probability\s*\(\s*(\w+)\s*\|\s*([^)]+)\s*\)"
    matches = list(re.finditer(prob_pattern, content))

    if matches:
        for match in matches:
            child = match.group(1)
            parents_str = match.group(2)
            parents = [p.strip() for p in parents_str.split(",")]
            for parent in parents:
                if parent:
                    ground_truth_edges.add((parent, child))
    else:
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "->" in line:
                parts = line.split("->")
                if len(parts) == 2:
                    source = parts[0].strip()
                    target = parts[1].strip()
                    if source and target:
                        ground_truth_edges.add((source, target))

    return ground_truth_edges


def parse_fci_csv(fci_csv_path):
    df = pd.read_csv(fci_csv_path)

    if "Source" in df.columns and "Target" in df.columns:
        source_col, target_col = "Source", "Target"
    elif "source" in df.columns and "target" in df.columns:
        source_col, target_col = "source", "target"
    else:
        source_col, target_col = df.columns[0], df.columns[1]

    if "edge_type" in df.columns:
        edge_type_col = "edge_type"
    elif "Edge_Type" in df.columns:
        edge_type_col = "Edge_Type"
    elif "type" in df.columns:
        edge_type_col = "type"
    else:
        edge_type_col = None

    fci_directed = set()
    fci_undirected = set()
    edge_counts = {
        "directed": 0,
        "undirected": 0,
        "partial": 0,
        "tail-tail": 0,
        "bidirected": 0,
    }

    for _, row in df.iterrows():
        source = row[source_col]
        target = row[target_col]
        edge_type = row[edge_type_col] if edge_type_col else "directed"
        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1

        if edge_type == "directed":
            fci_directed.add((source, target))
        elif edge_type in ["undirected", "partial", "tail-tail", "bidirected"]:
            fci_undirected.add(tuple(sorted([source, target])))

    return fci_directed, fci_undirected, edge_counts


def compute_fci_unresolved_ratio(fci_csv_path):
    _, _, edge_counts = parse_fci_csv(fci_csv_path)
    total_edges = sum(edge_counts.values())
    directed_edges = edge_counts.get("directed", 0)
    unresolved_edges = total_edges - directed_edges
    unresolved_ratio = unresolved_edges / total_edges if total_edges > 0 else 0
    return {
        "unresolved_count": unresolved_edges,
        "resolved_count": directed_edges,
        "total_edges": total_edges,
        "unresolved_ratio": unresolved_ratio,
        "edge_type_breakdown": edge_counts,
    }


def compute_shd(fci_csv_path, ground_truth_path):
    gt_edges = parse_ground_truth(ground_truth_path)
    fci_directed, fci_undirected, _ = parse_fci_csv(fci_csv_path)

    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in gt_edges}
    fci_all_undirected = {tuple(sorted([e[0], e[1]])) for e in fci_directed}
    fci_all_undirected.update(fci_undirected)

    additions = len(fci_all_undirected - gt_undirected)
    deletions = len(gt_undirected - fci_all_undirected)

    reversals = 0
    unresolved_as_errors = 0
    for fci_edge in fci_directed:
        undirected_edge = tuple(sorted([fci_edge[0], fci_edge[1]]))
        if undirected_edge in gt_undirected:
            reversed_edge = (fci_edge[1], fci_edge[0])
            if reversed_edge in gt_edges and fci_edge not in gt_edges:
                reversals += 1
    for fci_undir_edge in fci_undirected:
        if fci_undir_edge in gt_undirected:
            unresolved_as_errors += 1

    skeleton_shd = additions + deletions
    full_shd = additions + deletions + reversals + unresolved_as_errors
    return {
        "skeleton_shd": skeleton_shd,
        "full_shd": full_shd,
        "shd": full_shd,
        "additions": additions,
        "deletions": deletions,
        "reversals": reversals,
        "unresolved_as_errors": unresolved_as_errors,
        "total_direction_errors": reversals + unresolved_as_errors,
    }


def evaluate_fci(fci_csv_path, ground_truth_path, output_dir=None):
    gt_edges = parse_ground_truth(ground_truth_path)
    fci_directed, fci_undirected, edge_counts = parse_fci_csv(fci_csv_path)
    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in gt_edges}
    fci_all_undirected = {tuple(sorted([e[0], e[1]])) for e in fci_directed}
    fci_all_undirected.update(fci_undirected)

    undirected_tp = len(fci_all_undirected & gt_undirected)
    undirected_fp = len(fci_all_undirected - gt_undirected)
    undirected_fn = len(gt_undirected - fci_all_undirected)

    edge_precision = undirected_tp / (undirected_tp + undirected_fp) if (undirected_tp + undirected_fp) > 0 else 0
    edge_recall = undirected_tp / (undirected_tp + undirected_fn) if (undirected_tp + undirected_fn) > 0 else 0
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0

    correctly_oriented = 0
    incorrectly_oriented = 0
    for fci_edge in fci_directed:
        undirected_edge = tuple(sorted([fci_edge[0], fci_edge[1]]))
        if undirected_edge in gt_undirected:
            if fci_edge in gt_edges:
                correctly_oriented += 1
            else:
                reversed_edge = (fci_edge[1], fci_edge[0])
                if reversed_edge in gt_edges:
                    incorrectly_oriented += 1

    orientation_accuracy = correctly_oriented / (correctly_oriented + incorrectly_oriented) if (correctly_oriented + incorrectly_oriented) > 0 else 0
    unresolved_stats = compute_fci_unresolved_ratio(fci_csv_path)
    shd_stats = compute_shd(fci_csv_path, ground_truth_path)
    undirected_ratio = len(fci_undirected) / len(fci_all_undirected) if len(fci_all_undirected) > 0 else 0

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"evaluation_FCI_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("FCI EVALUATION REPORT\n")
            f.write(f"Edge F1: {edge_f1:.4f}\n")
            f.write(f"Edge Precision: {edge_precision:.4f}\n")
            f.write(f"Edge Recall: {edge_recall:.4f}\n")
            f.write(f"Orientation Accuracy: {orientation_accuracy:.4f}\n")
            f.write(f"Unresolved Ratio: {unresolved_stats['unresolved_ratio']:.4f}\n")
            f.write(f"Full SHD: {shd_stats['full_shd']}\n")

    return {
        "skeleton_shd": shd_stats["skeleton_shd"],
        "full_shd": shd_stats["full_shd"],
        "shd": shd_stats["shd"],
        "shd_additions": shd_stats["additions"],
        "shd_deletions": shd_stats["deletions"],
        "shd_reversals": shd_stats["reversals"],
        "edge_f1": edge_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "orientation_accuracy": orientation_accuracy,
        "unresolved_ratio": unresolved_stats["unresolved_ratio"],
        "unresolved_count": unresolved_stats["unresolved_count"],
        "resolved_count": unresolved_stats["resolved_count"],
        "undirected_ratio": undirected_ratio,
        "undirected_tp": undirected_tp,
        "undirected_fp": undirected_fp,
        "undirected_fn": undirected_fn,
        "correctly_oriented": correctly_oriented,
        "incorrectly_oriented": incorrectly_oriented,
        "edge_type_breakdown": edge_counts,
    }


def find_latest_fci_csv(output_dir):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    rfci_csvs = list(output_path.glob("edges_RFCI_*.csv"))
    if rfci_csvs:
        return max(rfci_csvs, key=lambda p: p.stat().st_mtime)
    fci_csvs = list(output_path.glob("edges_FCI_*.csv"))
    if not fci_csvs:
        return None
    return max(fci_csvs, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    out_dir = config.get_constraint_output_dir(config.DATASET)
    gt_path = config.get_current_dataset_config()["ground_truth_path"]
    latest = find_latest_fci_csv(out_dir)
    if latest and Path(gt_path).exists():
        metrics = evaluate_fci(latest, gt_path, output_dir=out_dir)
        print(metrics)
