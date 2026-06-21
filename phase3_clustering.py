"""
Phase 3 — Clustering
Clusters threat groups by TTP similarity and techniques by co-usage,
using K-Means and hierarchical clustering with UMAP visualization.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed, falling back to PCA")
    from sklearn.decomposition import PCA

OUTPUT_DIR = Path(__file__).parent / "output"


def load_data():
    matrix = pd.read_csv(OUTPUT_DIR / "group_technique_matrix.csv", index_col=0)
    groups_df = pd.read_csv(OUTPUT_DIR / "groups.csv")
    techniques_df = pd.read_csv(OUTPUT_DIR / "techniques.csv")
    return matrix, groups_df, techniques_df


def find_best_k(data: np.ndarray, k_range=range(3, 9)) -> int:
    best_k, best_score = 3, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(data)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(data, labels, metric="cosine")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def reduce_2d(data: np.ndarray, metric="cosine") -> np.ndarray:
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, metric=metric, random_state=42, n_neighbors=15)
        return reducer.fit_transform(data)
    else:
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(data)


def cluster_groups(matrix: pd.DataFrame, groups_df: pd.DataFrame):
    print("\n  --- Group Clustering ---")
    data = matrix.values.astype(np.float64)

    row_sums = data.sum(axis=1)
    valid_mask = row_sums >= 3
    data_filtered = data[valid_mask]
    group_ids = matrix.index[valid_mask].tolist()
    print(f"  Groups with >=3 techniques: {len(group_ids)}")

    best_k = find_best_k(data_filtered)
    print(f"  Best k (silhouette): {best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(data_filtered)

    coords = reduce_2d(data_filtered)

    group_names = dict(zip(groups_df["group_id"], groups_df["name"]))
    results = pd.DataFrame({
        "group_id": group_ids,
        "group_name": [group_names.get(gid, gid) for gid in group_ids],
        "cluster": labels,
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
    })
    results.to_csv(OUTPUT_DIR / "group_clusters.csv", index=False)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.7, s=40)
    for i, gid in enumerate(group_ids):
        name = group_names.get(gid, gid)
        if len(name) > 20:
            name = name[:18] + "…"
        plt.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=5, alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Threat Group Clusters by TTP Similarity (k={best_k})")
    plt.xlabel("UMAP-1" if HAS_UMAP else "PC-1")
    plt.ylabel("UMAP-2" if HAS_UMAP else "PC-2")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "group_clusters_umap.png", dpi=150)
    plt.close()

    json_data = {
        "k": best_k,
        "points": [
            {
                "id": row["group_id"],
                "name": row["group_name"],
                "cluster": int(row["cluster"]),
                "x": round(float(row["umap_x"]), 4),
                "y": round(float(row["umap_y"]), 4),
            }
            for _, row in results.iterrows()
        ]
    }
    with open(OUTPUT_DIR / "group_clusters.json", "w") as f:
        json.dump(json_data, f)

    print(f"  Saved group_clusters.csv, group_clusters_umap.png, group_clusters.json")
    return results


def cluster_techniques(matrix: pd.DataFrame, techniques_df: pd.DataFrame):
    print("\n  --- Technique Clustering ---")
    data = matrix.values.T.astype(np.float64)

    col_sums = data.sum(axis=1)
    valid_mask = col_sums >= 3
    data_filtered = data[valid_mask]
    tech_ids = matrix.columns[valid_mask].tolist()
    print(f"  Techniques used by >=3 groups: {len(tech_ids)}")

    best_k = find_best_k(data_filtered, k_range=range(4, 12))
    print(f"  Best k (silhouette): {best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(data_filtered)

    coords = reduce_2d(data_filtered)

    tech_names = dict(zip(techniques_df["technique_id"], techniques_df["name"]))
    results = pd.DataFrame({
        "technique_id": tech_ids,
        "technique_name": [tech_names.get(tid, tid) for tid in tech_ids],
        "cluster": labels,
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
    })
    results.to_csv(OUTPUT_DIR / "technique_clusters.csv", index=False)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=20)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Technique Clusters by Group Co-usage (k={best_k})")
    plt.xlabel("UMAP-1" if HAS_UMAP else "PC-1")
    plt.ylabel("UMAP-2" if HAS_UMAP else "PC-2")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "technique_clusters_umap.png", dpi=150)
    plt.close()

    json_data = {
        "k": best_k,
        "points": [
            {
                "id": row["technique_id"],
                "name": row["technique_name"],
                "cluster": int(row["cluster"]),
                "x": round(float(row["umap_x"]), 4),
                "y": round(float(row["umap_y"]), 4),
            }
            for _, row in results.iterrows()
        ]
    }
    with open(OUTPUT_DIR / "technique_clusters.json", "w") as f:
        json.dump(json_data, f)

    print(f"  Saved technique_clusters.csv, technique_clusters_umap.png, technique_clusters.json")
    return results


def main():
    print("Phase 3 — Clustering")
    print("=" * 50)

    matrix, groups_df, techniques_df = load_data()
    print(f"  Loaded matrix: {matrix.shape[0]} groups x {matrix.shape[1]} techniques")

    cluster_groups(matrix, groups_df)
    cluster_techniques(matrix, techniques_df)

    print(f"\nPhase 3 complete.")


if __name__ == "__main__":
    main()
