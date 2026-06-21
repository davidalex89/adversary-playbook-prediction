"""
Phase 2 — Co-occurrence Model
Builds technique co-occurrence matrices (Jaccard and Lift) from the
group-technique matrix, and exports top-N predictions per technique.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"


def load_matrix() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "group_technique_matrix.csv", index_col=0)


def build_cooccurrence(matrix: pd.DataFrame) -> pd.DataFrame:
    """Raw co-occurrence counts: how many groups use both technique i and j."""
    binary = matrix.values.astype(np.float64)
    cooccur = binary.T @ binary
    return pd.DataFrame(cooccur, index=matrix.columns, columns=matrix.columns)


def jaccard_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    binary = matrix.values.astype(np.float64)
    intersection = binary.T @ binary
    row_sums = binary.sum(axis=0)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    union[union == 0] = 1
    jaccard = intersection / union
    np.fill_diagonal(jaccard, 0)
    return pd.DataFrame(jaccard, index=matrix.columns, columns=matrix.columns)


def lift_score(matrix: pd.DataFrame) -> pd.DataFrame:
    binary = matrix.values.astype(np.float64)
    n_groups = binary.shape[0]
    intersection = binary.T @ binary
    support = binary.sum(axis=0) / n_groups
    expected = support[:, None] * support[None, :] * n_groups
    expected[expected == 0] = 1
    lift = intersection / expected
    np.fill_diagonal(lift, 0)
    return pd.DataFrame(lift, index=matrix.columns, columns=matrix.columns)


def predict(technique_id: str, similarity_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if technique_id not in similarity_df.index:
        return pd.DataFrame(columns=["technique_id", "score"])
    scores = similarity_df.loc[technique_id].sort_values(ascending=False).head(top_n)
    return pd.DataFrame({"technique_id": scores.index, "score": scores.values})


def export_top_n_json(similarity_df: pd.DataFrame, techniques_df: pd.DataFrame, top_n: int = 10) -> dict:
    tech_names = dict(zip(techniques_df["technique_id"], techniques_df["name"]))
    result = {}
    for tech_id in similarity_df.index:
        top = similarity_df.loc[tech_id].sort_values(ascending=False).head(top_n)
        result[tech_id] = [
            {"id": tid, "name": tech_names.get(tid, tid), "score": round(float(s), 4)}
            for tid, s in zip(top.index, top.values)
            if s > 0
        ]
    return result


def main():
    print("Phase 2 — Co-occurrence Model")
    print("=" * 50)

    matrix = load_matrix()
    techniques_df = pd.read_csv(OUTPUT_DIR / "techniques.csv")

    print(f"  Matrix loaded: {matrix.shape[0]} groups x {matrix.shape[1]} techniques")

    print("  Computing Jaccard similarity...")
    jaccard_df = jaccard_similarity(matrix)
    jaccard_df.to_csv(OUTPUT_DIR / "cooccurrence_jaccard.csv")

    print("  Computing Lift scores...")
    lift_df = lift_score(matrix)
    lift_df.to_csv(OUTPUT_DIR / "cooccurrence_lift.csv")

    print("  Exporting top-10 JSON (Jaccard)...")
    top10_jaccard = export_top_n_json(jaccard_df, techniques_df, top_n=10)
    with open(OUTPUT_DIR / "cooccurrence_top10_jaccard.json", "w") as f:
        json.dump(top10_jaccard, f)

    print("  Exporting top-10 JSON (Lift)...")
    top10_lift = export_top_n_json(lift_df, techniques_df, top_n=10)
    with open(OUTPUT_DIR / "cooccurrence_top10_lift.json", "w") as f:
        json.dump(top10_lift, f)

    example_tech = "T1566"
    print(f"\n  Example: Top 5 co-occurring techniques for {example_tech} (Jaccard):")
    preds = predict(example_tech, jaccard_df, top_n=5)
    tech_names = dict(zip(techniques_df["technique_id"], techniques_df["name"]))
    for _, row in preds.iterrows():
        name = tech_names.get(row["technique_id"], "")
        print(f"    {row['technique_id']:10s} {name:40s} {row['score']:.4f}")

    print(f"\n  Example: Top 5 co-occurring techniques for {example_tech} (Lift):")
    preds = predict(example_tech, lift_df, top_n=5)
    for _, row in preds.iterrows():
        name = tech_names.get(row["technique_id"], "")
        print(f"    {row['technique_id']:10s} {name:40s} {row['score']:.4f}")

    print(f"\nPhase 2 complete. Output saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
