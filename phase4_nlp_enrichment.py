"""
Phase 4 — NLP Enrichment (Stretch)
Embeds technique descriptions with sentence-transformers and correlates
semantic similarity with co-occurrence similarity.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    print("Phase 4 — NLP Enrichment")
    print("=" * 50)

    if not HAS_ST:
        print("  ERROR: sentence-transformers not installed. Skipping Phase 4.")
        print("  Install with: pip install sentence-transformers")
        return

    techniques_df = pd.read_csv(OUTPUT_DIR / "techniques.csv")
    print(f"  Loaded {len(techniques_df)} techniques")

    descriptions = techniques_df["description"].fillna("").tolist()
    tech_ids = techniques_df["technique_id"].tolist()

    print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("  Encoding technique descriptions...")
    embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)

    print("  Computing semantic similarity matrix...")
    sem_sim = cosine_similarity(embeddings)
    np.fill_diagonal(sem_sim, 0)

    sem_sim_df = pd.DataFrame(sem_sim, index=tech_ids, columns=tech_ids)
    sem_sim_df.to_csv(OUTPUT_DIR / "semantic_similarity.csv")

    jaccard_path = OUTPUT_DIR / "cooccurrence_jaccard.csv"
    if jaccard_path.exists():
        print("  Correlating semantic similarity with co-occurrence (Jaccard)...")
        jaccard_df = pd.read_csv(jaccard_path, index_col=0)

        common_techs = sorted(set(sem_sim_df.index) & set(jaccard_df.index))
        sem_vals = sem_sim_df.loc[common_techs, common_techs].values
        jac_vals = jaccard_df.loc[common_techs, common_techs].values

        upper_idx = np.triu_indices(len(common_techs), k=1)
        sem_flat = sem_vals[upper_idx]
        jac_flat = jac_vals[upper_idx]

        corr, pvalue = spearmanr(sem_flat, jac_flat)
        print(f"  Spearman correlation: {corr:.4f} (p={pvalue:.2e})")

        stats = {
            "spearman_correlation": round(float(corr), 4),
            "p_value": float(pvalue),
            "n_techniques_compared": len(common_techs),
            "n_pairs": len(sem_flat),
        }
        with open(OUTPUT_DIR / "nlp_correlation_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
    else:
        print("  No Jaccard matrix found — skipping correlation.")

    matrix_path = OUTPUT_DIR / "group_technique_matrix.csv"
    if matrix_path.exists():
        print("\n  Predicting technique expansions for sparse groups...")
        matrix = pd.read_csv(matrix_path, index_col=0)
        groups_df = pd.read_csv(OUTPUT_DIR / "groups.csv")
        group_names = dict(zip(groups_df["group_id"], groups_df["name"]))

        sparse_groups = matrix.index[matrix.sum(axis=1).between(3, 8)]
        expansions = []

        for gid in sparse_groups[:20]:
            known = matrix.columns[matrix.loc[gid] == 1].tolist()
            unknown = matrix.columns[matrix.loc[gid] == 0].tolist()

            if not known or not unknown:
                continue

            known_in_sim = [t for t in known if t in sem_sim_df.index]
            unknown_in_sim = [t for t in unknown if t in sem_sim_df.columns]

            if not known_in_sim or not unknown_in_sim:
                continue

            avg_sim_to_known = sem_sim_df.loc[unknown_in_sim, known_in_sim].mean(axis=1)
            top5 = avg_sim_to_known.sort_values(ascending=False).head(5)

            tech_names = dict(zip(techniques_df["technique_id"], techniques_df["name"]))
            for tid, score in top5.items():
                expansions.append({
                    "group_id": gid,
                    "group_name": group_names.get(gid, gid),
                    "predicted_technique": tid,
                    "technique_name": tech_names.get(tid, tid),
                    "semantic_score": round(float(score), 4),
                })

        if expansions:
            exp_df = pd.DataFrame(expansions)
            exp_df.to_csv(OUTPUT_DIR / "predicted_expansions.csv", index=False)
            print(f"  Saved {len(expansions)} predicted expansions for {len(sparse_groups[:20])} sparse groups")

            with open(OUTPUT_DIR / "predicted_expansions.json", "w") as f:
                json.dump(expansions, f)
        else:
            print("  No expansions generated.")

    print(f"\nPhase 4 complete.")


if __name__ == "__main__":
    main()
