"""
Phase 1 — Data Pipeline
Fetches MITRE ATT&CK Enterprise STIX data, parses techniques, groups,
and their relationships into clean CSVs for downstream analysis.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

STIX_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
OUTPUT_DIR = Path(__file__).parent / "output"


def fetch_stix_bundle() -> dict:
    print("Fetching ATT&CK Enterprise STIX bundle...")
    resp = requests.get(STIX_URL, timeout=120)
    resp.raise_for_status()
    bundle = resp.json()
    print(f"  Loaded {len(bundle['objects'])} STIX objects")
    return bundle


def is_valid(obj: dict) -> bool:
    if obj.get("revoked", False):
        return False
    if obj.get("x_mitre_deprecated", False):
        return False
    return True


def extract_techniques(objects: list[dict]) -> pd.DataFrame:
    rows = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if not is_valid(obj):
            continue

        tech_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tech_id = ref.get("external_id", "")
                break

        if not tech_id:
            continue

        tactics = []
        for phase in obj.get("kill_chain_phases", []):
            if phase.get("kill_chain_name") == "mitre-attack":
                tactics.append(phase["phase_name"])

        platforms = obj.get("x_mitre_platforms", [])

        rows.append({
            "technique_id": tech_id,
            "name": obj.get("name", ""),
            "tactics": "|".join(tactics),
            "platforms": "|".join(platforms),
            "description": (obj.get("description") or "")[:500],
            "stix_id": obj["id"],
        })

    df = pd.DataFrame(rows)
    return df.sort_values("technique_id").reset_index(drop=True)


def extract_groups(objects: list[dict]) -> pd.DataFrame:
    rows = []
    for obj in objects:
        if obj.get("type") != "intrusion-set":
            continue
        if not is_valid(obj):
            continue

        group_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                group_id = ref.get("external_id", "")
                break

        rows.append({
            "group_id": group_id,
            "name": obj.get("name", ""),
            "description": (obj.get("description") or "")[:500],
            "stix_id": obj["id"],
        })

    df = pd.DataFrame(rows)
    return df.sort_values("group_id").reset_index(drop=True)


def build_group_technique_matrix(
    objects: list[dict],
    techniques_df: pd.DataFrame,
    groups_df: pd.DataFrame,
) -> pd.DataFrame:
    tech_stix_to_id = dict(zip(techniques_df["stix_id"], techniques_df["technique_id"]))
    group_stix_to_id = dict(zip(groups_df["stix_id"], groups_df["group_id"]))

    usage_pairs = []
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "uses":
            continue
        if not is_valid(obj):
            continue

        src = obj.get("source_ref", "")
        tgt = obj.get("target_ref", "")

        if src in group_stix_to_id and tgt in tech_stix_to_id:
            usage_pairs.append((group_stix_to_id[src], tech_stix_to_id[tgt]))

    all_techniques = sorted(techniques_df["technique_id"].unique())
    all_groups = sorted(groups_df["group_id"].unique())

    matrix = pd.DataFrame(0, index=all_groups, columns=all_techniques)
    for group_id, tech_id in usage_pairs:
        if group_id in matrix.index and tech_id in matrix.columns:
            matrix.loc[group_id, tech_id] = 1

    return matrix


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = fetch_stix_bundle()
    objects = bundle["objects"]

    techniques_df = extract_techniques(objects)
    groups_df = extract_groups(objects)
    matrix = build_group_technique_matrix(objects, techniques_df, groups_df)

    techniques_df.drop(columns=["stix_id"]).to_csv(
        OUTPUT_DIR / "techniques.csv", index=False
    )
    groups_df.drop(columns=["stix_id"]).to_csv(
        OUTPUT_DIR / "groups.csv", index=False
    )
    matrix.to_csv(OUTPUT_DIR / "group_technique_matrix.csv")

    total_cells = matrix.shape[0] * matrix.shape[1]
    density = matrix.values.sum() / total_cells if total_cells > 0 else 0

    print(f"\n{'='*50}")
    print(f"Phase 1 — Data Pipeline Complete")
    print(f"{'='*50}")
    print(f"  Techniques: {len(techniques_df)}")
    print(f"  Groups:     {len(groups_df)}")
    print(f"  Matrix:     {matrix.shape[0]} groups x {matrix.shape[1]} techniques")
    print(f"  Usage pairs:{int(matrix.values.sum())}")
    print(f"  Density:    {density:.4f} ({density*100:.2f}%)")
    print(f"\nOutput saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
