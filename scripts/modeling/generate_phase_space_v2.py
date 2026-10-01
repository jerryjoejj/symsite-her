#!/usr/bin/env python3
"""Generate the 96-dimensional feature matrix for 28,560 compositions.

Output: data/space/features_28560_v2.h5.
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from herald.features.calculator import FeatureCalculator, load_element_properties
from herald.phase_space.generator import generate_all_structures

OUTPUT_DIR = PROJECT_ROOT / 'data/space'
OUTPUT_FILE = OUTPUT_DIR / "features_28560_v2.h5"


def main():
    print("=" * 70)
    print("Generate the 28,560 x 96 feature matrix")
    print("=" * 70)

    # 1. Generate the composition space.
    print("\n[1/4] Generating 28,560 compositions with M3/M4 exchange deduplication...")
    start = time.time()
    structures = generate_all_structures(enforce_m34_symmetry=True)
    print(f"  Generated {len(structures)} compositions in {time.time()-start:.1f}s")

    # 2. Load element properties.
    print("\n[2/4] Loading element properties...")
    props = load_element_properties(PROJECT_ROOT / "data/elements/element_properties.json")
    calc = FeatureCalculator(props)
    feat_names = calc.get_feature_names()
    feat_dim = calc.get_feature_dim()
    print(f"  Feature dimension: {feat_dim}")
    print(f"  First five features: {feat_names[:5]}")

    # 3. Compute features in batches.
    print(f"\n[3/4] Computing features (28,560 x {feat_dim})...")
    print("  Estimated time: about 10 minutes, depending on CPU")
    start = time.time()

    # Process batches to limit memory use.
    batch_size = 5000
    all_features = []

    for i in tqdm(range(0, len(structures), batch_size), desc="  Batch"):
        batch = structures.iloc[i:i+batch_size]
        batch_feat = calc.compute(batch)
        all_features.append(batch_feat)

    features_df = pd.concat(all_features, ignore_index=True)
    elapsed = time.time() - start
    print(f"  Computed {features_df.shape} in {elapsed:.1f}s ({len(structures)/elapsed:.0f} compositions/second)")

    # 4. Save to HDF5.
    print(f"\n[4/4] Saving to {OUTPUT_FILE}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(OUTPUT_FILE, "w") as f:
        # Composition records
        f.create_dataset("structures/M1", data=structures["M1"].astype("S10"))
        f.create_dataset("structures/M2", data=structures["M2"].astype("S10"))
        f.create_dataset("structures/M3", data=structures["M3"].astype("S10"))
        f.create_dataset("structures/M4", data=structures["M4"].astype("S10"))
        f.create_dataset("structures/index", data=structures["index"].values)

        # Feature matrix
        feat_matrix = features_df.values.astype(np.float32)
        f.create_dataset("features/matrix", data=feat_matrix, compression="gzip")
        f.create_dataset("features/names", data=[n.encode() for n in feat_names])

        # Metadata
        f.attrs["n_structures"] = len(structures)
        f.attrs["n_features"] = feat_dim
        f.attrs["symmetry"] = "M3_M4_symmetric"
        f.attrs["version"] = "Phase_E_v2"
        f.attrs["generation_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    file_size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"  Saved: {file_size_mb:.1f} MB")

    # 5. Check for feature collisions.
    print("\n[Validation] Checking feature collisions...")
    from collections import Counter
    feat_hashes = [hash(tuple(row)) for row in feat_matrix]
    hash_counts = Counter(feat_hashes)
    collisions = sum(1 for c in hash_counts.values() if c > 1)

    if collisions == 0:
        print(f"  No collisions: 28,560 distinct feature vectors")
    else:
        print(f"  Found {collisions} collisions!")
        # Identify compositions with identical feature vectors.
        for h, count in hash_counts.items():
            if count > 1:
                idx = [i for i, fh in enumerate(feat_hashes) if fh == h]
                print(f"    Collision group ({count} records): indices {idx[:5]}")
                for i in idx[:2]:
                    print(f"      {structures.iloc[i][['M1','M2','M3','M4']].to_dict()}")

    print("\n" + "=" * 70)
    print("Composition-space generation complete")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Compositions: {len(structures)}")
    print(f"  Feature dimension: {feat_dim}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
