"""Generate the composition space for multimetallic single-atom catalysts.

Four distinct metals occupy the M1/M2/M3/M4 coordination sites on
N-doped graphene. M1 is the adsorption center, M2 its neighbor, and
M3/M4 the remote pair.

Deduplicating remote exchange gives C(17,4)*4!/2 = 2380*24/2 = 28,560
compositions. Retain only M3 <= M4 in element-symbol lexicographic order.
For example, keep (Ag,Co,Cu,Fe) and omit (Ag,Co,Fe,Cu).
"""

from itertools import combinations, permutations
import pandas as pd

# Seventeen transition metals, ordered by atomic number
METAL_POOL = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ru", "Rh", "Pd", "Ag", "Ir", "Pt", "Au"
]


def generate_all_structures(metal_pool=None, enforce_m34_symmetry=True, allow_duplicate_metals=False):
    """Generate distinct compositions for the four-site catalyst platform.

    Args:
        metal_pool: Element pool, defaulting to 17 transition metals.
        enforce_m34_symmetry: Deduplicate M3/M4 exchange, default True.
        allow_duplicate_metals: Allow repeated elements, default False.

    Returns:
        DataFrame with index, M1, M2, M3, M4 and structure_str columns.

    Space sizes for 17 metals:
        Repeats allowed, no exchange deduplication: 17**4 = 83,521.
        Distinct metals, no exchange deduplication: P(17,4) = 57,120.
        Distinct metals, exchange deduplicated: C(17,4)*4!/2 = 28,560.
    """
    pool = metal_pool or METAL_POOL
    structures = []

    if not enforce_m34_symmetry and allow_duplicate_metals:
        # Repeated metals allowed: 17**4 = 83,521.
        for idx, (m1, m2, m3, m4) in enumerate(
            ((a, b, c, d) for a in pool for b in pool for c in pool for d in pool)
        ):
            structures.append({
                "index": idx,
                "M1": m1,
                "M2": m2,
                "M3": m3,
                "M4": m4,
                "structure_str": f"{m1}-{m2}-{m3}-{m4}"
            })
    elif not enforce_m34_symmetry and not allow_duplicate_metals:
        # Distinct metals without exchange deduplication: P(17,4) = 57,120.
        idx = 0
        for four_metals in combinations(pool, 4):
            for perm in permutations(four_metals):
                m1, m2, m3, m4 = perm
                structures.append({
                    "index": idx,
                    "M1": m1,
                    "M2": m2,
                    "M3": m3,
                    "M4": m4,
                    "structure_str": f"{m1}-{m2}-{m3}-{m4}"
                })
                idx += 1
    else:
        # Deduplicate M3/M4 exchange to obtain 28,560 compositions.
        idx = 0
        for four_metals in combinations(pool, 4):
            # Enumerate permutations of the four metals.
            for perm in permutations(four_metals):
                m1, m2, m3, m4 = perm
                # Keep only M3 <= M4 in lexicographic order.
                if m3 <= m4:
                    structures.append({
                        "index": idx,
                        "M1": m1,
                        "M2": m2,
                        "M3": m3,
                        "M4": m4,
                        "structure_str": f"{m1}-{m2}-{m3}-{m4}"
                    })
                    idx += 1

    return pd.DataFrame(structures)


def verify_m34_symmetry(df: pd.DataFrame) -> dict:
    """Check that remote-exchange counterparts were deduplicated.

    For each (M1,M2,M3,M4), (M1,M2,M4,M3) should be absent.

    Returns:
        total: Number of compositions.
        has_symmetric_pair: Count with exchange counterparts, expected 0.
        is_valid: Whether the check passed.
    """
    # Build a composition set for fast lookup.
    structure_set = set(
        (row.M1, row.M2, row.M3, row.M4) for row in df.itertuples()
    )

    # Check whether each M3/M4-exchanged counterpart exists.
    symmetric_pairs = 0
    for row in df.itertuples():
        swapped = (row.M1, row.M2, row.M4, row.M3)
        if swapped in structure_set:
            symmetric_pairs += 1

    return {
        'total': len(df),
        'has_symmetric_pair': symmetric_pairs,
        'is_valid': symmetric_pairs == 0,
    }


if __name__ == "__main__":
    # Generate and validate the 28,560-composition space.
    print("=" * 70)
    print("Composition-space generator check")
    print("=" * 70)

    # Generate compositions.
    print("\nGenerating compositions with M3/M4 exchange deduplication...")
    df = generate_all_structures(enforce_m34_symmetry=True)
    print(f"  Total: {len(df)}")
    print(f"  Expected: 28,560 = C(17,4) x 4! / 2")
    print(f"  First five rows:\n{df.head()}")

    # Validate exchange deduplication.
    print("\nChecking M3/M4 exchange deduplication...")
    result = verify_m34_symmetry(df)
    print(f"  Total compositions: {result['total']}")
    print(f"  Exchange counterparts present: {result['has_symmetric_pair']}")
    print(f"  Check passed: {'✓' if result['is_valid'] else '✗'}")

    # Compare with the undeduplicated space.
    print("\nComparing with the undeduplicated space...")
    df_old = generate_all_structures(enforce_m34_symmetry=False)
    print(f"  Undeduplicated count: {len(df_old)}")
    print(f"  Deduplicated count: {len(df)}")
    print(f"  Reduction: {(1 - len(df)/len(df_old))*100:.1f}%")

    print("\n" + "=" * 70)
    if result['is_valid'] and len(df) == 28560:
        print("Check passed")
    else:
        print("Check failed")
    print("=" * 70)
