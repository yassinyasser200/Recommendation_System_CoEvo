"""
save_seeds.py
─────────────────────────────────────────────────────────────────────────────
Seed Persistence Module for the Coevolutionary Recommendation Engine
─────────────────────────────────────────────────────────────────────────────

WHY THIS EXISTS
───────────────
Reproducibility is the cornerstone of empirical EA research. Without a
deterministic seed sequence, two runs with "the same settings" can diverge
wildly due to floating-point non-determinism in NumPy's PRNG chain. By
anchoring every benchmark run to a pre-committed seed list, we satisfy two
requirements simultaneously:

  1. The course mandates that seeds "be stored and provided" alongside results.
  2. Statistical validity: 30 independent runs drawn from non-overlapping seeds
     give unbiased estimates of mean/std RMSE across the fitness landscape.

DESIGN CHOICE — JSON over CSV/TXT
──────────────────────────────────
JSON is chosen because it:
  • Carries metadata (timestamp, algo version, seed count) in one atomic file.
  • Is trivially machine-readable for automated benchmark pipelines.
  • Survives round-trip serialisation without type coercion (integers stay ints).
  • Is human-readable — a reviewer can eyeball the file in any text editor.

USAGE
─────
  from save_seeds import SEEDS, export_seeds, load_seeds

  export_seeds()                    # writes seeds.json next to this file
  seeds = load_seeds("seeds.json")  # reads back and validates
"""

import json
import os
from datetime import datetime, timezone

# ─── Canonical seed sequence ──────────────────────────────────────────────────
# Range [1, 30] is conventional in EA literature (avoids seed=0 edge cases in
# some PRNG implementations) and gives exactly 30 independent runs — the
# minimum recommended by Derrac et al. (2011) for non-parametric statistical
# comparisons (Wilcoxon signed-rank test requires n ≥ 20).
SEEDS: list[int] = list(range(1, 31))

# ─── Metadata snapshot ────────────────────────────────────────────────────────
# VERSION is bumped manually whenever the EA engine's stochastic logic changes
# (e.g., a new operator is added). Old seed files with a different version are
# then flagged as potentially incompatible by load_seeds().
_VERSION = "1.0.0"


def export_seeds(path: str = "seeds.json") -> None:
    """
    Serialise SEEDS + audit metadata to a JSON file.

    The payload schema is intentionally flat so it can be imported directly
    into pandas (pd.read_json) or R (jsonlite::fromJSON) without pre-processing.

    Parameters
    ----------
    path : str
        Destination file path. Overwrites silently — callers should version
        the filename (e.g. seeds_v2.json) if immutability is required.
    """
    payload = {
        # ISO-8601 UTC timestamp: lets a reviewer verify the file predates
        # any submitted results, ruling out post-hoc seed cherry-picking.
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_version": _VERSION,
        "seed_count": len(SEEDS),
        # The seeds array is the single source of truth consumed by run_benchmarks().
        "seeds": SEEDS,
    }
    with open(path, "w", encoding="utf-8") as fh:
        # indent=2 keeps the file diff-friendly in version control (git blame
        # shows meaningful per-seed changes rather than one giant line delta).
        json.dump(payload, fh, indent=2)
    print(f"[seeds] Exported {len(SEEDS)} seeds → {os.path.abspath(path)}")


def load_seeds(path: str = "seeds.json") -> list[int]:
    """
    Deserialise and validate a previously exported seed file.

    Validation guards against the most common failure modes:
      • Truncated files (seed_count mismatch) — catches partial writes.
      • Version drift — warns when engine logic has changed since export.
      • Type errors — ensures all elements are plain Python ints, not floats
        (JSON doesn't distinguish, but numpy.default_rng requires int-compatible).

    Returns
    -------
    list[int]  The validated seed sequence, ready to pass to SEEDS[:n_runs].

    Raises
    ------
    ValueError  If the file is structurally invalid or corrupted.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    seeds = [int(s) for s in data["seeds"]]          # coerce JSON numbers → int

    # Count integrity check — detects files truncated mid-write by a crash.
    if len(seeds) != data["seed_count"]:
        raise ValueError(
            f"Seed count mismatch: header says {data['seed_count']}, "
            f"array has {len(seeds)}. File may be corrupted."
        )

    # Version compatibility warning — not a hard error because minor engine
    # tweaks (e.g., UI changes) don't invalidate old seed files.
    if data.get("engine_version") != _VERSION:
        print(
            f"[seeds] WARNING: file version {data['engine_version']!r} ≠ "
            f"current {_VERSION!r}. Results may not be reproducible."
        )

    print(f"[seeds] Loaded {len(seeds)} seeds (generated {data['generated_at']})")
    return seeds


# ─── CLI convenience ──────────────────────────────────────────────────────────
# Running `python save_seeds.py` regenerates seeds.json in-place.
# Useful as a pre-benchmark hook in shell scripts:
#   python save_seeds.py && python app.py
if __name__ == "__main__":
    export_seeds()
