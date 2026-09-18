"""Exact translation-orbit compression of the four outer-block column spectra.

Only files below this directory are generated.  No key slice is imposed:
all low-three-bit keys at output-active cells are exhausted.  Compression
is checked coefficient by coefficient on the complete interface mask set.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
from engine.connected import Transfer
from engine.core import cells, extract, independent, local_indicator, parity, solve, fwht


def _signs(masks: np.ndarray, value: int) -> np.ndarray:
    p = masks.astype(np.uint64) & np.uint64(value)
    for shift in (32, 16, 8, 4, 2, 1):
        p ^= p >> np.uint64(shift)
    return 1 - 2 * (p & np.uint64(1)).astype(np.int64)


def analyze(output: Path) -> dict:
    started = time.monotonic()
    from engine.core import U, T
    tr = Transfer()
    output.mkdir(parents=True, exist_ok=True)
    all_columns = []
    arrays = {}
    for col in range(4):
        di, do = extract(T, col), extract(U, col)
        positions = [4 * row + bit for row, d in enumerate(cells(do, 4)) if d for bit in range(3)]
        masks = np.unique(tr.bcols[col]).astype(np.uint16)
        mask_index = {int(mask): idx for idx, mask in enumerate(masks)}
        classes = {}
        canonical_vectors = []
        records = []
        for code in range(1 << len(positions)):
            key = sum(((code >> i) & 1) << bit for i, bit in enumerate(positions))
            wave = fwht(local_indicator(di, do, key))[masks]
            nonzero = np.flatnonzero(wave)
            if len(nonzero) == 0:
                records.append({"key_code": code, "key": key, "class": -1, "scale": 0, "translation": 0})
                continue
            scale = math.gcd(*(int(value) for value in wave[nonzero]))
            basis = independent(int(masks[i]) for i in nonzero)
            translation = solve([(mask, int(wave[mask_index[mask]] < 0)) for mask in basis])
            canonical = (wave // scale) * _signs(masks, translation)
            signature = canonical.tobytes()
            if signature not in classes:
                classes[signature] = len(canonical_vectors)
                canonical_vectors.append(canonical)
            class_id = classes[signature]
            assert np.array_equal(wave, canonical_vectors[class_id] * scale * _signs(masks, translation))
            records.append({"key_code": code, "key": key, "class": class_id, "scale": scale, "translation": translation})
        class_sizes = Counter(record["class"] for record in records)
        summary = {
            "column": col,
            "input_difference": f"0x{di:04x}",
            "output_difference": f"0x{do:04x}",
            "key_positions": positions,
            "keys_exhausted": len(records),
            "interface_input_masks": len(masks),
            "nonzero_translation_classes_up_to_scale": len(classes),
            "class_key_counts": dict(sorted(class_sizes.items())),
            "scale_counts": dict(sorted(Counter(record["scale"] for record in records).items())),
            "classes": [
                {"class": class_id,
                 "keys": class_sizes[class_id],
                 "scales": dict(sorted(Counter(record["scale"] for record in records if record["class"] == class_id).items())),
                 "representative_key": next(record["key"] for record in records if record["class"] == class_id),
                 "representative_translation": next(record["translation"] for record in records if record["class"] == class_id),
                 "canonical_sha256": hashlib.sha256(canonical.tobytes()).hexdigest()}
                for class_id, canonical in enumerate(canonical_vectors)
            ],
            "key_records": records,
        }
        arrays[f"masks_{col}"] = masks
        arrays[f"canonical_{col}"] = np.asarray(canonical_vectors, dtype=np.int64)
        all_columns.append(summary)
        print(json.dumps({key: value for key, value in summary.items() if key not in ("key_records", "classes")}), flush=True)
    np.savez_compressed(output / "column_translation_orbits.npz", **arrays)
    result = {
        "scope": "Complete R(c,k) interface, all relevant physical local keys; no fixed-key slice.",
        "identity": "wave_c(k,a) = scale_c(k) * canonical_c[class_c(k),a] * (-1)^(a dot translation_c(k)) for every interface mask a",
        "columns": all_columns,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "column_translation_orbits.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    result = analyze(args.output)
    print(f"Complete in {result['elapsed_seconds']:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
