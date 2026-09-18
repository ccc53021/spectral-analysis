"""Exact shared-key labels and first moments of unrestricted outer factors.

The 6 shared forms involve normalized inner-key columns 0 and 1, last
row, low three bits.  Connector masks are disjoint from this interface.
No independence of reused physical round keys is assumed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from engine.core import RC, RCI, T, U, Space, embed, extract, fraction_record, mix, parity
from engine.local import counts


def analyze(output: Path) -> dict:
    orbit_file = output / "column_translation_orbits.json"
    orbits = json.loads(orbit_file.read_text(encoding="utf-8"))
    shared = [embed(1 << (12 + bit), col) for col in (0, 1) for bit in range(3)]
    physical = [mix(mask) for mask in shared]
    local = []
    all_means = []
    for col, orbit in enumerate(orbits["columns"]):
        full_counts = counts(extract(T, col), extract(U, col))
        full_mean = Fraction(int(full_counts.sum()), 1 << 32)
        conditional_means = [
            Fraction(sum(int(full_counts[key]) for key in range(65536) if ((key >> 12) & 7) == label), (1 << 13) * (1 << 16))
            for label in range(8)
        ]
        assert all(mean == full_mean for mean in conditional_means)
        all_means.append(full_mean)
        rows = orbit["key_records"]
        if col == 3:
            # The complete R support uses only bits 0,2,4,6 here. The
            # other two low key bits have no effect at this interface.
            selected = [row for row in rows if not row["key"] & 0x22]
            for row in selected:
                for delta in (0, 2, 0x20, 0x22):
                    other_key = row["key"] ^ delta
                    other = next(other for other in rows if other["key"] == other_key)
                    assert (other["class"], other["scale"], other["translation"]) == (row["class"], row["scale"], row["translation"])
            rows = selected
        groups = range(8) if col < 2 else range(1)
        label_tables = []
        for label in groups:
            selected = [row for row in rows if col >= 2 or ((row["key"] >> 12) & 7) == label]
            classes = Counter(row["class"] for row in selected)
            label_tables.append({
                "shared_label": label if col < 2 else None,
                "key_count": len(selected),
                "zero_column_keys": classes.get(-1, 0),
                "nonzero_class_key_counts": {str(class_id): multiplicity for class_id, multiplicity in sorted(classes.items()) if class_id >= 0},
            })
        local.append({
            "column": col,
            "effective_key_rank_for_R": 9 if col < 3 else 4,
            "mean_J_over_all_keys_and_inputs": fraction_record(full_mean),
            "mean_J_for_each_fixed_last_row_low3": [fraction_record(mean) for mean in conditional_means],
            "template_key_counts_by_shared_label": label_tables,
        })
    mean_R = Fraction(1, 1 << 12)
    for mean in all_means:
        mean_R *= mean
    assert mean_R == Fraction(3, 1 << 41)
    assert Space(shared).rank == 6
    rows = []
    for label in range(64):
        value = Fraction(1, 1 << 12)
        for col in range(4):
            value *= all_means[col]
        rows.append({"shared_syndrome": label, "mean_R": fraction_record(value)})
    result = {
        "scope": "All physical endpoint keys; exact conditional first moments and template multiplicities, NOT full R probability histograms.",
        "shared_rank": 6,
        "syndrome_order": "bits 0..2: normalized key column 0 row 3 low3; bits 3..5: column 1 row 3 low3",
        "normalized_shared_masks": [f"0x{mask:016x}" for mask in shared],
        "physical_inner_shared_masks": [f"0x{mask:016x}" for mask in physical],
        "head_conditions": {
            "word": "RK1",
            "equation": "physical_inner_shared_masks[i] dot RK1 = syndrome_i xor head_constant_bits[i]",
            "head_constant_bits": [parity(mask & RC[0]) for mask in physical],
        },
        "tail_conditions": {
            "word": "RK5",
            "equation": "physical_inner_shared_masks[i] dot RK5 = syndrome_i xor tail_constant_bits[i]",
            "tail_constant_bits": [parity(mask & RCI[4]) for mask in physical],
        },
        "local_factors": local,
        "conditional_means": rows,
        "constant_conditional_mean": fraction_record(mean_R),
        "necessary_nonzero_J_key_fraction_at_each_shared_syndrome": fraction_record(Fraction(27, 2048)),
        "necessary_nonzero_scope": "This is a necessary live-template condition only; do not infer R has no additional zeros without checking all connector syndromes.",
        "head_open_input_identity": "E[H4(X,RK1,RK2) | all middle-left key forms] = 3/2^41, because averaging X gives R, the only shared R forms are the six listed, and every conditional R first moment is identical.",
        "conditional_template_formula": "R histograms given each 6-bit label are obtained by weighting the full 21-bit connector histogram for each of 66^3*4 canonical templates with these column class multiplicities. Input translations only permute the free connector syndromes.",
        "nonlinear_compression_warning": "Template labels and XOR translations compress computation but do not reduce the 52-dimensional linear coset space.",
        "unrestricted_histogram_complete": False,
    }
    (output / "outer_conditional_factors.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    result = analyze(args.output)
    print(json.dumps({"shared_rank": result["shared_rank"], "constant_conditional_mean": result["constant_conditional_mean"], "conditional_rows": len(result["conditional_means"]), "full_R_histogram": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
