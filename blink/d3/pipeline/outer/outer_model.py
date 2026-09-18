"""Complete, reusable outer R(c,k) model with exact local orbit compression.

The key is never fixed to a slice.  Nonlinear template indices are an
evaluation compression, NOT a replacement for the 52-dimensional linear
quotient.  Physical-key masks are exported separately for exact joins.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from engine.connected import Transfer, signs
from engine.core import Coordinates, RC, RCI, Space, embed, extract, fraction_record, fwht, independent, mix, perm


class OuterModel:
    """Exact R evaluator and complete finite template representation.

    Load requires local_orbits.py to have generated its two output files.
    Cached canonical spectra are bounded, so no 2^52 object is allocated.
    """

    def __init__(self, data: Path | None = None) -> None:
        data = data or HERE / "output"
        self.transfer = Transfer()
        self.orbits = json.loads((data / "column_translation_orbits.json").read_text(encoding="utf-8"))
        self.column_data = self.orbits["columns"]
        self.templates = []
        self.template_locations = []
        with np.load(data / "column_translation_orbits.npz") as npz:
            for col in range(4):
                masks = npz[f"masks_{col}"]
                locations = np.searchsorted(masks, self.transfer.bcols[col])
                assert np.array_equal(masks[locations], self.transfer.bcols[col])
                self.templates.append(npz[f"canonical_{col}"].copy())
                self.template_locations.append(locations)

    def tokens(self, key: int) -> tuple[tuple[int, ...], int, int]:
        """Return class tuple, scalar, and connector translation for a full key."""
        classes, scalar, translation = [], 1, 0
        for col, column in enumerate(self.column_data):
            value = extract(key, col)
            code = sum(((value >> bit) & 1) << i for i, bit in enumerate(column["key_positions"]))
            record = column["key_records"][code]
            if record["class"] == -1:
                return (), 0, 0
            classes.append(record["class"])
            scalar *= record["scale"]
            translation ^= embed(record["translation"], col)
        return tuple(classes), scalar, mix(perm(translation))

    @lru_cache(maxsize=8)
    def canonical_spectrum(self, classes: tuple[int, ...]) -> tuple[np.ndarray, int]:
        if len(classes) != 4:
            raise ValueError("four nonzero local classes are required")
        spectrum = self.transfer.weights.copy()
        bound = int(np.abs(spectrum).max())
        for col, class_id in enumerate(classes):
            values = self.templates[col][class_id, self.template_locations[col]]
            bound *= int(np.abs(values).max())
            if bound * len(spectrum) >= 1 << 62:
                raise OverflowError("canonical sum requires wider than signed 64-bit arithmetic")
            spectrum *= values
        return spectrum, self.transfer.denominator * (1 << 64)

    def evaluate(self, connector_key: int, normalized_outer_key: int) -> Fraction:
        classes, scalar, translation = self.tokens(normalized_outer_key)
        if not scalar:
            return Fraction(0)
        spectrum, denominator = self.canonical_spectrum(classes)
        numerator = int(np.dot(spectrum, signs(self.transfer.beta, connector_key ^ translation)))
        return Fraction(numerator * scalar, denominator)

    def physical_head(self, rk1: int, rk2: int) -> Fraction:
        return self.evaluate(rk2 ^ RC[1], mix(rk1 ^ RC[0]))

    def physical_tail(self, rk4: int, rk5: int) -> Fraction:
        return self.evaluate(rk4 ^ RCI[3], mix(rk5 ^ RCI[4]))

    def template_histogram(self, classes: tuple[int, ...]) -> dict:
        """All 2^21 connector syndromes for ONE template, with exact multiplicities.

        This is a reusable conditional factor, not the all-key R histogram.
        The caller must keep original key tokens and all shared forms when
        joining the middle block.  The mean/scale is for canonical waves.
        """
        spectrum, denominator = self.canonical_spectrum(classes)
        dense = np.zeros(1 << len(self.transfer.beta_basis), dtype=np.int64)
        dense[self.transfer.indices] = spectrum
        truth = fwht(dense)
        assert np.all(truth >= 0)
        unique, counts = np.unique(truth, return_counts=True)
        histogram = {int(value): int(count) for value, count in zip(unique, counts)}
        return {
            "classes": list(classes),
            "connector_rank": 21,
            "denominator": denominator,
            "histogram": [{"numerator": value, "count": count} for value, count in sorted(histogram.items())],
            "maximum_syndromes": np.flatnonzero(truth == truth.max()).tolist(),
            "minimum_nonzero_syndromes": np.flatnonzero(truth == truth[truth > 0].min()).tolist() if np.any(truth > 0) else [],
            "scope": "Canonical template over all connector syndromes; multiply numerator by key scalar. Original connector syndrome is translated by tokens(key)[2].",
        }


def support_report() -> dict:
    source = ROOT / "data/four_zero_complete_endpoint_support.json"
    old = json.loads(source.read_text(encoding="utf-8"))
    basis = [int(row["mask"], 0) for row in old["basis"]]
    low_mask = (1 << 64) - 1
    connector_basis = independent(mask & low_mask for mask in basis)
    inner_basis = independent(mask >> 64 for mask in basis)
    assert Space(basis).rank == 52
    assert len(connector_basis) == 21 and len(inner_basis) == 31
    assert Space([mask for mask in connector_basis] + [mask << 64 for mask in inner_basis]).rank == 52
    rows = []
    for row in old["basis"]:
        c = int(row["connector_key_mask"], 0)
        q = int(row["normalized_outer_key_mask"], 0)
        physical = mix(q)
        rows.append({
            "connector_mask": f"0x{c:016x}",
            "normalized_inner_mask": f"0x{q:016x}",
            "physical_inner_mask": f"0x{physical:016x}",
            "head_mask_rk1_rk2": f"0x{((c << 64) | physical):032x}",
            "tail_mask_rk4_rk5": f"0x{((physical << 64) | c):032x}",
            "head_constant_phase": ((c & RC[1]).bit_count() ^ (physical & RC[0]).bit_count()) & 1,
            "tail_constant_phase": ((c & RCI[3]).bit_count() ^ (physical & RCI[4]).bit_count()) & 1,
            "coefficient_before_constant_phase": row["coefficient"],
        })
    column_bases = [independent(extract(mask, col) for mask in inner_basis) for col in range(4)]
    return {
        "scope": "Complete support of a single exact four-round endpoint R. No all-key extrema or histogram claimed.",
        "source": str(source),
        "exact_rank": 52,
        "nonzero_fourier_terms": old["nonzero_fourier_terms"],
        "connector_rank": 21,
        "normalized_inner_rank": 31,
        "connector_basis": [f"0x{mask:016x}" for mask in connector_basis],
        "normalized_inner_basis": [f"0x{mask:016x}" for mask in inner_basis],
        "physical_inner_basis": [f"0x{mix(mask):016x}" for mask in inner_basis],
        "normalized_inner_column_ranks": [len(b) for b in column_bases],
        "normalized_inner_column_bases": [[f"0x{mask:04x}" for mask in b] for b in column_bases],
        "complete_support_basis": rows,
        "rank_proof": "The stored 52 independent nonzero Fourier terms span both projection spaces; their dimensions sum to 52, so the endpoint quotient is the direct sum of the 21 connector forms and 31 normalized-inner forms.",
        "mean_over_all_endpoint_keys": old["mean"],
        "physical_mappings": {"head": "R(rk2 xor RC2, M(rk1 xor RC1))", "tail": "R(rk4 xor RCI4, M(rk5 xor RCI5))"},
        "shared_key_warning": "Head forms live on rk1/rk2 and tail forms on rk4/rk5, but the middle-six block reuses all four. Join their explicit physical forms; an unconditional endpoint histogram is insufficient for the scheme-A joint distribution.",
    }


def verify(model: OuterModel, cases: int = 64) -> dict:
    started = time.monotonic()
    rng = random.Random(202609150464)
    checked, nonzero = 0, 0
    for case in range(cases):
        connector_key = rng.getrandbits(64)
        normalized_key = rng.getrandbits(64)
        if case % 2 == 0:
            # Exercise live branches as well as the naturally frequent zeros.
            for col, column in enumerate(model.column_data):
                record = rng.choice([row for row in column["key_records"] if row["class"] >= 0])
                for i, bit in enumerate(column["key_positions"]):
                    physical_bit = 4 * (col + 4 * (bit // 4)) + bit % 4
                    normalized_key &= ~(1 << physical_bit)
                    normalized_key |= ((record["key_code"] >> i) & 1) << physical_bit
        direct = model.transfer.evaluate(connector_key, normalized_key)
        compressed = model.evaluate(connector_key, normalized_key)
        assert direct == compressed, (case, direct, compressed)
        checked += 1
        nonzero += bool(direct)
    sat = [int(value, 16) for value in ("2874cc1e532a2747", "1062f52829c95f66", "18c078facb7bfa85", "73ae068f503c26b9", "90479091156bcc2d")]
    head, tail = model.physical_head(sat[0], sat[1]), model.physical_tail(sat[3], sat[4])
    assert head == Fraction(12451, 1 << 44)
    assert tail == Fraction(6987, 1 << 43)
    return {
        "random_and_conditioned_complete_endpoint_comparisons": checked,
        "nonzero_comparisons": nonzero,
        "all_equal": True,
        "sat_head": fraction_record(head),
        "sat_tail": fraction_record(tail),
        "seed": 202609150464,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--checks", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = support_report()
    (args.output / "outer_support.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model = OuterModel(args.output)
    checks = verify(model, args.checks)
    (args.output / "outer_verification.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
