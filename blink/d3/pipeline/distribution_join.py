"""Exact conditional joins for the two scheme-A distributions.

The expensive outer histograms must be COMPLETE before this module will
publish global extrema. Counts are Python integers: 114/206-dimensional
coset counts must never be accumulated in uint64.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTER_DEN = 1 << 54
MIDDLE_DEN = 1 << 38
KEY_DEN = OUTER_DEN * MIDDLE_DEN * OUTER_DEN
JOINT_DEN = MIDDLE_DEN * OUTER_DEN
SHARED6 = [0x100010001, 0x200020002, 0x400040004,
           0x1000100010, 0x2000200020, 0x4000400040]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def frac_record(n, d):
    q = Fraction(int(n), int(d))
    return {"numerator": q.numerator, "denominator": q.denominator,
            "text": str(q)}


def linear_coordinates(target, basis):
    pivots = {}
    for i, row in enumerate(basis):
        v, c = int(row), 1 << i
        while v:
            p = v.bit_length() - 1
            if p not in pivots:
                pivots[p] = (v, c)
                break
            v ^= pivots[p][0]
            c ^= pivots[p][1]
        if not v:
            raise ValueError("Coordinate basis is not independent")
    v, c = int(target), 0
    while v:
        p = v.bit_length() - 1
        if p not in pivots:
            raise ValueError("Mask is outside the declared shared space")
        v ^= pivots[p][0]
        c ^= pivots[p][1]
    return c


def syndrome(forms, value):
    return sum(((int(f) & int(value)).bit_count() & 1) << i
               for i, f in enumerate(forms))


def row_stats(hist):
    live = [(n, c) for n, c in hist.items() if n and c]
    return {"total": sum(hist.values()), "zero": hist.get(0, 0),
            "positive": sum(c for _, c in live),
            "minimum": min(live) if live else None,
            "maximum": max(live) if live else None}


class ExactJoin:
    def __init__(self, outer_npz, outer_summary):
        self.outer_npz = str(Path(outer_npz).resolve())
        self.outer_summary = str(Path(outer_summary).resolve())
        metadata = read_json(outer_summary)
        if metadata.get("complete") is not True:
            raise ValueError("Refusing global statistics from an incomplete outer run")
        declared_denominators = [metadata[name] for name in
                                 ("denominator", "probability_denominator")
                                 if name in metadata]
        if not declared_denominators or any(
                not isinstance(value, (int, str)) or int(value) != OUTER_DEN
                for value in declared_denominators):
            raise ValueError("Complete outer metadata must declare denominator 2^54")
        with np.load(outer_npz, allow_pickle=False) as data:
            numerators = data["numerators"]
            counts = data["counts_by_shared6"]
            if numerators.dtype.kind not in "ui" or counts.dtype.kind not in "ui":
                raise ValueError("Outer numerators and counts must be integer arrays")
            if numerators.ndim != 1 or counts.ndim != 2:
                raise ValueError("Expected a one-dimensional value axis and a count matrix")
            if np.any(numerators < 0) or np.any(numerators > OUTER_DEN):
                raise ValueError("Outer probability numerator is outside [0,2^54]")
            if np.any(counts < 0) or np.any(counts > (1 << 46)):
                raise ValueError("Outer class count is outside [0,2^46]")
            self.numerators = numerators.astype(np.uint64)
            self.counts = counts.astype(np.uint64)
        if self.counts.shape != (64, len(self.numerators)):
            raise ValueError("Expected 64 conditional shared-six histograms")
        if len(set(map(int, self.numerators))) != len(self.numerators):
            raise ValueError("Duplicate numerator bins")
        self.index = {int(n): i for i, n in enumerate(self.numerators)}
        self.outer_rows = [Counter({int(n): int(c) for n, c in
                                   zip(self.numerators, row) if c})
                           for row in self.counts]
        for row in self.outer_rows:
            # Python integers also reject malformed rows whose uint64 sums
            # would wrap around to the expected total.
            if sum(row.values()) != 1 << 46:
                raise ValueError("An outer row does not contain exactly 2^46 cosets")
            if sum(n * c for n, c in row.items()) != 3 * (1 << 59):
                raise ValueError("An outer conditional first moment is inconsistent")
        self.outer_stats = [row_stats(row) for row in self.outer_rows]

        phases = read_json(HERE / "outer/output/outer_conditional_factors.json")
        self.head_phase = sum(b << i for i, b in enumerate(
            phases["head_conditions"]["head_constant_bits"]))
        self.tail_phase = sum(b << i for i, b in enumerate(
            phases["tail_conditions"]["tail_constant_bits"]))
        middle_meta = read_json(HERE / "middle/output/middle_conditional_on_outer_shared8.json")
        assert middle_meta["denominator"] == MIDDLE_DEN
        combined_shared = SHARED6 + [m << 256 for m in SHARED6]
        self.middle_forms = [linear_coordinates(int(m, 0), combined_shared)
                             for m in middle_meta["shared_basis_hex"]]
        self.middle_table = np.load(HERE / "middle/output/middle_shared8_remaining10_numerators.npy")
        assert self.middle_table.shape == (256, 1024)
        self.middle_rows = [Counter(map(int, row)) for row in self.middle_table]
        self.middle_stats = [row_stats(row) for row in self.middle_rows]
        self.middle_indices = np.array([[syndrome(self.middle_forms, h | (t << 6))
                                         for t in range(64)] for h in range(64)])
        self.tail_middle = read_json(HERE / "middle/output/middle_conditional_on_tail.json")
        self.tail_forms = [linear_coordinates(int(m, 0) >> 256, SHARED6)
                           for m in self.tail_middle["shared_basis_hex"]]
        self._middle_weight_cache = {}

    def key_extrema(self):
        positive_count, weighted_sum = 0, 0
        best, least = -1, None
        best_count, least_count = 0, 0
        best_selectors, least_selectors = [], []
        for h in range(64):
            hs = self.outer_stats[h ^ self.head_phase]
            for t in range(64):
                ts = self.outer_stats[t ^ self.tail_phase]
                shared8 = int(self.middle_indices[h, t])
                ms = self.middle_stats[shared8]
                positive_count += hs["positive"] * ms["positive"] * ts["positive"]
                if not hs["positive"] or not ms["positive"] or not ts["positive"]:
                    continue
                for which in ("maximum", "minimum"):
                    hn, hc = hs[which]
                    mn, mc = ms[which]
                    tn, tc = ts[which]
                    n, count = hn * mn * tn, hc * mc * tc
                    selector = {"physical_head_shared6": h, "physical_tail_shared6": t,
                                "middle_shared8": shared8,
                                "head_raw_numerator": hn, "middle_raw_numerator": mn,
                                "tail_raw_numerator": tn, "coset_count": count}
                    if which == "maximum":
                        if n > best:
                            best, best_count, best_selectors = n, 0, []
                        if n == best:
                            best_count += count
                            best_selectors.append(selector)
                    else:
                        if least is None or n < least:
                            least, least_count, least_selectors = n, 0, []
                        if n == least:
                            least_count += count
                            least_selectors.append(selector)
        assert best > 0 and least is not None
        return {
            "quotient_rank": 114, "rank_status": "proved minimal Fourier support rank",
            "total_cosets": 1 << 114, "keys_per_coset": 1 << 334,
            "maximum": {"probability": frac_record(best, KEY_DEN),
                        "coset_count": best_count, "conditional_selectors": best_selectors},
            "minimum_nonzero": {"probability": frac_record(least, KEY_DEN),
                                "coset_count": least_count, "conditional_selectors": least_selectors},
            "zero_cosets": (1 << 114) - positive_count, "positive_cosets": positive_count,
            "mean": frac_record(27, 1 << 114),
            "distribution_representation": "complete shared-condition factorization; not a flattened list",
            "all_value_count_formula": "N(n/2^146)=sum_{h,t,a*b*c=n} H[h](a) M[shared8(h,t)](b) T[t](c)",
        }

    def tail_rows_shared2(self):
        rows = [Counter() for _ in range(4)]
        for physical6 in range(64):
            gamma = syndrome(self.tail_forms, physical6)
            rows[gamma].update(self.outer_rows[physical6 ^ self.tail_phase])
        assert all(sum(row.values()) == 1 << 50 for row in rows)
        return rows

    def middle_rows_shared2(self):
        rows = []
        for row in self.tail_middle["rows"]:
            hist = Counter()
            for item in row["histogram"]:
                p = item["probability"]
                scaled = Fraction(p["numerator"], p["denominator"]) * MIDDLE_DEN
                assert scaled.denominator == 1
                hist[int(scaled)] += int(item["remaining_coset_count"])
            assert sum(hist.values()) == 1 << 16
            rows.append(hist)
        return rows

    def joint_histogram(self):
        tail, middle = self.tail_rows_shared2(), self.middle_rows_shared2()
        mt = Counter()
        for mrow, trow in zip(middle, tail):
            mt[0] += mrow.get(0, 0) * sum(trow.values())
            mt[0] += trow.get(0, 0) * (sum(mrow.values()) - mrow.get(0, 0))
            for mn, mc in mrow.items():
                if mn:
                    for tn, tc in trow.items():
                        if tn:
                            mt[mn * tn] += mc * tc
        assert sum(mt.values()) == 1 << 68
        multiplier = 3 * (1 << 97)
        joint = Counter({n: c * multiplier for n, c in mt.items() if n})
        joint[0] = (1 << 206) - sum(joint.values())
        assert sum(joint.values()) == 1 << 206
        # Independent DC certificate is a strict end-to-end normalization check.
        assert Fraction(sum(n * c for n, c in joint.items()), JOINT_DEN << 206) == Fraction(27, 1 << 114)
        stats = row_stats(joint)
        return joint, {
            "quotient_rank": 206, "rank_status": "proved minimal Fourier support rank",
            "rank_certificate": str(HERE.parent / "data/joint_rank_certificate.json"),
            "total_cosets": 1 << 206, "plaintext_key_pairs_per_coset": 1 << 306,
            "distinct_probability_values": len(joint),
            "maximum": {"probability": frac_record(stats["maximum"][0], JOINT_DEN),
                        "coset_count": stats["maximum"][1]},
            "minimum_nonzero": {"probability": frac_record(stats["minimum"][0], JOINT_DEN),
                                "coset_count": stats["minimum"][1]},
            "zero_cosets": stats["zero"], "positive_cosets": stats["positive"],
            "mean": frac_record(27, 1 << 114),
            "counting_identity": "N_joint(v)=3*2^97*N_MT(v) for v>0; zero by complement",
        }

    def key_frequency(self, probability):
        """Count a specified probability exactly, without materializing all products."""
        scaled = Fraction(probability) * KEY_DEN
        if scaled < 0 or scaled.denominator != 1:
            return 0
        target = int(scaled)
        if not target:
            return self.key_extrema()["zero_cosets"]
        answer = 0
        # Integer matrices keep all 114-dimensional counts exact.
        head_order = np.arange(64) ^ self.head_phase
        tail_order = np.arange(64) ^ self.tail_phase
        for mn in sorted({n for row in self.middle_rows for n in row if n}):
            if target % mn:
                continue
            if mn not in self._middle_weight_cache:
                self._middle_weight_cache[mn] = np.array([
                    [self.middle_rows[int(s)][mn] for s in row]
                    for row in self.middle_indices], dtype=object)
            weights = self._middle_weight_cache[mn]
            ht = target // mn
            for hi, hn in enumerate(self.numerators):
                hn = int(hn)
                if not hn or ht % hn:
                    continue
                ti = self.index.get(ht // hn)
                if ti is not None:
                    # Only the two needed 64-entry columns become Python
                    # integers; do not duplicate a potentially huge 64*N
                    # outer histogram into two object matrices per query.
                    heads = self.counts[head_order, hi].astype(object)
                    tails = self.counts[tail_order, ti].astype(object)
                    answer += int(heads @ weights @ tails)
        return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-npz", type=Path, required=True)
    parser.add_argument("--outer-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "output/final_join")
    parser.add_argument("--query-key-probability", help="Exact rational, e.g. 34935/1180591620717411303424")
    parser.add_argument("--skip-coset-finalization", action="store_true",
                        help="Only export histograms; do not reconstruct extremal coset equations")
    args = parser.parse_args()
    join = ExactJoin(args.outer_npz, args.outer_summary)
    if args.query_key_probability is not None:
        print(json.dumps({"probability": args.query_key_probability,
                          "coset_count": join.key_frequency(args.query_key_probability)}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    key = join.key_extrema()
    joint_hist, joint = join.joint_histogram()
    summary = {"model": "scheme A restricted 4+6+4, zero tweak, two zero outer value connectors",
               "outer_npz": join.outer_npz, "outer_summary": join.outer_summary,
               "outer_distribution_complete": True, "key": key, "joint": joint,
               "extremal_coset_conditions_complete": False,
               "key_distribution_flattened": False, "joint_distribution_flattened": False,
               "joint_probability_artifact_complete": False,
               "shared6_physical_masks_on_RK1_or_RK5": [hex(m) for m in SHARED6],
               "middle_shared8_masks_on_12_shared_bits": [hex(m) for m in join.middle_forms],
               "head_normalized_xor_physical_syndrome": join.head_phase,
               "tail_normalized_xor_physical_syndrome": join.tail_phase}
    write_json_atomic(args.output / "summary.json", summary)
    # Decimal strings avoid int64 overflow and remain lossless in every JSON reader.
    import gzip
    artifact = args.output / "joint_all_probabilities.jsonl.gz"
    temporary_artifact = artifact.with_name(artifact.name + ".tmp")
    with gzip.open(temporary_artifact, "wt", encoding="utf-8") as f:
        for n, c in sorted(joint_hist.items()):
            f.write(json.dumps({"probability": frac_record(n, JOINT_DEN),
                                "coset_count_decimal": str(c)}) + "\n")
    temporary_artifact.replace(artifact)
    summary["joint_probability_artifact_complete"] = True
    summary["joint_distribution_flattened"] = True
    write_json_atomic(args.output / "summary.json", summary)
    print(json.dumps({"key_maximum": key["maximum"]["probability"],
                      "joint_maximum": joint["maximum"]["probability"],
                      "joint_distinct_values": len(joint_hist)}, indent=2))
    if not args.skip_coset_finalization:
        del joint_hist, join
        import gc
        gc.collect()
        from finalize_cosets import finalize
        status_path = args.output / "coset_finalization_status.json"
        write_json_atomic(status_path, {"state": "running"})
        try:
            result = finalize(args.outer_npz, args.outer_summary, args.output)
        except BaseException as exc:
            write_json_atomic(status_path, {"state": "failed", "error": str(exc)})
            raise
        write_json_atomic(status_path, result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
