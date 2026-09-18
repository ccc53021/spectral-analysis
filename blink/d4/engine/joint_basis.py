"""Exact minimum-weight basis for the zero-connector ten-round model.

The model fixes A1=A2=A3=A4=0 and keeps A0, q2, and q4 open.  Unlike the
older sparse candidate search, this script represents the complete nonzero
Fourier support:

* every nonzero A0 mask from the exact, fixed-middle SB1 spectra;
* every nonzero q2 and q4 mask from the exact variable-middle spectra.

The Cartesian product is traversed lazily in exact global-coefficient order.
Greedy GF(2) rank selection therefore gives a minimum-weight independent
prefix for every requested dimension up to the full support rank.
"""

from fractions import Fraction
import heapq
from itertools import product
import json
import math
from pathlib import Path

from .superbox_spectrum import fourier_coefficients
from .joint_transfer import (
    GLOBAL_TRANSFER_DENOMINATOR_BITS,
    contracted_joint_mask,
    contracted_trail_numerator,
    internal_trail_vector,
)
from .superbox_transfer import (
    activity_column,
    fixed_middle_column_spectrum,
    insert_column,
)
from .trail import C, D, SUPERBOXES


STATE_BITS = 64
INTERNAL_BITS = 7 * STATE_BITS
ACTIVE_COLUMN = 3


def column_mask(state_mask, column):
    return sum(
        ((state_mask >> (column + 4 * row)) & 1) << row
        for row in range(4)
    )


SB2_MASKS = (column_mask(C, ACTIVE_COLUMN), column_mask(D, ACTIVE_COLUMN))
SB4_MASKS = (column_mask(D, ACTIVE_COLUMN), column_mask(C, ACTIVE_COLUMN))


class RankAccumulator:
    """Incremental GF(2) row space using highest-set-bit pivots."""

    def __init__(self, rows=()):
        self.pivots = {}
        for row in rows:
            self.add(row)

    @property
    def rank(self):
        return len(self.pivots)

    def reduce(self, value):
        value = int(value)
        while value:
            pivot = value.bit_length() - 1
            row = self.pivots.get(pivot)
            if row is None:
                break
            value ^= row
        return value

    def add(self, value):
        value = self.reduce(value)
        if not value:
            return False
        pivot = value.bit_length() - 1
        self.pivots[pivot] = value
        return True

    def contains(self, value):
        return self.reduce(value) == 0


def exact_a0_support():
    """Return all exact nonzero SB1 input masks and their common numerators."""
    row = SUPERBOXES[0]
    columns = []
    column_summary = []
    for column in range(4):
        source = activity_column(row["source"], column)
        target = activity_column(row["target"], column)
        spectrum = fixed_middle_column_spectrum(source, target, 0, 0)
        entries = [
            (insert_column(local_mask, column), int(numerator))
            for local_mask, numerator in enumerate(spectrum)
            if int(numerator)
        ]
        local_rank = RankAccumulator(mask for mask, _ in entries).rank
        columns.append(entries)
        column_summary.append({
            "column": column,
            "source_activity": source,
            "target_activity": target,
            "support_size": len(entries),
            "rank": local_rank,
        })

    support = []
    for choices in product(*columns):
        mask = 0
        numerator = 1
        for local_mask, local_numerator in choices:
            mask ^= local_mask
            numerator *= local_numerator
        support.append((mask, numerator))
    support.sort(key=lambda item: (-abs(item[1]), item[0]))
    if len({mask for mask, _ in support}) != len(support):
        raise AssertionError("duplicate A0 masks in exact Cartesian support")
    return support, column_summary


def exact_q_support(activity_pair):
    """Return all exact active-column q masks for SB2 or SB4."""
    coefficients = fourier_coefficients(*activity_pair)
    return [
        (insert_column(local_mask, ACTIVE_COLUMN), int(numerator))
        for local_mask, numerator in enumerate(coefficients)
        if int(numerator)
    ]


def exact_rest_support(sb1_zero_numerator):
    """Return the 36 exact signed factors for q2/q4 and SB2--SB5."""
    q2_support = exact_q_support(SB2_MASKS)
    q4_support = exact_q_support(SB4_MASKS)
    boundaries = (0, 0, 0, 0, 0)
    rows = []
    for q2, _ in q2_support:
        for q4, _ in q4_support:
            numerator = contracted_trail_numerator(boundaries, q2, q4)
            quotient, remainder = divmod(numerator, sb1_zero_numerator)
            if remainder:
                raise AssertionError("global numerator does not factor over SB1")
            if quotient:
                rows.append((q2, q4, quotient))
    if len(rows) != len(q2_support) * len(q4_support):
        raise AssertionError("unexpected zero in q2/q4 Cartesian support")
    return rows, q2_support, q4_support


def internal_vector(a0, q2, q4):
    return a0 | (q2 << (5 * STATE_BITS)) | (q4 << (6 * STATE_BITS))


def old_internal_space(q2_support, q4_support):
    rows = []
    rows.extend(internal_vector(0, q2, 0) for q2, _ in q2_support)
    rows.extend(internal_vector(0, 0, q4) for q4, _ in q4_support)
    return RankAccumulator(rows)


def candidate_row(a0_entry, rest_entry, examined):
    a0, sb1_numerator = a0_entry
    q2, q4, rest_numerator = rest_entry
    common_numerator = sb1_numerator * rest_numerator
    boundaries = (a0, 0, 0, 0, 0)
    vector = internal_vector(a0, q2, q4)
    reference_vector = internal_trail_vector(boundaries, q2, q4)
    if vector != reference_vector:
        raise AssertionError("packed internal vector mismatch")
    reference_numerator = contracted_trail_numerator(boundaries, q2, q4)
    if common_numerator != reference_numerator:
        raise AssertionError("factorized global coefficient mismatch")
    coefficient = Fraction(common_numerator, 1 << GLOBAL_TRANSFER_DENOMINATOR_BITS)
    return {
        "boundary_inputs": boundaries,
        "q2": q2,
        "q4": q4,
        "internal_vector": vector,
        "joint_mask": contracted_joint_mask(boundaries, q2, q4),
        "common_numerator": common_numerator,
        "coefficient": coefficient,
        "cluster_abs_weight": (
            GLOBAL_TRANSFER_DENOMINATOR_BITS - math.log2(abs(common_numerator))
        ),
        "ranked_candidates_examined": examined,
    }


def exact_ranked_basis(a0_support, rest_support, target_rank):
    """Merge all Cartesian streams and greedily select an exact basis."""
    heap = []
    for rest_index, (q2, q4, rest_numerator) in enumerate(rest_support):
        a0, sb1_numerator = a0_support[0]
        vector = internal_vector(a0, q2, q4)
        heapq.heappush(
            heap,
            (
                -abs(sb1_numerator * rest_numerator),
                vector,
                rest_index,
                0,
            ),
        )

    span = RankAccumulator()
    selected = []
    examined = 0
    while heap and span.rank < target_rank:
        _, vector, rest_index, a0_index = heapq.heappop(heap)
        examined += 1
        if span.add(vector):
            row = candidate_row(
                a0_support[a0_index], rest_support[rest_index], examined
            )
            row["rank"] = span.rank
            selected.append(row)
            print(
                f"  b{span.rank - 1}: weight={row['cluster_abs_weight']:.12f} "
                f"examined={examined}",
                flush=True,
            )

        next_index = a0_index + 1
        if next_index < len(a0_support):
            a0, sb1_numerator = a0_support[next_index]
            q2, q4, rest_numerator = rest_support[rest_index]
            next_vector = internal_vector(a0, q2, q4)
            heapq.heappush(
                heap,
                (
                    -abs(sb1_numerator * rest_numerator),
                    next_vector,
                    rest_index,
                    next_index,
                ),
            )

    if span.rank != target_rank:
        raise RuntimeError(
            f"complete support rank is only {span.rank}, requested {target_rank}"
        )
    return selected, examined


def serialize_row(row, trail_id):
    coefficient = row["coefficient"]
    return {
        "type": "trail",
        "trail_id": trail_id,
        "rank": row["rank"],
        "selection_weight": "exact_global_cluster_abs_weight",
        "cluster_abs_weight": row["cluster_abs_weight"],
        "boundary_inputs": [
            f"0x{value:016x}" for value in row["boundary_inputs"]
        ],
        "q2": f"0x{row['q2']:016x}",
        "q4": f"0x{row['q4']:016x}",
        "internal_vector": f"0x{row['internal_vector']:0112x}",
        "joint_mask": f"0x{row['joint_mask']:0128x}",
        "global_common_numerator": row["common_numerator"],
        "global_common_denominator_bits": GLOBAL_TRANSFER_DENOMINATOR_BITS,
        "coefficient_numerator": coefficient.numerator,
        "coefficient_denominator": coefficient.denominator,
        "ranked_candidates_examined": row["ranked_candidates_examined"],
    }


def write_basis(path, meta, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(meta)]
    lines.extend(
        json.dumps(serialize_row(row, index)) for index, row in enumerate(rows)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    a0_support, column_summary = exact_a0_support()
    sb1_zero = next(numerator for mask, numerator in a0_support if mask == 0)
    rest_support, q2_support, q4_support = exact_rest_support(sb1_zero)
    a0_rank = RankAccumulator(mask for mask, _ in a0_support).rank
    old_space = old_internal_space(q2_support, q4_support)
    full_rank = a0_rank + old_space.rank
    if args.ranked_dimension > full_rank:
        raise ValueError(
            f"ranked dimension {args.ranked_dimension} exceeds full rank {full_rank}"
        )
    if args.dimension > args.ranked_dimension:
        raise ValueError("prefix dimension exceeds ranked dimension")

    global_support_size = len(a0_support) * len(rest_support)
    print(
        f"exact support: A0={len(a0_support)} q2={len(q2_support)} "
        f"q4={len(q4_support)} global={global_support_size} rank={full_rank}",
        flush=True,
    )
    selected, examined = exact_ranked_basis(
        a0_support, rest_support, args.ranked_dimension
    )

    ranked_span = RankAccumulator(row["internal_vector"] for row in selected)
    prefix = selected[:args.dimension]
    prefix_span = RankAccumulator(row["internal_vector"] for row in prefix)
    old_vectors = list(old_space.pivots.values())
    meta_common = {
        "type": "meta",
        "cipher": "Blink-64",
        "scope": "Figure-3 ten-round cluster; A1=A2=A3=A4=0",
        "selection": (
            "complete exact global Fourier support; ascending exact weight; "
            "GF(2) rank greedy; deterministic internal-vector tie break"
        ),
        "selection_weight": "-log2(abs(exact ten-round clustered coefficient))",
        "candidate_generation": "complete factored support; no local top-N truncation",
        "global_common_denominator_bits": GLOBAL_TRANSFER_DENOMINATOR_BITS,
        "a0_column_support": column_summary,
        "a0_support_size": len(a0_support),
        "a0_rank": a0_rank,
        "q2_support_size": len(q2_support),
        "q4_support_size": len(q4_support),
        "q2_q4_rank": old_space.rank,
        "global_support_size": global_support_size,
        "full_support_rank": full_rank,
        "ranked_candidates_examined": examined,
    }
    ranked_meta = {
        **meta_common,
        "selected_dimension": len(selected),
        "old_rank6_contained": all(
            ranked_span.contains(value) for value in old_vectors
        ),
        "basis_role": "master ordered basis; use prefix b0..b(d-1)",
    }
    prefix_meta = {
        **meta_common,
        "selected_dimension": len(prefix),
        "ranked_source_dimension": len(selected),
        "old_rank6_contained": all(
            prefix_span.contains(value) for value in old_vectors
        ),
        "basis_role": f"dimension-{args.dimension} prefix b0..b{args.dimension - 1}",
        "ranked_basis": str(args.ranked_output),
    }
    write_basis(args.ranked_output, ranked_meta, selected)
    if Path(args.output).resolve() != Path(args.ranked_output).resolve():
        write_basis(args.output, prefix_meta, prefix)
    print(f"Saved ranked basis: {args.ranked_output}", flush=True)
    if Path(args.output).resolve() != Path(args.ranked_output).resolve():
        print(f"Saved dimension-{args.dimension} prefix: {args.output}", flush=True)
    return selected, prefix_meta

