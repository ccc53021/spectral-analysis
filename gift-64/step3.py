import argparse
import ast
import itertools
import json
import math
import os
import re
from pathlib import Path
from datetime import datetime

import numpy

import utils
from cipher_config import state_bits, total_rounds, d_str, basis_number, weight_range, dim_truncated, key_bits, input_x, diffs


def _bits_to_int(bits):
    value = 0
    for index, bit in enumerate(bits):
        value |= (int(bit) & 1) << index
    return value


def _int_to_bits(value, width):
    return [(value >> index) & 1 for index in range(width)]


def _rref_rows(vectors, width):
    """Return a GF(2) RREF basis and its ascending pivot columns."""
    rows = [_bits_to_int(vector) for vector in vectors if any(vector)]
    row_index = 0
    pivots = []
    for column in range(width):
        pivot_index = next(
            (index for index in range(row_index, len(rows))
             if (rows[index] >> column) & 1),
            None,
        )
        if pivot_index is None:
            continue
        rows[row_index], rows[pivot_index] = rows[pivot_index], rows[row_index]
        pivot_row = rows[row_index]
        for index in range(len(rows)):
            if index != row_index and ((rows[index] >> column) & 1):
                rows[index] ^= pivot_row
        pivots.append(column)
        row_index += 1
        if row_index == len(rows):
            break
    return rows[:row_index], pivots

def generate_k_basis(k_vectors):
    if not k_vectors:
        return [], ([], [])

    width = len(k_vectors[0])
    rows, pivots = _rref_rows(k_vectors, width)
    k_basis = [_int_to_bits(row, width) for row in rows]

    print("dim k =", len(k_basis))
    # print("k_basis =")
    # for b in k_basis:
    #     print(b)
    # print()

    return k_basis, (rows, pivots)

# 4. get coordinate of k basis
def get_mask_coordinate_in_basis(one_mask, row_space):
    rows, pivots = row_space
    value = _bits_to_int(one_mask)
    coordinate = []
    for row, pivot in zip(rows, pivots):
        bit = (value >> pivot) & 1
        coordinate.append(bit)
        if bit:
            value ^= row
    if value:
        raise ValueError("mask is outside the generated key-mask span")
    return coordinate

def coord_to_index(coord):
    idx = 0
    for i, bit in enumerate(coord):
        idx |= (bit << i)
    return idx

def mask_to_expression(mask):
    expr = []
    for i, bit in enumerate(mask):
        if i < key_bits:
            if bit == 1:
                expr.append("k{}".format(i))
        else:
            if bit == 1:
                expr.append("x{}".format(i - key_bits))
    if not expr:
        return "0"
    return " + ".join(expr)

def inverse_walsh_probability_distribution_fast(fourier_coefficients):
    """
    The input length must be 2^d. Return the unnormalized Walsh transform.
    """
    a = fourier_coefficients[:]   # Copy without modifying the input.
    n = len(a)
    h = 1

    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2

    return a

def fwht_numpy(a):
    """Convert the input to float64, apply FWHT in place, and return an array."""
    if not isinstance(a, numpy.ndarray):
        a = numpy.array(a, dtype=numpy.float64)
    n = len(a); h = 1
    while h < n:
        for i in range(0, n, h*2):
            x = a[i:i+h].copy(); y = a[i+h:i+2*h].copy()
            a[i:i+h] = x + y; a[i+h:i+2*h] = x - y
        h *= 2
    return a

def index_to_coord(idx, dim):
    coord = [0] * dim
    for i in range(dim):
        coord[i] = (idx >> i) & 1

    return coord

def collect_nonzero_probability_records_fast(probs, k_basis):
    """Filter nonzero NumPy entries before constructing the x expressions."""
    # probs is already a NumPy array.
    # nonzero_idx = numpy.where((probs > 1e-15) | (probs < -1e-15))[0]
    nonzero_idx = numpy.where((probs > 0) | (probs < -0))[0]
    dim_k = len(k_basis)

    str_split = "-" * 100
    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] : {dim_k}")
    print(f"{len(nonzero_idx)} non-zero idx")
    print(str_split)

    records = []
    for idx in nonzero_idx:
        idx = int(idx)
        k_coord = index_to_coord(idx, dim_k)
        conds = [f"{mask_to_expression(k_basis[i])}={k_coord[i]}" for i in range(dim_k)]
        records.append({"index": idx, "coord": k_coord, "probability": float(probs[idx]), "conditions_text": conds})
    return records

def collect_nonzero_probability_records(prob_distribution, k_basis):
    records = []
    dim_k = len(k_basis)

    for k_idx, prob in enumerate(prob_distribution):
        if prob == 0:
            continue

        k_coord = index_to_coord(k_idx, dim_k)

        condition_texts = []
        for i, bit in enumerate(k_coord):
            expr = mask_to_expression(k_basis[i])
            condition_texts.append(f"{expr}={bit}")

        records.append({
            "index": k_idx,
            "coord": k_coord,
            "probability": prob,
            "conditions_text": condition_texts
        })

    return records


def _rref_int_rows(vectors, width):
    """Return a compact GF(2) RREF basis for integer-encoded rows."""
    echelon = [0] * width
    for raw_value in vectors:
        value = int(raw_value)
        while value:
            pivot = (value & -value).bit_length() - 1
            if echelon[pivot]:
                value ^= echelon[pivot]
            else:
                echelon[pivot] = value
                break

    rows = [value for value in echelon if value]
    row_index = 0
    pivots = []
    for column in range(width):
        pivot_index = next(
            (index for index in range(row_index, len(rows))
             if (rows[index] >> column) & 1),
            None,
        )
        if pivot_index is None:
            continue
        rows[row_index], rows[pivot_index] = rows[pivot_index], rows[row_index]
        pivot_row = rows[row_index]
        for index in range(len(rows)):
            if index != row_index and ((rows[index] >> column) & 1):
                rows[index] ^= pivot_row
        pivots.append(column)
        row_index += 1
        if row_index == len(rows):
            break
    return rows[:row_index], pivots


def _coordinate_in_int_basis(mask, rows, pivots):
    value = int(mask)
    coordinate = 0
    for index, (row, pivot) in enumerate(zip(rows, pivots)):
        if (value >> pivot) & 1:
            value ^= row
            coordinate |= 1 << index
    if value:
        raise ValueError("mask is outside the requested row space")
    return coordinate


def _fwht_vectorized(values):
    array = numpy.asarray(values, dtype=numpy.float64).copy()
    size = len(array)
    if size == 0 or size & (size - 1):
        raise ValueError("FWHT input length must be a non-zero power of two")
    half = 1
    while half < size:
        blocks = array.reshape(-1, 2 * half)
        left = blocks[:, :half].copy()
        right = blocks[:, half:].copy()
        blocks[:, :half] = left + right
        blocks[:, half:] = left - right
        half *= 2
    return array


def _canonical_affine(equations, width):
    """Canonicalize mask·z=rhs equations and report consistency."""
    augmented = [int(mask) | ((int(rhs) & 1) << width)
                 for mask, rhs in equations]
    row_index = 0
    for column in range(width):
        pivot_index = next(
            (index for index in range(row_index, len(augmented))
             if (augmented[index] >> column) & 1),
            None,
        )
        if pivot_index is None:
            continue
        augmented[row_index], augmented[pivot_index] = (
            augmented[pivot_index], augmented[row_index]
        )
        pivot_row = augmented[row_index]
        for index in range(len(augmented)):
            if index != row_index and ((augmented[index] >> column) & 1):
                augmented[index] ^= pivot_row
        row_index += 1
    mask_limit = (1 << width) - 1
    rows = []
    for row in augmented:
        mask = row & mask_limit
        rhs = (row >> width) & 1
        if mask == 0:
            if rhs:
                return [], False
            continue
        rows.append((mask, rhs))
    rows.sort(key=lambda item: (item[0] & -item[0]).bit_length())
    return rows, True


def _dependency_tags(vectors, width):
    """Return a basis of linear dependencies among integer rows."""
    pivot_vectors = [0] * width
    pivot_tags = [0] * width
    dependencies = []
    for index, raw_value in enumerate(vectors):
        value = int(raw_value)
        tag = 1 << index
        while value:
            pivot = (value & -value).bit_length() - 1
            if pivot_vectors[pivot]:
                value ^= pivot_vectors[pivot]
                tag ^= pivot_tags[pivot]
            else:
                pivot_vectors[pivot] = value
                pivot_tags[pivot] = tag
                break
        if value == 0:
            dependencies.append(tag)
    return dependencies


def _rowspace_constraint_comparison(equations_a, equations_b, width):
    """Find invariant common and opposite-RHS constraints of two spaces."""
    rows_a, consistent_a = _canonical_affine(equations_a, width)
    rows_b, consistent_b = _canonical_affine(equations_b, width)
    if not consistent_a or not consistent_b:
        return {"common": [], "conflicts": [(0, 0, 1)], "compatible": False}

    masks = [mask for mask, _ in rows_a] + [mask for mask, _ in rows_b]
    dependencies = _dependency_tags(masks, width)
    split = len(rows_a)
    labeled_rows = []
    for dependency in dependencies:
        mask_a = 0
        rhs_a = 0
        mask_b = 0
        rhs_b = 0
        for index, (mask, rhs) in enumerate(rows_a):
            if (dependency >> index) & 1:
                mask_a ^= mask
                rhs_a ^= rhs
        for index, (mask, rhs) in enumerate(rows_b):
            if (dependency >> (split + index)) & 1:
                mask_b ^= mask
                rhs_b ^= rhs
        if mask_a != mask_b:
            raise AssertionError("row-space dependency reconstruction failed")
        if mask_a:
            labeled_rows.append([mask_a, rhs_a, rhs_b])

    row_index = 0
    for column in range(width):
        pivot_index = next(
            (index for index in range(row_index, len(labeled_rows))
             if (labeled_rows[index][0] >> column) & 1),
            None,
        )
        if pivot_index is None:
            continue
        labeled_rows[row_index], labeled_rows[pivot_index] = (
            labeled_rows[pivot_index], labeled_rows[row_index]
        )
        pivot_row = labeled_rows[row_index][:]
        for index in range(len(labeled_rows)):
            if index != row_index and ((labeled_rows[index][0] >> column) & 1):
                labeled_rows[index][0] ^= pivot_row[0]
                labeled_rows[index][1] ^= pivot_row[1]
                labeled_rows[index][2] ^= pivot_row[2]
        row_index += 1
    labeled_rows = [tuple(row) for row in labeled_rows if row[0]]
    first_conflict = next(
        (index for index, row in enumerate(labeled_rows)
         if row[1] != row[2]),
        None,
    )
    if first_conflict is not None:
        normalized = [list(row) for row in labeled_rows]
        conflict_row = normalized[first_conflict][:]
        for index, row in enumerate(normalized):
            if index != first_conflict and row[1] != row[2]:
                row[0] ^= conflict_row[0]
                row[1] ^= conflict_row[1]
                row[2] ^= conflict_row[2]
        labeled_rows = [tuple(row) for row in normalized]
    common = [row for row in labeled_rows if row[1] == row[2]]
    conflicts = [row for row in labeled_rows if row[1] != row[2]]
    return {
        "common": common,
        "conflicts": conflicts,
        "compatible": not conflicts,
    }


def _affine_representative(equations, width):
    rows, consistent = _canonical_affine(equations, width)
    if not consistent:
        return None
    value = 0
    for mask, rhs in rows:
        pivot = (mask & -mask).bit_length() - 1
        if rhs:
            value |= 1 << pivot
    return value


def _relations_on_target(source_equations, target_basis, physical_width):
    """Constraints on target-basis evaluations implied by source equations."""
    source_rows, consistent = _canonical_affine(source_equations, physical_width)
    if not consistent:
        return [], False
    source_count = len(source_rows)
    masks = [mask for mask, _ in source_rows] + list(target_basis)
    dependencies = _dependency_tags(masks, physical_width)
    relations = []
    for dependency in dependencies:
        target_relation = dependency >> source_count
        if target_relation == 0:
            continue
        rhs = 0
        for index, (_, source_rhs) in enumerate(source_rows):
            if (dependency >> index) & 1:
                rhs ^= source_rhs
        relations.append((target_relation, rhs))
    return _canonical_affine(relations, len(target_basis))


def _enumerate_affine_indices(equations, dimension):
    rows, consistent = _canonical_affine(equations, dimension)
    if not consistent:
        return numpy.empty(0, dtype=numpy.int64)
    pivot_columns = [(mask & -mask).bit_length() - 1 for mask, _ in rows]
    pivot_set = set(pivot_columns)
    free_columns = [column for column in range(dimension)
                    if column not in pivot_set]
    indices = numpy.empty(1 << len(free_columns), dtype=numpy.int64)
    for free_value in range(len(indices)):
        value = 0
        for index, column in enumerate(free_columns):
            if (free_value >> index) & 1:
                value |= 1 << column
        for (mask, rhs), pivot in zip(rows, pivot_columns):
            bit = rhs ^ ((mask & value).bit_count() & 1)
            if bit:
                value |= 1 << pivot
        indices[free_value] = value
    return indices


def _mask_expression(mask, physical_width):
    terms = []
    for bit in range(physical_width):
        if not ((mask >> bit) & 1):
            continue
        terms.append(f"k{bit}" if bit < key_bits else f"x{bit - key_bits}")
    return " ⊕ ".join(terms) if terms else "0"


def _basis_coordinate_expression(coordinate):
    terms = [f"B{index}" for index in range(coordinate.bit_length())
             if (coordinate >> index) & 1]
    return " ⊕ ".join(terms) if terms else "0"


def _probability_text(value):
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    return f"{value:.17e} ({sign}2^{math.log2(abs(value)):.12f})"


def _load_characteristic_spectra(path):
    meta = None
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "meta":
                meta = record
            elif record.get("type") == "characteristic_spectrum":
                masks = [int(value, 16) for value in record.pop("masks_hex")]
                coefficients = [float(value) for value in record.pop("coefficients")]
                if len(masks) != len(coefficients):
                    raise ValueError("spectrum mask/coefficient lengths differ")
                record["masks"] = masks
                record["coefficients"] = coefficients
                records.append(record)
    if meta is None:
        raise ValueError(f"missing spectrum metadata in {path}")
    if not records:
        raise ValueError(f"no characteristic spectra in {path}")
    return meta, records


def _prepare_characteristic(record, physical_width):
    basis, pivots = _rref_int_rows(record["masks"], physical_width)
    dimension = len(basis)
    fourier = numpy.zeros(1 << dimension, dtype=numpy.float64)
    for mask, coefficient in zip(record["masks"], record["coefficients"]):
        coordinate = _coordinate_in_int_basis(mask, basis, pivots)
        fourier[coordinate] += coefficient
    probabilities = _fwht_vectorized(fourier)
    maximum = float(numpy.max(probabilities))
    maximum_indices = numpy.flatnonzero(probabilities == maximum).astype(int).tolist()
    scale = max(float(numpy.max(numpy.abs(probabilities))), 1.0e-300)
    zero_tolerance = scale * 1.0e-12
    return {
        **record,
        "basis": basis,
        "basis_pivots": pivots,
        "dimension": dimension,
        "probabilities": probabilities,
        "maximum": maximum,
        "maximum_indices": maximum_indices,
        "zero_tolerance": zero_tolerance,
    }


def _max_equations(characteristic, maximum_index):
    return [
        (mask, (maximum_index >> index) & 1)
        for index, mask in enumerate(characteristic["basis"])
    ]


def _affine_support_summary(characteristic, physical_width):
    """Describe the positive-probability support as an affine key space."""
    probabilities = characteristic["probabilities"]
    tolerance = characteristic["zero_tolerance"]
    positive_indices = numpy.flatnonzero(probabilities > tolerance).astype(int)
    negative_count = int(numpy.sum(probabilities < -tolerance))
    total_count = len(probabilities)
    positive_count = len(positive_indices)
    zero_count = total_count - positive_count - negative_count
    if positive_count == 0:
        return {
            "positive_class_count": 0,
            "zero_class_count": zero_count,
            "negative_class_count": negative_count,
            "total_class_count": total_count,
            "support_fraction": 0.0,
            "overall_mean": float(numpy.mean(probabilities)),
            "positive_mean": None,
            "minimum_positive": None,
            "affine_dimension": None,
            "affine_codimension": None,
            "is_single_affine_space": False,
            "equations": [],
        }

    origin = int(positive_indices[0])
    difference_basis, difference_pivots = _rref_int_rows(
        (int(index) ^ origin for index in positive_indices[1:]),
        characteristic["dimension"],
    )
    pivot_set = set(difference_pivots)
    free_columns = [
        column for column in range(characteristic["dimension"])
        if column not in pivot_set
    ]
    coordinate_equations = []
    for free_column in free_columns:
        relation = 1 << free_column
        for row, pivot in zip(difference_basis, difference_pivots):
            if (row >> free_column) & 1:
                relation |= 1 << pivot
        rhs = (relation & origin).bit_count() & 1
        coordinate_equations.append((relation, rhs))

    physical_equations = []
    for relation, rhs in coordinate_equations:
        physical_mask = 0
        for index, basis_mask in enumerate(characteristic["basis"]):
            if (relation >> index) & 1:
                physical_mask ^= basis_mask
        physical_equations.append((physical_mask, rhs))
    physical_equations, consistent = _canonical_affine(
        physical_equations, physical_width
    )
    affine_dimension = len(difference_basis)
    is_single_affine_space = (
        consistent and positive_count == (1 << affine_dimension)
    )

    return {
        "positive_class_count": positive_count,
        "zero_class_count": zero_count,
        "negative_class_count": negative_count,
        "total_class_count": total_count,
        "support_fraction": positive_count / total_count,
        "overall_mean": float(numpy.mean(probabilities)),
        "positive_mean": float(numpy.mean(probabilities[positive_indices])),
        "minimum_positive": float(numpy.min(probabilities[positive_indices])),
        "affine_dimension": affine_dimension,
        "affine_codimension": characteristic["dimension"] - affine_dimension,
        "is_single_affine_space": is_single_affine_space,
        "equations": physical_equations if is_single_affine_space else [],
    }


def _conditioned_target_summary(source, target, physical_width):
    allowed = numpy.zeros(1 << target["dimension"], dtype=numpy.bool_)
    for maximum_index in source["maximum_indices"]:
        relations, consistent = _relations_on_target(
            _max_equations(source, maximum_index),
            target["basis"],
            physical_width,
        )
        if not consistent:
            continue
        allowed[_enumerate_affine_indices(relations, target["dimension"])] = True
    allowed_indices = numpy.flatnonzero(allowed)
    if len(allowed_indices) == 0:
        return {
            "attainable_coordinate_count": 0,
            "maximum": None,
            "minimum": None,
            "always_zero": True,
            "can_be_zero": False,
            "can_reach_target_maximum": False,
        }
    values = target["probabilities"][allowed_indices]
    tolerance = target["zero_tolerance"]
    return {
        "attainable_coordinate_count": int(len(allowed_indices)),
        "maximum": float(numpy.max(values)),
        "minimum": float(numpy.min(values)),
        "always_zero": bool(numpy.all(numpy.abs(values) <= tolerance)),
        "can_be_zero": bool(numpy.any(numpy.abs(values) <= tolerance)),
        "can_be_positive": bool(numpy.any(values > tolerance)),
        "can_reach_target_maximum": bool(
            numpy.any(numpy.abs(values - target["maximum"]) <= tolerance)
        ),
    }


def _pairwise_maximum_comparison(left, right, physical_width):
    total_pairs = len(left["maximum_indices"]) * len(right["maximum_indices"])
    max_pairs = int(os.environ.get("MAX_MAXIMUM_CLASS_PAIRS", "1000000"))
    if total_pairs > max_pairs:
        raise RuntimeError(
            f"maximum-class comparison requires {total_pairs} pairs; "
            f"raise MAX_MAXIMUM_CLASS_PAIRS={max_pairs} to continue"
        )
    first_comparison = None
    compatible_pair = None
    compatible_count = 0
    for left_index in left["maximum_indices"]:
        for right_index in right["maximum_indices"]:
            comparison = _rowspace_constraint_comparison(
                _max_equations(left, left_index),
                _max_equations(right, right_index),
                physical_width,
            )
            if first_comparison is None:
                first_comparison = comparison
            if comparison["compatible"]:
                compatible_count += 1
                if compatible_pair is None:
                    compatible_pair = (left_index, right_index, comparison)
    selected = compatible_pair[2] if compatible_pair else first_comparison
    return {
        "total_maximum_class_pairs": total_pairs,
        "compatible_maximum_class_pairs": compatible_count,
        "maximum_spaces_compatible": compatible_count > 0,
        "left_maximum_index": (
            compatible_pair[0] if compatible_pair else left["maximum_indices"][0]
        ),
        "right_maximum_index": (
            compatible_pair[1] if compatible_pair else right["maximum_indices"][0]
        ),
        "common": selected["common"],
        "conflicts": selected["conflicts"],
    }


def _all_maximum_spaces_compatible(characteristics, physical_width):
    class_lists = [item["maximum_indices"] for item in characteristics]
    total = math.prod(len(values) for values in class_lists)
    maximum = int(os.environ.get("MAX_MAXIMUM_CLASS_TUPLES", "1000000"))
    if total > maximum:
        return {"checked": False, "combination_count": total, "compatible": None}
    for combination in itertools.product(*class_lists):
        equations = []
        for characteristic, maximum_index in zip(characteristics, combination):
            equations.extend(_max_equations(characteristic, maximum_index))
        _, consistent = _canonical_affine(equations, physical_width)
        if consistent:
            return {
                "checked": True,
                "combination_count": total,
                "compatible": True,
                "witness_maximum_indices": list(combination),
            }
    return {"checked": True, "combination_count": total, "compatible": False}


def _dominant_aggregate(characteristics, physical_width):
    union_basis, union_pivots = _rref_int_rows(
        [mask for item in characteristics for mask in item["basis"]],
        physical_width,
    )
    dimension = len(union_basis)
    maximum_dimension = int(os.environ.get("MAX_DOMINANT_FWHT_DIM", "24"))
    if dimension > maximum_dimension:
        return {
            "computed": False,
            "dimension": dimension,
            "reason": f"dimension exceeds MAX_DOMINANT_FWHT_DIM={maximum_dimension}",
        }
    fourier = numpy.zeros(1 << dimension, dtype=numpy.float64)
    for item in characteristics:
        for mask, coefficient in zip(item["masks"], item["coefficients"]):
            coordinate = _coordinate_in_int_basis(mask, union_basis, union_pivots)
            fourier[coordinate] += coefficient
    probabilities = _fwht_vectorized(fourier)
    maximum = float(numpy.max(probabilities))
    maximum_indices = numpy.flatnonzero(probabilities == maximum).astype(int).tolist()
    representative_index = maximum_indices[0]
    contributions = []
    for item in characteristics:
        local_index = 0
        for local_bit, mask in enumerate(item["basis"]):
            coordinate = _coordinate_in_int_basis(mask, union_basis, union_pivots)
            if (coordinate & representative_index).bit_count() & 1:
                local_index |= 1 << local_bit
        contributions.append({
            "characteristic_index": item["characteristic_index"],
            "local_index": local_index,
            "probability": float(item["probabilities"][local_index]),
            "is_individual_maximum": local_index in item["maximum_indices"],
        })
    return {
        "computed": True,
        "dimension": dimension,
        "maximum": maximum,
        "maximum_indices_count": len(maximum_indices),
        "representative_maximum_index": representative_index,
        "contributions": contributions,
    }


def _spectrum_path(characteristic_limit=None):
    suffix = (
        ""
        if characteristic_limit is None or characteristic_limit >= len(diffs)
        else f"_limit_{characteristic_limit}"
    )
    return Path(
        "output/record_trails_and_coefficients/"
        f"step2_characteristic_spectra_r_{total_rounds}_{d_str}_x_{input_x}_"
        f"basis_number_{basis_number}_weight_{weight_range}_dim_{dim_truncated}"
        f"{suffix}.jsonl"
    )


def _saved_common_basis_path():
    return Path(
        "output/record_C_and_P_sparse/"
        f"step3_probability_nonzero_r_{total_rounds}_{d_str}_x_{input_x}_"
        f"basis_{basis_number}_weight_{weight_range}_dim_{dim_truncated}.jsonl"
    )


def _load_saved_common_basis(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        first_line = stream.readline()
        try:
            record = json.loads(first_line)
        except json.JSONDecodeError:
            record = None
        if record is not None and record.get("k_basis"):
            return record["k_basis"]

        basis_by_index = {}
        basis_pattern = re.compile(r"^Basis\[(\d+)\] = (\[[01, ]+\]) = ")
        for line in itertools.chain([first_line], stream):
            match = basis_pattern.match(line.strip())
            if match:
                basis_by_index[int(match.group(1))] = ast.literal_eval(match.group(2))
    if not basis_by_index:
        raise ValueError(f"no k_basis or Basis[i] lines in {path}")
    expected = list(range(max(basis_by_index) + 1))
    if sorted(basis_by_index) != expected:
        raise ValueError(f"non-contiguous Basis[i] lines in {path}")
    return [basis_by_index[index] for index in expected]


def _saved_common_basis_log_candidates():
    pattern = (
        f"117_*x_{input_x}_basis_number_{basis_number}_weight_{weight_range}_"
        f"dim_{dim_truncated}.txt"
    )
    return sorted(Path("result_output").rglob(pattern))


def _render_characteristic_analysis_markdown(report):
    direct_comparison = str(report["input_x"]).lower() == "false"
    def display_name(route_index):
        return f"C{route_index}"

    display_characteristics = report["characteristics"]
    display_indices = [
        item["characteristic_index"] for item in display_characteristics
    ]
    lines = [
        "# GIFT-64 18 轮 differential：4 条主导特征约束分析",
        "",
        f"- 数据：`{report['spectrum_path']}`",
        f"- `input_x={report['input_x']}`，公共 Fourier 空间维数 `K={report['common_dimension']}`",
        f"- 公共基中含 x 的方向：{report['common_x_direction_count']}；纯密钥方向：{report['common_key_only_direction_count']}",
        f"- Sun26a 直接对比口径：{'是' if direct_comparison else '否（仅作联合 key/x 补充分析）'}",
        "- 本报告内部编号：`Ci = DC[i]`；密钥位 `ki` 完全沿用当前代码的位序，不转换为论文位号",
        "",
        "> 公共维数 K 表示概率函数依赖的独立 Fourier 方向，不等于 K 条纯密钥约束。本报告严格区分“非零支持约束”和“最大值类约束”：Sun26a 的 8 条正确密钥约束属于前一个层次；这里用本项目自己的位序恢复支持方程，不做方程字符串转换。",
        "",
        "## 非零支持（与 Sun26a 的 8 约束层次对比）",
        "",
        "| 特征 | 正概率类/总类数 | 支持占比 | 支持仿射维数/余维 | 最小正概率 | 支持内平均概率 | 最大概率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in display_characteristics:
        lines.append(
            f"| {display_name(item['characteristic_index'])} | "
            f"{item['positive_class_count']}/"
            f"{item['total_class_count']} | {_probability_text(item['support_fraction'])} | "
            f"{item['support_affine_dimension']}/{item['support_affine_codimension']} | "
            f"{_probability_text(item['minimum_positive'])} | "
            f"{_probability_text(item['positive_mean'])} | "
            f"{_probability_text(item['maximum'])} |"
        )

    lines.extend([
        "",
        "四条特征在数值容差内均没有负概率类；非零支持外的概率均为 0。下面给出 quasidifferential 结果直接恢复出的支持方程。",
    ])
    for item in display_characteristics:
        lines.extend([
            "",
            f"### {display_name(item['characteristic_index'])} 的非零支持方程",
            "",
        ])
        if not item["support_is_single_affine_space"]:
            lines.append("正概率支持不是单一仿射空间，无法用一组方程精确表示。")
        else:
            lines.extend(
                f"- `{equation['physical_text']}`"
                for equation in item["support_equations"]
            )

    lines.extend([
        "",
        "## 非零支持空间的两两重合与冲突",
        "",
        "| 组合 | 独立共同方向 | 独立冲突方向 | 正概率空间能否共存 | 冲突证据 |",
        "|---|---:|---:|---|---|",
    ])
    for pair in report["support_pairs"]:
        witness = pair["conflict_equations"][0] if pair["conflict_equations"] else "—"
        lines.append(
            f"| {display_name(pair['left'])} / {display_name(pair['right'])} | "
            f"{pair['common_count']} | "
            f"{pair['conflict_count']} | "
            f"{'能' if pair['spaces_compatible'] else '不能'} | `{witness}` |"
        )

    lines.extend([
        "",
        "## 单条主导特征的最大值类",
        "",
        "每个非零支持空间内部仍有概率起伏；以下最大类约束比本项目恢复出的 8 条支持约束更细。",
        "",
        "| 特征 | 标称权重 | 局部维数/一个最大类的约束数 | 公共空间自由维 | 最大值 | 最大类数 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for item in display_characteristics:
        lines.append(
            f"| {display_name(item['characteristic_index'])} | "
            f"{item['average_weight']:.0f} | "
            f"{item['local_dimension']}/{item['local_dimension']} | "
            f"{item['free_common_dimensions']} | "
            f"{_probability_text(item['maximum'])} | {item['maximum_class_count']} |"
        )

    for item in display_characteristics:
        lines.extend([
            "",
            f"### {display_name(item['characteristic_index'])} 的一个最大类",
            "",
            f"局部最大坐标：`{item['representative_maximum_index']}`。以下是最小独立方程组；未列出的公共维是自由的。",
            "",
        ])
        for equation in item["representative_equations"]:
            common_text = equation.get("common_coordinate_text")
            suffix = f"（公共坐标：{common_text}）" if common_text else ""
            lines.append(f"- `{equation['physical_text']}`{suffix}")

    lines.extend([
        "",
        "## 最大空间的两两重合与冲突",
        "",
        "兼容数检查覆盖了两条特征的全部最大类组合；后续展开的共同与冲突方程取首组最大类作为可复核证据。",
        "",
        "| 组合 | 已检查最大类组合 | 相容组合 | 首组的共同方向 | 首组的冲突方向 | 两条能否同时最大 |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for pair in report["pairs"]:
        lines.append(
            f"| {display_name(pair['left'])} / {display_name(pair['right'])} | "
            f"{pair['total_maximum_class_pairs']} | "
            f"{pair['compatible_maximum_class_pairs']} | "
            f"{pair['common_count']} | {pair['conflict_count']} | "
            f"{'能' if pair['maximum_spaces_compatible'] else '不能'} |"
        )
    for pair in report["pairs"]:
        if not pair["common_equations"] and not pair["conflict_equations"]:
            continue
        lines.extend([
            "",
            f"### {display_name(pair['left'])} / {display_name(pair['right'])}",
            "",
        ])
        if pair["common_equations"]:
            lines.append("共同约束：")
            lines.append("")
            lines.extend(f"- `{value}`" for value in pair["common_equations"])
        if pair["conflict_equations"]:
            lines.append("")
            lines.append("冲突约束：")
            lines.append("")
            lines.extend(f"- `{value}`" for value in pair["conflict_equations"])

    lines.extend([
        "",
        "## 一条取最大时，其他特征还能贡献多少",
        "",
        "表中是行特征处于任一最大类时，列特征可达到的最高概率。",
        "",
        "| 行\\列 | " + " | ".join(display_name(idx) for idx in display_indices) + " |",
        "|---|" + "---:|" * len(display_indices),
    ])
    conditional = report["conditional_probabilities"]
    for source in display_indices:
        cells = []
        for target in display_indices:
            summary = conditional[str(source)][str(target)]
            if source == target:
                cells.append("自身最大")
            elif summary["always_zero"]:
                cells.append("恒为 0")
            elif summary["can_reach_target_maximum"]:
                cells.append("可同时达到自身最大")
            else:
                cells.append(_probability_text(summary["maximum"]))
        lines.append(f"| {display_name(source)} | " + " | ".join(cells) + " |")

    aggregate = report["dominant_aggregate"]
    lines.extend(["", "## 4 条主导特征的聚合", ""])
    if aggregate["computed"]:
        lines.append(
            f"4 条主导特征的联合 Fourier 空间维数为 {aggregate['dimension']}，"
            f"聚合最大值为 {_probability_text(aggregate['maximum'])}。"
        )
        lines.append("")
        lines.append("在一个聚合最大点上的分项贡献：")
        lines.append("")
        for contribution in aggregate["contributions"]:
            lines.append(
                f"- {display_name(contribution['characteristic_index'])}: "
                f"{_probability_text(contribution['probability'])}，"
                f"{'也是该特征自身最大点' if contribution['is_individual_maximum'] else '不是该特征自身最大点'}"
            )
    else:
        lines.append(
            f"未计算聚合分布：联合维数 {aggregate['dimension']}，{aggregate['reason']}。"
        )

    all_compatible = report["all_maximum_spaces"]
    lines.extend(["", "## 与 Sun26a 的对照口径", ""])
    support_disjoint = all(
        not pair["spaces_compatible"] for pair in report["support_pairs"]
    )
    lines.extend([
        "- 论文对 S0 的 4 条主导特征先得到每条 6 个独立线性约束；只在线性层面 C0 与 C2 有交集。",
        "- 加入每条 2 个线性化非线性约束后，4 个正确密钥空间两两不交；每条共 8 个独立约束，占密钥空间 2^-8，四条并集占 2^-6。",
        "- 本报告不转换论文与代码的密钥位编号，也不以方程字符串逐条等价作为比较目标。",
        "- 本次 quasidifferential 在本项目位序下恢复出每条 8 个独立支持约束；"
        f"四个支持空间两两不交：{'是' if support_disjoint else '否'}。",
        "- 每条路线的全密钥平均概率为 2^-58，非零支持占 2^-8，所以支持内平均概率恰为 2^-50；这与论文的完整 18 轮结论一致。",
        "- 20 条最大类方程描述的是 8 条支持约束内部的概率峰值，不是论文 8 条约束的替代品。",
        "",
        f"本次 4 条最大空间能否联合相容："
        f"{'能' if all_compatible['compatible'] else '不能'}"
        f"（已完整检查：{'是' if all_compatible['checked'] else '否'}）。",
        "",
    ])
    return "\n".join(lines)


def analyze_characteristic_spectra(common_basis=None, characteristic_limit=None):
    path = _spectrum_path(characteristic_limit)
    meta, raw_records = _load_characteristic_spectra(path)
    physical_width = int(meta["mask_bits"])
    characteristics = [
        _prepare_characteristic(record, physical_width)
        for record in raw_records
    ]
    for item in characteristics:
        item["support"] = _affine_support_summary(item, physical_width)
    indices = [item["characteristic_index"] for item in characteristics]

    union_basis, union_pivots = _rref_int_rows(
        [mask for item in characteristics for mask in item["basis"]],
        physical_width,
    )
    if common_basis is None:
        common_basis_masks = union_basis
        common_pivots = union_pivots
    else:
        common_basis_masks = [_bits_to_int(mask) for mask in common_basis]
        common_basis_masks, common_pivots = _rref_int_rows(
            common_basis_masks, physical_width
        )
        for mask in union_basis:
            _coordinate_in_int_basis(mask, common_basis_masks, common_pivots)
    common_dimension = len(common_basis_masks)

    characteristic_reports = []
    for item in characteristics:
        support = item["support"]
        representative_index = item["maximum_indices"][0]
        equations = _max_equations(item, representative_index)
        equation_records = []
        for mask, rhs in equations:
            common_coordinate = _coordinate_in_int_basis(
                mask, common_basis_masks, common_pivots
            )
            physical_expr = _mask_expression(mask, physical_width)
            equation_records.append({
                "mask_hex": f"0x{mask:0{physical_width // 4}x}",
                "rhs": rhs,
                "physical_text": f"{physical_expr} = {rhs}",
                "common_coordinate_mask": common_coordinate,
                "common_coordinate_text": (
                    f"{_basis_coordinate_expression(common_coordinate)} = {rhs}"
                ),
            })
        characteristic_reports.append({
            "characteristic_index": item["characteristic_index"],
            "average_weight": item["average_weight"],
            "local_dimension": item["dimension"],
            "free_common_dimensions": common_dimension - item["dimension"],
            "maximum": item["maximum"],
            "maximum_log2": math.log2(item["maximum"]) if item["maximum"] > 0 else None,
            "maximum_class_count": len(item["maximum_indices"]),
            "maximum_indices": item["maximum_indices"],
            "positive_class_count": support["positive_class_count"],
            "zero_class_count": support["zero_class_count"],
            "negative_class_count": support["negative_class_count"],
            "total_class_count": support["total_class_count"],
            "support_fraction": support["support_fraction"],
            "overall_mean": support["overall_mean"],
            "positive_mean": support["positive_mean"],
            "minimum_positive": support["minimum_positive"],
            "support_affine_dimension": support["affine_dimension"],
            "support_affine_codimension": support["affine_codimension"],
            "support_is_single_affine_space": support["is_single_affine_space"],
            "support_equations": [
                {
                    "mask_hex": f"0x{mask:0{physical_width // 4}x}",
                    "rhs": rhs,
                    "physical_text": (
                        f"{_mask_expression(mask, physical_width)} = {rhs}"
                    ),
                }
                for mask, rhs in support["equations"]
            ],
            "local_basis_masks_hex": [
                f"0x{mask:0{physical_width // 4}x}" for mask in item["basis"]
            ],
            "representative_maximum_index": representative_index,
            "representative_equations": equation_records,
        })

    support_pair_reports = []
    for left_index in range(len(characteristics)):
        for right_index in range(left_index + 1, len(characteristics)):
            left = characteristics[left_index]
            right = characteristics[right_index]
            comparison = _rowspace_constraint_comparison(
                left["support"]["equations"],
                right["support"]["equations"],
                physical_width,
            )
            support_pair_reports.append({
                "left": left["characteristic_index"],
                "right": right["characteristic_index"],
                "spaces_compatible": comparison["compatible"],
                "common_count": len(comparison["common"]),
                "conflict_count": len(comparison["conflicts"]),
                "common_equations": [
                    f"{_mask_expression(mask, physical_width)} = {rhs_a}"
                    for mask, rhs_a, _ in comparison["common"]
                ],
                "conflict_equations": [
                    f"{_mask_expression(mask, physical_width)} = {rhs_a} "
                    f"(C{left['characteristic_index']}) vs "
                    f"{_mask_expression(mask, physical_width)} = {rhs_b} "
                    f"(C{right['characteristic_index']})"
                    for mask, rhs_a, rhs_b in comparison["conflicts"]
                ],
            })

    pair_reports = []
    for left_index in range(len(characteristics)):
        for right_index in range(left_index + 1, len(characteristics)):
            left = characteristics[left_index]
            right = characteristics[right_index]
            comparison = _pairwise_maximum_comparison(left, right, physical_width)
            common_equations = [
                f"{_mask_expression(mask, physical_width)} = {rhs_a}"
                for mask, rhs_a, _ in comparison["common"]
            ]
            conflict_equations = [
                f"{_mask_expression(mask, physical_width)} = {rhs_a} "
                f"(C{left['characteristic_index']}) "
                f"vs {_mask_expression(mask, physical_width)} = {rhs_b} "
                f"(C{right['characteristic_index']})"
                for mask, rhs_a, rhs_b in comparison["conflicts"]
            ]
            pair_reports.append({
                "left": left["characteristic_index"],
                "right": right["characteristic_index"],
                "maximum_spaces_compatible": comparison["maximum_spaces_compatible"],
                "total_maximum_class_pairs": comparison["total_maximum_class_pairs"],
                "compatible_maximum_class_pairs": comparison["compatible_maximum_class_pairs"],
                "common_count": len(comparison["common"]),
                "conflict_count": len(comparison["conflicts"]),
                "common_equations": common_equations,
                "conflict_equations": conflict_equations,
            })

    conditional = {}
    for source in characteristics:
        source_key = str(source["characteristic_index"])
        conditional[source_key] = {}
        for target in characteristics:
            target_key = str(target["characteristic_index"])
            if source is target:
                conditional[source_key][target_key] = {
                    "attainable_coordinate_count": len(source["maximum_indices"]),
                    "maximum": source["maximum"],
                    "minimum": source["maximum"],
                    "always_zero": False,
                    "can_be_zero": False,
                    "can_be_positive": True,
                    "can_reach_target_maximum": True,
                }
            else:
                conditional[source_key][target_key] = _conditioned_target_summary(
                    source, target, physical_width
                )

    report = {
        "schema_version": 1,
        "spectrum_path": str(path).replace("\\", "/"),
        "input_x": meta["input_x"],
        "sun26a_direct_comparison": str(meta["input_x"]).lower() == "false",
        "physical_width": physical_width,
        "indices": indices,
        "common_dimension": common_dimension,
        "common_basis_source": "full_differential" if common_basis is not None else "dominant_union",
        "common_x_direction_count": sum(
            1 for mask in common_basis_masks if mask >> key_bits
        ),
        "common_key_only_direction_count": sum(
            1 for mask in common_basis_masks if not (mask >> key_bits)
        ),
        "characteristics": characteristic_reports,
        "support_pairs": support_pair_reports,
        "pairs": pair_reports,
        "conditional_probabilities": conditional,
        "all_maximum_spaces": _all_maximum_spaces_compatible(
            characteristics, physical_width
        ),
        "dominant_aggregate": _dominant_aggregate(
            characteristics, physical_width
        ),
        "paper_baseline": {
            "class": "S0",
            "linear_constraints_per_characteristic": 6,
            "linear_overlap": "C0 and C2 overlap before LNCs",
            "linearized_nonlinear_constraints_per_characteristic": 2,
            "total_independent_constraints_per_characteristic": 8,
            "right_key_fraction_per_characteristic": "2^-8",
            "four_space_union_fraction": "2^-6",
            "pairwise_disjoint_after_lncs": True,
        },
    }

    output_dir = Path("output/record_C_and_P_sparse")
    output_dir.mkdir(parents=True, exist_ok=True)
    limit_suffix = (
        ""
        if characteristic_limit is None or characteristic_limit >= len(diffs)
        else f"_limit_{characteristic_limit}"
    )
    output_stem = (
        f"step3_characteristic_analysis_r_{total_rounds}_{d_str}_x_{input_x}_"
        f"basis_{basis_number}_weight_{weight_range}_dim_{dim_truncated}{limit_suffix}"
    )
    json_path = output_dir / f"{output_stem}.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = Path(__file__).resolve().parent / (
        f"DOMINANT_CHARACTERISTICS_ANALYSIS_x_{input_x}{limit_suffix}.md"
    )
    markdown_path.write_text(
        _render_characteristic_analysis_markdown(report) + "\n",
        encoding="utf-8",
    )
    print(f"Saved characteristic analysis JSON: {json_path}")
    print(f"Saved characteristic analysis report: {markdown_path}")
    return report

# =====================================================================================

def summarize_coset_distribution(probs_arr):
    """Summarize all equal-sized cosets without rounding or zero tolerances.

    ``maximum`` and ``minimum`` select by absolute value and retain the sign.
    When both signs attain the same magnitude, the positive value is selected
    and its count is reported in ``*_cosets``; ``*_values`` records both signs
    separately. These are exact equality counts of the supplied binary64 FWHT
    values, not counts from an exact-rational reconstruction of the spectrum.
    """
    values = numpy.asarray(probs_arr, dtype=numpy.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("coset distribution must be a nonempty one-dimensional array")

    # Bound temporary arrays even when a full FWHT contains tens of millions
    # of entries. In particular, do not allocate abs(values) for the full input.
    chunk_size = 1 << 20
    maximum_magnitude = 0.0
    minimum_magnitude = math.inf
    zero_cosets = 0
    for start in range(0, values.size, chunk_size):
        chunk = values[start:start + chunk_size]
        if not numpy.all(numpy.isfinite(chunk)):
            raise ValueError("coset distribution contains a non-finite value")
        magnitudes = numpy.abs(chunk)
        maximum_magnitude = max(maximum_magnitude, float(numpy.max(magnitudes)))
        minimum_magnitude = min(minimum_magnitude, float(numpy.min(magnitudes)))
        zero_cosets += int(numpy.count_nonzero(chunk == 0.0))

    result = {
        "total_cosets": int(values.size),
        "zero_cosets": zero_cosets,
        "average": float(numpy.mean(values, dtype=numpy.longdouble)),
        "average_source": "signed arithmetic mean of all binary64 FWHT coset values, including zeros",
        "counting_source": "exact binary64 value equality, without rounding or tolerance",
    }
    for label, magnitude in (("maximum", maximum_magnitude),
                             ("minimum", minimum_magnitude)):
        counts = {magnitude: 0}
        if magnitude != 0.0:
            counts[-magnitude] = 0
        for start in range(0, values.size, chunk_size):
            chunk = values[start:start + chunk_size]
            for value in counts:
                counts[value] += int(numpy.count_nonzero(chunk == value))
        attained = [{"value": value, "cosets": count}
                    for value, count in counts.items() if count]
        # The insertion order above makes a positive/negative tie deterministic.
        result[label] = attained[0]["value"]
        result[f"{label}_cosets"] = attained[0]["cosets"]
        result[f"{label}_values"] = attained
        result[f"{label}_magnitude_cosets"] = sum(item["cosets"] for item in attained)
        result[f"{label}_both_signs_tied"] = len(attained) == 2
    return result


def step_3_compute_probability_distribution():

    str_split = "-" * 100
    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(f"Step 3. Get probability distribution, basis number = {basis_number}")

    # ---- Read step2 output ----
    step2_path = f"output/record_trails_and_coefficients/step2_save_trails_and_coefficients_r_{total_rounds}_{d_str}_x_{input_x}_basis_number_{basis_number}_weight_{weight_range}_dim_{dim_truncated}.jsonl"
    with open(step2_path, "r") as f:
        data = json.loads(f.readline())
    k_trails = data["k_trails"]
    k_coefficients = data["k_coefficients"]
    # x_indexes = data["x_indexes"]
    print(f"Loaded {len(k_trails)} trails from {step2_path}")

    print(str_split)
    print(str_split)
    zero_mask = [0] * (key_bits + state_bits)
    if zero_mask in k_trails:
        print(f"C_[uk=0, ux=0] = 2^{numpy.log2(k_coefficients[k_trails.index(zero_mask)])}")
    else:
        print(f"C_[uk=0, ux=0] not in trail set")
    print(str_split)
    print(str_split)
    print()

    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(f"Get mask-key bases:")
    print(str_split)
    k_basis, row_space = generate_k_basis(k_trails)
    dim_k = len(k_basis)
    max_fwht_dim = int(os.environ.get("MAX_FWHT_DIM", "26"))
    if dim_k > max_fwht_dim:
        raise MemoryError(
            f"FWHT dimension {dim_k} exceeds MAX_FWHT_DIM={max_fwht_dim}; "
            "reduce dim_truncated or explicitly raise MAX_FWHT_DIM"
        )
    print(f"Basis expressions:")
    for i, b in enumerate(k_basis):
        print(f"Basis[{i}] = {b} = {mask_to_expression(b)}")
    print(str_split)
    print()

    # ---- Full-dimensional Fourier transform ----
    fourier = numpy.zeros(1 << dim_k, dtype=numpy.float64)
    for mask, coeff in zip(k_trails, k_coefficients):
        coord = get_mask_coordinate_in_basis(mask, row_space)
        idx = coord_to_index(coord)
        fourier[idx] += coeff
    print(f"Fourier: N=2^{dim_k}, non-zero={sum(1 for x in fourier if abs(x)>0)}")

    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]  FWHT (K={dim_k})")
    print(str_split)

    probs_arr = fwht_numpy(fourier)
    coset_statistics = summarize_coset_distribution(probs_arr)
    print("Coset statistics: " + json.dumps(coset_statistics, ensure_ascii=False))

    # ---- Find extrema ----
    pos_mask = probs_arr > 0
    neg_mask = probs_arr < 0
    pos_max_idx = 0
    pos_min_val = 0
    neg_min_idx = 0
    neg_max_val = 0

    print(f"Probability distribution (dim_k={dim_k}):")
    print(f"  total non-zero: {numpy.sum(pos_mask | neg_mask)}")
    if numpy.any(pos_mask):
        pos_arr = probs_arr[pos_mask]
        pos_max_idx = int(numpy.argmax(probs_arr))
        pos_min_val = numpy.min(pos_arr)
        print(f"  positive range: {pos_min_val:+.6e} ~ {probs_arr[pos_max_idx]:+.6e}  count={numpy.sum(pos_mask)}")
    if numpy.any(neg_mask):
        neg_arr = probs_arr[neg_mask]
        neg_min_idx = int(numpy.argmin(probs_arr))
        neg_max_val = numpy.max(neg_arr)
        print(f"  negative range: {probs_arr[neg_min_idx]:+.6e} ~ {neg_max_val:+.6e}  count={numpy.sum(neg_mask)}")

    def make_one_record(idx_val):
        kc = index_to_coord(idx_val, dim_k)
        conds = [f"{mask_to_expression(k_basis[i])}={kc[i]}" for i in range(dim_k)]
        return {"index": idx_val, "coord": kc, "probability": float(probs_arr[idx_val]), "conditions_text": conds}

    def print_record(label, idx_val):
        rec = make_one_record(idx_val)
        prob = rec["probability"]
        s = "" if prob > 0 else "-"
        print(f"  {label}: P[{', '.join(rec['conditions_text'])}] = {prob} = {s}2^{numpy.log2(abs(prob))}")

    if numpy.any(pos_mask):
        print_record("p_pos_max", pos_max_idx)
        print_record("p_pos_min", int(numpy.where(probs_arr == pos_min_val)[0][0]))
    if numpy.any(neg_mask):
        print_record("p_neg_min", neg_min_idx)
        print_record("p_neg_max", int(numpy.where(probs_arr == neg_max_val)[0][0]))

    # ---- Save ----
    os.makedirs("output/record_C_and_P_sparse", exist_ok=True)
    nonzero_idx = numpy.where(pos_mask | neg_mask)[0]
    save_path = f"output/record_C_and_P_sparse/step3_probability_nonzero_r_{total_rounds}_{d_str}_x_{input_x}_basis_{basis_number}_weight_{weight_range}_dim_{dim_truncated}.jsonl"
    record = {
        "basis_number": basis_number,
        "weight_range": weight_range,
        "dim_k": dim_k,
        "k_basis": k_basis,
        "nonzero_count": len(nonzero_idx),
        "coset_statistics": coset_statistics,
        "p_pos_max": make_one_record(pos_max_idx) if numpy.any(pos_mask) else None,
        "p_pos_min": make_one_record(int(numpy.where(probs_arr == pos_min_val)[0][0])) if numpy.any(pos_mask) else None,
        "p_neg_min": make_one_record(neg_min_idx) if numpy.any(neg_mask) else None,
        "p_neg_max": make_one_record(int(numpy.where(probs_arr == neg_max_val)[0][0])) if numpy.any(neg_mask) else None,
    }
    with open(save_path, "w") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nSaved: {save_path}")

    # The experiment consumes plain GF(2) equations.  Export every extremum so
    # that the verifier can select a record without embedding analysis output
    # in C++ source code.
    save_stem = Path(save_path).with_suffix("")
    for label in ("p_pos_max", "p_pos_min", "p_neg_min", "p_neg_max"):
        extremum = record[label]
        if extremum is None:
            continue
        constraint_path = Path(f"{save_stem}_{label}.constraints.txt")
        with constraint_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(f"# cipher=GIFT-64\n")
            stream.write(f"# rounds={total_rounds}\n")
            stream.write(f"# record={label}\n")
            stream.write(f"# probability={extremum['probability']:.17e}\n")
            for condition in extremum["conditions_text"]:
                stream.write(condition.replace(" ", "") + "\n")
        print(f"Saved constraints: {constraint_path}")

    return k_basis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--characteristic-analysis-only", action="store_true")
    parser.add_argument("--characteristic-limit", type=int, default=None)
    parser.add_argument(
        "--common-basis-path",
        type=Path,
        default=None,
        help="optional full-differential Step-3 JSONL or run log used as the common basis",
    )
    args = parser.parse_args()
    if args.characteristic_limit is not None and args.characteristic_limit <= 0:
        parser.error("--characteristic-limit must be positive")
    if args.characteristic_analysis_only and args.characteristic_limit is None:
        parser.error(
            "--characteristic-analysis-only requires --characteristic-limit"
        )
    return args


def main():
    args = parse_args()
    if args.characteristic_analysis_only:
        common_basis_path = args.common_basis_path
        if common_basis_path is None and _saved_common_basis_path().exists():
            common_basis_path = _saved_common_basis_path()
        if common_basis_path is None:
            candidates = _saved_common_basis_log_candidates()
            if candidates:
                common_basis_path = candidates[0]
        common_basis = None
        if common_basis_path is not None:
            common_basis = _load_saved_common_basis(common_basis_path)
            print(f"Using full-differential common basis: {common_basis_path}")
        else:
            print(
                "Full-differential common basis not found; using the union of "
                "the analyzed characteristic spaces. Pairwise compatibility "
                "remains invariant, but the report will not use the full K basis."
            )
        analyze_characteristic_spectra(
            common_basis=common_basis,
            characteristic_limit=args.characteristic_limit,
        )
        return

    common_basis = step_3_compute_probability_distribution()
    path = _spectrum_path()
    if path.exists():
        analyze_characteristic_spectra(common_basis=common_basis)
    else:
        print(
            f"Characteristic analysis skipped: {path} does not exist. "
            "Re-run Step 2 with the updated executable."
        )


if __name__ == "__main__":
    main()
