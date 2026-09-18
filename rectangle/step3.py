import argparse
import json
import math
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path
from datetime import datetime

if __name__ == "__main__":
    # Do not hand failures to a host-installed crash-reporting hook (e.g. apport).
    sys.excepthook = sys.__excepthook__

import numpy

from cipher_config import state_bits, total_rounds, d_str, basis_number, weight_range, dim_truncated, key_bits, input_x, diffs
from cipher_config import cipher_name, key_model, round_key_bits, begin_round, key_bit_name


def _bits_to_int(bits):
    value = 0
    for index, bit in enumerate(bits):
        value |= (int(bit) & 1) << index
    return value


def _int_to_bits(value, width):
    return [(value >> index) & 1 for index in range(width)]


def _rref_rows(vectors, width):
    """Return a GF(2) RREF basis and its ascending pivot columns."""
    # Consume rows incrementally: retain at most `width` independent integers,
    # not a second list containing every input mask.
    return _rref_int_rows((_bits_to_int(vector) for vector in vectors), width)

def generate_k_basis(k_vectors):
    if not k_vectors:
        return [], ([], [])

    width = len(k_vectors[0])
    rows, pivots = _rref_rows(k_vectors, width)
    k_basis = [_int_to_bits(row, width) for row in rows]

    print("dim k =", len(k_basis))

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
                expr.append(key_bit_name(i))
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

WORK_CHUNK_ELEMENTS = 1 << 20
STREAM_HEADER_MAX_BYTES = 1 << 20
LEGACY_STEP2_MAX_BYTES = 64 << 20


def fwht_numpy(a, chunk_elements=WORK_CHUNK_ELEMENTS):
    """Unnormalized in-place float64 FWHT with bounded vectorized scratch space.

    Small butterflies are grouped into many rows per NumPy operation; large
    butterflies are split into chunks. Only the left operands are copied, so
    scratch storage never grows with the full transform length.
    """
    if type(chunk_elements) is not int or chunk_elements < 2:
        raise ValueError("FWHT chunk_elements must be an integer of at least two")
    a = numpy.asarray(a, dtype=numpy.float64)
    if a.ndim != 1 or not a.size or a.size & (a.size - 1):
        raise ValueError("FWHT input must be one-dimensional with power-of-two length")
    if not a.flags.c_contiguous or not a.flags.writeable:
        a = a.copy()
    n, half = a.size, 1
    with numpy.errstate(over="raise", invalid="raise"):
        while half < n:
            width = 2 * half
            if width <= chunk_elements:
                batch = max(1, chunk_elements // width) * width
                for start in range(0, n, batch):
                    rows = a[start:min(start + batch, n)].reshape(-1, width)
                    left, right = rows[:, :half], rows[:, half:]
                    saved_left = left.copy()
                    numpy.add(saved_left, right, out=left)
                    numpy.subtract(saved_left, right, out=right)
                    del saved_left
            else:
                for start in range(0, n, width):
                    for offset in range(0, half, chunk_elements):
                        length = min(chunk_elements, half - offset)
                        left = a[start + offset:start + offset + length]
                        right = a[start + half + offset:start + half + offset + length]
                        saved_left = left.copy()
                        numpy.add(saved_left, right, out=left)
                        numpy.subtract(saved_left, right, out=right)
                        del saved_left
            half *= 2
    return a

def index_to_coord(idx, dim):
    coord = [0] * dim
    for i in range(dim):
        coord[i] = (idx >> i) & 1

    return coord

def _rref_int_rows(vectors, width):
    """Return a compact GF(2) RREF basis for integer-encoded rows."""
    echelon = [0] * width
    for raw_value in vectors:
        value = int(raw_value)
        if value < 0 or value.bit_length() > width:
            raise ValueError("GF(2) row exceeds the configured mask width")
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


# =====================================================================================

def _check_fwht_dimension(dimension):
    if type(dimension) is not int or dimension < 0:
        raise ValueError("FWHT dimension must be a nonnegative integer")
    maximum = int(os.environ.get("MAX_FWHT_DIM", "26"))
    if maximum < 0:
        raise ValueError("MAX_FWHT_DIM must be nonnegative")
    if dimension > maximum:
        raise MemoryError(
            f"FWHT dimension {dimension} exceeds MAX_FWHT_DIM={maximum}; "
            "reduce dim_truncated or explicitly raise MAX_FWHT_DIM"
        )
    if (1 << dimension) > numpy.iinfo(numpy.intp).max // 8:
        raise MemoryError("FWHT array exceeds the platform addressable array size")


def _reject_json_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field: " + key)
        result[key] = value
    return result


def _expected_step2_config():
    return {
        "cipher": cipher_name, "key_model": key_model,
        "round_key_bits": round_key_bits, "key_bits": key_bits,
        "mask_bits": key_bits + state_bits, "total_rounds": total_rounds,
        "begin_round": begin_round, "input_x": str(input_x),
        "basis_number": basis_number, "weight": weight_range,
        "dim": dim_truncated,
    }


def _validate_config_fields(metadata, expected):
    if not isinstance(metadata, dict):
        raise ValueError("Step-2 metadata must be a JSON object")
    for name, wanted in expected.items():
        actual = metadata.get(name)
        if type(actual) is not type(wanted) or actual != wanted:
            raise ValueError(
                f"Step-2 configuration mismatch for {name}: "
                f"expected {wanted!r}, received {actual!r}"
            )


def _validate_stream_metadata(metadata):
    expected = _expected_step2_config()
    expected.update(type="rectangle_fourier", version=1, d_str=d_str,
                    characteristic_count=len(diffs), coefficient_dtype="<f8")
    _validate_config_fields(metadata, expected)
    dimension = metadata.get("dim_k")
    _check_fwht_dimension(dimension)
    width = key_bits + state_bits
    if dimension > min(width, dim_truncated * len(diffs)):
        raise ValueError("Stream basis dimension exceeds the configured trail span")
    count = metadata.get("entry_count")
    if type(count) is not int or count != 1 << dimension:
        raise ValueError("Stream entry_count must equal 2**dim_k")
    aggregated = metadata.get("aggregated_mask_count")
    nonzero = metadata.get("nonzero_coefficient_count")
    if (type(aggregated) is not int or type(nonzero) is not int
            or not 0 <= nonzero <= aggregated <= count):
        raise ValueError("Invalid stream aggregated/nonzero coefficient counts")
    encoded_rows, pivots = metadata.get("basis_masks_hex"), metadata.get("pivots")
    if (not isinstance(encoded_rows, list) or not isinstance(pivots, list)
            or len(encoded_rows) != dimension or len(pivots) != dimension):
        raise ValueError("Stream basis and pivot lengths must equal dim_k")
    if (any(type(pivot) is not int or not 0 <= pivot < width for pivot in pivots)
            or pivots != sorted(set(pivots))):
        raise ValueError("Stream pivots must be distinct, ascending mask-bit indices")
    rows = []
    for encoded in encoded_rows:
        if (not isinstance(encoded, str) or len(encoded) > (width + 3) // 4 + 2
                or re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]+", encoded) is None):
            raise ValueError("Invalid hexadecimal stream basis mask")
        row = int(encoded, 16)
        if row <= 0 or row.bit_length() > width:
            raise ValueError("Stream basis mask is zero or exceeds mask_bits")
        rows.append(row)
    for index, (row, pivot) in enumerate(zip(rows, pivots)):
        if (row & -row).bit_length() - 1 != pivot:
            raise ValueError("Stream basis is not ascending-pivot RREF")
        if any(((row >> other) & 1) != int(index == column)
               for column, other in enumerate(pivots)):
            raise ValueError("Stream basis pivot columns are not RREF/independent")
    return [_int_to_bits(row, width) for row in rows]


def _count_finite_nonzero(values, chunk_elements=WORK_CHUNK_ELEMENTS):
    count = 0
    for start in range(0, len(values), chunk_elements):
        chunk = values[start:start + chunk_elements]
        if not numpy.isfinite(chunk).all():
            raise ValueError("Fourier coefficients must all be finite")
        count += int(numpy.count_nonzero(chunk))
    return count


def read_step2_fourier_stream(stream):
    """Read the bounded JSON header plus raw little-endian float64 payload.

    The caller must also check the producer's exit status. EOF is required here
    so extra stdout/log bytes cannot silently become a valid probability result.
    """
    header = stream.readline(STREAM_HEADER_MAX_BYTES + 1)
    if not header or len(header) > STREAM_HEADER_MAX_BYTES or not header.endswith(b"\n"):
        raise ValueError("Missing, oversized, or unterminated Step-2 stream header")
    metadata = json.loads(header.decode("utf-8"), parse_constant=_reject_json_constant,
                          object_pairs_hook=_unique_json_object)
    k_basis = _validate_stream_metadata(metadata)
    count = metadata["entry_count"]
    print(f"Receiving Fourier stream: K={metadata['dim_k']}, entries={count}, "
          f"aggregated masks={metadata['aggregated_mask_count']}", flush=True)
    fourier = numpy.empty(count, dtype=numpy.dtype("<f8"))
    raw = memoryview(fourier).cast("B")
    received = 0
    try:
        while received < raw.nbytes:
            end = min(received + 8 * WORK_CHUNK_ELEMENTS, raw.nbytes)
            amount = stream.readinto(raw[received:end])
            if amount is None:
                raise OSError("Step-2 stream readinto did not return a byte count")
            if type(amount) is not int or not 0 <= amount <= end - received:
                raise OSError("Step-2 stream returned an invalid byte count")
            if amount == 0:
                raise ValueError(
                    f"Truncated Step-2 Fourier payload: received {received} of {raw.nbytes} bytes"
                )
            received += amount
    finally:
        raw.release()
    if stream.read(1) != b"":
        raise ValueError("Unexpected trailing bytes after the Step-2 Fourier payload")
    nonzero = _count_finite_nonzero(fourier)
    if nonzero != metadata["nonzero_coefficient_count"]:
        raise ValueError("Stream nonzero_coefficient_count does not match the payload")
    return metadata, k_basis, fourier


def _load_legacy_step2_fourier(path):
    """Compatibility only: refuse large legacy monolithic JSON before reading."""
    path = Path(path)
    message = (
        "Legacy Step-2 JSON exceeds the 64 MiB compatibility limit; "
        "use the run_all.py Step2-to-Step3 streaming pipeline instead"
    )
    if path.stat().st_size > LEGACY_STEP2_MAX_BYTES:
        raise MemoryError(message)
    with path.open("rb") as stream:
        raw = stream.read(LEGACY_STEP2_MAX_BYTES + 1)
    if len(raw) > LEGACY_STEP2_MAX_BYTES:
        raise MemoryError(message)
    data = json.loads(raw, parse_constant=_reject_json_constant,
                      object_pairs_hook=_unique_json_object)
    del raw
    _validate_config_fields(data, _expected_step2_config())
    masks, coefficients = data.get("k_trails"), data.get("k_coefficients")
    if not isinstance(masks, list) or not isinstance(coefficients, list) or len(masks) != len(coefficients):
        raise ValueError("Invalid legacy mask/coefficient arrays")
    width = key_bits + state_bits
    for mask in masks:
        if (not isinstance(mask, list) or len(mask) != width
                or any(type(bit) is not int or bit not in (0, 1) for bit in mask)):
            raise ValueError("Invalid legacy expanded-key mask bits/width")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in coefficients):
        raise ValueError("Legacy Fourier coefficients must be finite numbers")
    k_basis, (rows, pivots) = generate_k_basis(masks)
    _check_fwht_dimension(len(k_basis))
    fourier = numpy.zeros(1 << len(k_basis), dtype=numpy.float64)
    for mask, coefficient in zip(masks, coefficients):
        fourier[_coordinate_in_int_basis(_bits_to_int(mask), rows, pivots)] += coefficient
    data = {key: value for key, value in data.items() if key not in ("k_trails", "k_coefficients")}
    data["aggregated_mask_count"] = len(masks)
    data["nonzero_coefficient_count"] = _count_finite_nonzero(fourier)
    return data, k_basis, fourier


def summarize_probabilities(values, chunk_elements=WORK_CHUNK_ELEMENTS):
    """Exact sign counts and first-occurrence extrema using bounded scratch."""
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("Probability chunk_elements must be positive")
    result = {"positive_count": 0, "negative_count": 0, "nonzero_count": 0}
    for label in ("p_pos_min", "p_pos_max", "p_neg_min", "p_neg_max"):
        result[label] = None
    for start in range(0, len(values), chunk_elements):
        chunk = values[start:start + chunk_elements]
        if not numpy.isfinite(chunk).all():
            raise ValueError("FWHT produced non-finite probabilities")
        for positive, count_name, low_label, high_label in (
            (True, "positive_count", "p_pos_min", "p_pos_max"),
            (False, "negative_count", "p_neg_min", "p_neg_max"),
        ):
            selected = chunk > 0 if positive else chunk < 0
            count = int(numpy.count_nonzero(selected))
            result[count_name] += count
            if not count:
                continue
            low = float(numpy.min(chunk, where=selected, initial=numpy.inf))
            high = float(numpy.max(chunk, where=selected, initial=-numpy.inf))
            for label, value, is_lower in ((low_label, low, True), (high_label, high, False)):
                current = result[label]
                if current is None or (value < current["value"] if is_lower else value > current["value"]):
                    # argmax over equality selects the same first index as the
                    # old where(...)[0][0], without an unbounded index array.
                    index = start + int(numpy.argmax(chunk == value))
                    result[label] = {"index": index, "value": value}
    result["nonzero_count"] = result["positive_count"] + result["negative_count"]
    return result


def _write_probability_outputs(save_path, record):
    """Stage all outputs first; publish the summary (success marker) last."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    stem = save_path.with_suffix("")
    outputs = []
    for label in ("p_pos_max", "p_pos_min", "p_neg_min", "p_neg_max"):
        extremum = record[label]
        if extremum is None:
            continue
        lines = [f"# cipher={cipher_name}",
                 f"# key_model={key_model}; kR_B = round R, state bit B (LSB-first)",
                 f"# begin_round={begin_round}; round_key_bits={round_key_bits}",
                 f"# rounds={total_rounds}", f"# record={label}",
                 f"# probability={extremum['probability']:.17e}"]
        lines.extend(condition.replace(" ", "") for condition in extremum["conditions_text"])
        outputs.append((Path(f"{stem}_{label}.constraints.txt"), "\n".join(lines) + "\n"))
    outputs.append((save_path, json.dumps(record, allow_nan=False) + "\n"))
    staged = []
    try:
        for destination, content in outputs:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=str(save_path.parent), prefix=".step3_",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                staged.append((temporary, destination))
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in staged:
            os.replace(str(temporary), str(destination))
    finally:
        for temporary, unused_destination in staged:
            if temporary.exists():
                temporary.unlink()
    for destination, unused_content in outputs:
        print(f"Saved: {destination}")


def step_3_compute_probability_distribution(from_step2_stream=False, stream=None):

    str_split = "-" * 100
    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(f"Step 3. Get probability distribution, basis number = {basis_number}")

    if from_step2_stream:
        data, k_basis, fourier = read_step2_fourier_stream(
            sys.stdin.buffer if stream is None else stream
        )
    else:
        if stream is not None:
            raise ValueError("A stream requires from_step2_stream=True")
        step2_path = f"output/record_trails_and_coefficients/step2_save_trails_and_coefficients_r_{total_rounds}_{d_str}_x_{input_x}_basis_number_{basis_number}_weight_{weight_range}_dim_{dim_truncated}.jsonl"
        data, k_basis, fourier = _load_legacy_step2_fourier(step2_path)
        print(f"Loaded {data['aggregated_mask_count']} masks from {step2_path}")
    if fourier[0] > 0:
        print(f"C_[uk=0, ux=0] = 2^{numpy.log2(fourier[0])}")
    else:
        print(f"C_[uk=0, ux=0] = {fourier[0]}")
    dim_k = len(k_basis)
    _check_fwht_dimension(dim_k)
    print(f"Basis expressions:")
    for i, b in enumerate(k_basis):
        print(f"Basis[{i}] = {b} = {mask_to_expression(b)}")
    print(str_split)
    print()

    print(f"Fourier: N=2^{dim_k}, non-zero={data['nonzero_coefficient_count']}")

    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]  FWHT (K={dim_k})")
    print(str_split)

    probs_arr = fwht_numpy(fourier)
    del fourier  # The transform is in place; do not retain a separate original.

    statistics = summarize_probabilities(probs_arr)
    print(f"Probability distribution (dim_k={dim_k}):")
    print(f"  total non-zero: {statistics['nonzero_count']}")
    for name, low, high in (("positive", "p_pos_min", "p_pos_max"),
                            ("negative", "p_neg_min", "p_neg_max")):
        if statistics[name + "_count"]:
            print(f"  {name} range: {statistics[low]['value']:+.6e} ~ "
                  f"{statistics[high]['value']:+.6e}  count={statistics[name + '_count']}")

    def make_one_record(idx_val):
        kc = index_to_coord(idx_val, dim_k)
        conds = [f"{mask_to_expression(k_basis[i])}={kc[i]}" for i in range(dim_k)]
        return {"index": idx_val, "coord": kc, "probability": float(probs_arr[idx_val]), "conditions_text": conds}

    def print_record(label, idx_val):
        rec = make_one_record(idx_val)
        prob = rec["probability"]
        s = "" if prob > 0 else "-"
        print(f"  {label}: P[{', '.join(rec['conditions_text'])}] = {prob} = {s}2^{numpy.log2(abs(prob))}")

    for label in ("p_pos_max", "p_pos_min", "p_neg_min", "p_neg_max"):
        if statistics[label] is not None:
            print_record(label, statistics[label]["index"])

    # ---- Save ----
    save_path = f"output/record_C_and_P_sparse/step3_probability_nonzero_r_{total_rounds}_{d_str}_x_{input_x}_basis_{basis_number}_weight_{weight_range}_dim_{dim_truncated}.jsonl"
    record = {
        "cipher": cipher_name,
        "key_model": key_model,
        "round_key_bits": round_key_bits,
        "key_bits": key_bits,
        "mask_bits": key_bits + state_bits,
        "total_rounds": total_rounds,
        "begin_round": begin_round,
        "input_x": input_x,
        "dim_truncated": dim_truncated,
        "basis_number": basis_number,
        "weight_range": weight_range,
        "dim_k": dim_k,
        "k_basis": k_basis,
        "nonzero_count": statistics["nonzero_count"],
    }
    for label in ("p_pos_max", "p_pos_min", "p_neg_min", "p_neg_max"):
        record[label] = (make_one_record(statistics[label]["index"])
                         if statistics[label] is not None else None)
    _write_probability_outputs(save_path, record)

    return k_basis


def parse_args():
    parser = argparse.ArgumentParser(description="Compute a RECTANGLE probability distribution")
    parser.add_argument("--from-step2-stream", action="store_true",
                        help="read a bounded JSON header and binary Fourier coefficients from stdin")
    return parser.parse_args()


def main():
    args = parse_args()
    step_3_compute_probability_distribution(from_step2_stream=args.from_step2_stream)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Step 3 interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
