import numpy, json, utils
from cipher_config import state_bits, total_rounds, d_str, basis_number, weight_range, dim_truncated, top_basis_number, mode

# from sage.all import *
from sage.all import GF, Matrix, vector

def generate_x_basis(x_vectors):
    F = GF(2)

    if not x_vectors:
        return [], None

    M = Matrix(F, x_vectors)
    row_space = M.row_space()
    basis_vecs = row_space.basis()

    x_basis = [[int(x) for x in v] for v in basis_vecs]

    print("dim x =", len(x_basis))

    return x_basis, row_space

def get_mask_coordinate_in_basis(one_mask, row_space):
    F = GF(2)
    v = vector(F, one_mask)
    coord = row_space.coordinate_vector(v)

    return [int(x) for x in coord]

def coord_to_index(coord):
    idx = 0
    for i, bit in enumerate(coord):
        idx |= (bit << i)
    return idx

def mask_to_expression(mask):
    expr = []
    for i, bit in enumerate(mask):
        if mode == "permutation":
            if bit == 1:
                expr.append("x_{}".format(i))
        elif mode == "equal":
            if bit == 1:
                if i in range(64, 128):
                    expr.append("k0_{}".format(i - 64))
                elif i in range(128, 192):
                    expr.append("k1_{}".format(i - 128))
                elif i in range(192, 256):
                    expr.append("n_{}".format(i - 192))
                else:
                    print("Error ! Trail is illegal in this mode !")
        elif mode == "unrestricted":
            if bit == 1:
                if i in range(64, 128):
                    expr.append("k0_{}".format(i - 64))
                elif i in range(128, 192):
                    expr.append("k1_{}".format(i - 128))
                elif i in range(192, 256):
                    expr.append("n0_{}".format(i - 192))
                elif i in range(256, 320):
                    expr.append("n1_{}".format(i - 256))
                else:
                    print("Error ! Trail is illegal in this mode !")
    if not expr:
        return "0"
    return " + ".join(expr)

def inverse_walsh_probability_distribution_fast(fourier_coefficients):
    """
    Input length must be 2^d. The output is the unnormalized Walsh transform.
    """
    a = fourier_coefficients[:]
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
    """Return the in-place float64 FWHT of a list or NumPy array."""
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

def collect_nonzero_probability_records_fast(probs, x_basis):
    """Build expressions only for nonzero entries of a NumPy array."""
    # probs is already a NumPy array.
    nonzero_idx = numpy.where((probs > 0) | (probs < -0))[0]
    dim_x = len(x_basis)

    str_split = "-" * 100
    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] : {dim_x}")
    print(f"{len(nonzero_idx)} non-zero idx")
    print(str_split)

    records = []
    for idx in nonzero_idx:
        idx = int(idx)
        x_coord = index_to_coord(idx, dim_x)
        conds = [f"{mask_to_expression(x_basis[i])}={x_coord[i]}" for i in range(dim_x)]
        records.append({"index": idx, "coord": x_coord, "probability": float(probs[idx]), "conditions_text": conds})
    return records

# =====================================================================================
from datetime import datetime

def step_3_compute_probability_distribution():
    str_split = "-" * 100
    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(f"Step 3. Get probability distribution")

    # Read step2 output.
    step2_path = f"output/record_bases_and_contributions/step2_bases_contributions_r{total_rounds}_{d_str}_mode_{mode}_basis_number_{basis_number}_weight_{weight_range}_dim_{dim_truncated}.jsonl"
    with open(step2_path, "r") as f:
        data = json.loads(f.readline())
    x_basis = data["x_basis"]
    dim_x = data["dim_x"]
    ranked = data["ranked"]
    coord_list = data["coord_list"]
    print(f"Loaded step3: dim_x={dim_x}, coord_list size={len(coord_list)}")

    # Project onto the first dim basis vectors.
    # use_dim = top_basis_nubmer if hasattr(__builtins__, 'dim') else min(top_basis_nubmer, dim_x)
    use_dim = min(top_basis_number, dim_x)
    print(f"Taking top K={use_dim} basis vectors")
    top_K = ranked[:use_dim]

    # "slice":
    #   Project every Fourier coefficient onto the selected top-K basis.
    #   Contributions involving omitted basis directions are folded into the
    #   resulting K-dimensional score.
    #
    # "marginal":
    #   Average over omitted coordinates.  Fourier coefficients whose masks
    #   involve any omitted direction do not contribute.
    projection_mode = "marginal"

    fourier = numpy.zeros(1 << use_dim, dtype=numpy.float64)

    if projection_mode == "slice":
        for idx, coeff in coord_list:
            sub = 0
            for new_bit, old_bit in enumerate(top_K):
                if (idx >> old_bit) & 1:
                    sub |= 1 << new_bit
            fourier[sub] += coeff

    elif projection_mode == "marginal":
        selected_mask = 0
        for old_bit in top_K:
            selected_mask |= 1 << old_bit

        all_mask = (1 << dim_x) - 1
        omitted_mask = all_mask ^ selected_mask

        for idx, coeff in coord_list:
            # Unselected Fourier coordinates must be zero.
            if idx & omitted_mask:
                continue

            sub = 0
            for new_bit, old_bit in enumerate(top_K):
                if (idx >> old_bit) & 1:
                    sub |= 1 << new_bit

            fourier[sub] += coeff

    else:
        raise ValueError(
            f"Invalid projection_mode: {projection_mode}"
        )

    print(
        f"K-dim fourier: mode={projection_mode}, "
        f"non-zero={sum(1 for x in fourier if x != 0.0)}"
    )
    # ---- FWHT ----
    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]  Fourier transformation (K={use_dim})")
    print(str_split)

    # probability_distribution = inverse_walsh_probability_distribution_fast(fourier)
    probability_distribution = fwht_numpy(fourier)
    x_basis_K = [x_basis[i] for i in top_K]

    print()

    # record_path = f"output/record_C_and_P/step4_compute_probability_distribution_r{total_rounds}_{d_str}_cid_{c_id}_cweight_{c_weight_list[c_id]}_dim_{use_dim}.jsonl"
    # record_cp = {
    #     "component_id": c_id,
    #     "weight": c_weight_list[c_id],
    #     "x_dim": use_dim,
    #     "x_basis": x_basis_K,
    #     "fourier_coefficients": fourier.tolist() if isinstance(fourier, numpy.ndarray) else fourier,
    #     "probability_distribution": probability_distribution.tolist() if isinstance(probability_distribution, numpy.ndarray) else probability_distribution
    # }
    # with open(record_path, "a") as f:
    #     f.write(json.dumps(record_cp) + "\n")
    # print(f"C and P records saved to: {record_path}")

    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(f"Probability distribution:")
    print(str_split)

    # Locate extrema directly because formatting millions of expressions is expensive.
    probs_arr = probability_distribution
    abs_arr = numpy.abs(probs_arr)
    # Positive range.
    # pos_mask = probs_arr > 1e-15
    pos_mask = probs_arr > 0
    pos_max_idx_global = None
    pos_min_val = None
    pos_min_idx_global = None
    if numpy.any(pos_mask):
        pos_arr = probs_arr[pos_mask]
        pos_max_idx_global = int(numpy.argmax(probs_arr))
        pos_min_val = numpy.min(pos_arr)
        pos_min_idx_global = int(numpy.where(probs_arr == pos_min_val)[0][0])
    # Negative range.
    neg_mask = probs_arr < 0
    neg_min_idx_global = None
    neg_max_val = None
    neg_max_idx_global = None
    if numpy.any(neg_mask):
        neg_arr = probs_arr[neg_mask]
        neg_min_idx_global = int(numpy.argmin(probs_arr))
        neg_max_val = numpy.max(neg_arr)
        neg_max_idx_global = int(numpy.where(probs_arr == neg_max_val)[0][0])

    print(f"Distribution stats:")
    print(f"  total non-zero: {numpy.sum(abs_arr > 0)}")
    if numpy.any(pos_mask):
        print(
            f"  positive range: {pos_min_val:+.6e} ~ {probs_arr[pos_max_idx_global]:+.6e}  count={numpy.sum(pos_mask)}")
    if numpy.any(neg_mask):
        print(
            f"  negative range: {probs_arr[neg_min_idx_global]:+.6e} ~ {neg_max_val:+.6e}  count={numpy.sum(neg_mask)}")

    def make_one_record(idx_val):
        xc = index_to_coord(idx_val, use_dim)
        conds = [
            f"{mask_to_expression(x_basis_K[i])}={xc[i]}"
            for i in range(use_dim)
        ]
        return {
            "index": idx_val,
            "coord": xc,
            "probability": float(probs_arr[idx_val]),
            "conditions_text": conds,
        }

    def print_record(label, idx_val):
        rec = make_one_record(idx_val)
        prob = rec["probability"]
        s = "" if prob > 0 else "-"
        print(f"{label}: index={rec['index']}, x = {rec['coord']}")
        print(
            f"{label}: P[{', '.join(rec['conditions_text'])}] = {prob} = {s}2^{numpy.log2(abs(prob))}")
        # print(f"{label}: expressions = {rec['conditions_text']}")
        print(f"{label}: expressions:")
        for expr in rec["conditions_text"]:
            print(f"    {json.dumps(expr)},")
        print()

    if numpy.any(pos_mask):
        print_record("p_pos_max", pos_max_idx_global)
        print_record("p_pos_min", pos_min_idx_global)
    if numpy.any(neg_mask):
        print_record("p_neg_min", neg_min_idx_global)
        print_record("p_neg_max", neg_max_idx_global)

    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(str_split)

    nonzero_idx = numpy.where((probability_distribution > 0) | (probability_distribution < 0))[0]

    record_sparse_path = f"output/record_C_and_P_sparse/step3_probability_nonzero_r{total_rounds}_{d_str}_mode_{mode}_basis_number_{basis_number}_weight_{weight_range}_dim_{dim_truncated}_top_{top_basis_number}_projection_{projection_mode}.jsonl"
    record_sparse = {
        "basis_number": basis_number,
        "weight": weight_range,
        "dim_truncated": dim_truncated,
        "top_basis_number": top_basis_number,
        "x_dim": use_dim,
        "x_basis": x_basis_K,
        "nonzero_probability_count": len(nonzero_idx),
        "p_pos_max": make_one_record(pos_max_idx_global) if numpy.any(pos_mask) else None,
        "p_pos_min": make_one_record(pos_min_idx_global) if numpy.any(pos_mask) else None,
        "p_neg_min": make_one_record(neg_min_idx_global) if numpy.any(neg_mask) else None,
        "p_neg_max": make_one_record(neg_max_idx_global) if numpy.any(neg_mask) else None,
    }
    with open(record_sparse_path, "w") as f:
        f.write(json.dumps(record_sparse) + "\n")
    print(f"Non-zero P records saved to: {record_sparse_path}")

    for label in ("p_pos_max", "p_pos_min", "p_neg_min", "p_neg_max"):
        record = record_sparse[label]
        if record is None:
            continue
        constraints_path = (
            f"output/record_C_and_P_sparse/"
            f"{d_str}_{mode}_{projection_mode}_{label}.constraints.txt"
        )
        with open(constraints_path, "w") as f:
            f.write("\n".join(record["conditions_text"]) + "\n")
        print(f"Constraints saved to: {constraints_path}")

    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")


step_3_compute_probability_distribution()
