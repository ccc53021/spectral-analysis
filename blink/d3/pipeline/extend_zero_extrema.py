"""Add the attained zero minima and readable complete coset equations.

This postprocessing preserves the completed maxima and positive minima. It
does not change the distribution algorithm, paper, or completed histogram.
"""

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np

HERE = Path(__file__).resolve().parent
from .key_coset_witness import KeyCosetBuilder, solve_affine
from .joint.coset_witness import WitnessBuilder, model
from .distribution_join import syndrome
from .middle.analyze import fixed, core
from .middle.crosscheck_global_rank import independent_rank


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def rational(record):
    return Fraction(record["numerator"], record["denominator"])


def describe(probability):
    p = Fraction(probability)
    if not p:
        return {"numerator": 0, "denominator": 1, "text": "0",
                "log2_probability": None, "log2_extended": "-infinity",
                "power2_approximation": "0 (no finite base-2 exponent)",
                "zero_note": "Zero has no finite real base-2 exponent; the extended notation is 2^(-infinity)."}
    exponent = math.log2(p.numerator) - math.log2(p.denominator)
    return {"numerator": p.numerator, "denominator": p.denominator,
            "text": str(p), "log2_probability": exponent,
            "power2_approximation": f"2^({exponent:.12f})"}


def power_factor(count):
    count = int(count)
    if not count:
        return "0"
    shift = (count & -count).bit_length()-1
    odd = count >> shift
    return (f"2^{shift}" if odd == 1 else f"{odd} * 2^{shift}") if shift else str(odd)


def conditions(document, kind):
    if kind == "key":
        return [(int(row["master_key_mask_hex"],16),int(row["rhs"]))
                for row in document["all_114_linear_conditions"]]
    return list(zip([int(m,16) for m in document["joint_coset_basis_physical_hex"]],
                    map(int,document["joint_coset_rhs_bits"])))


def independent_members(document, kind, members=16):
    equations = conditions(document, kind)
    masks, rhs = zip(*equations)
    bits = 448 if kind == "key" else 512
    expected_rank = 114 if kind == "key" else 206
    assert independent_rank(masks) == expected_rank
    if kind == "key":
        keys = [int(k,16) for k in document["physical_RK_representative_hex"]]
        base = sum(k << (128+64*i) for i,k in enumerate(keys))
    else:
        keys = [int(k,16) for k in document["physical_round_keys"]]
        base = int(document["X"],16) | sum(k << (192+64*i) for i,k in enumerate(keys))
    assert all(core.parity(mask & base) == bit for mask,bit in equations)
    rng = random.Random(202609150000 + bits)
    records = []
    for index in range(members):
        raw = rng.getrandbits(bits)
        correction,_ = solve_affine(masks,[core.parity(mask & raw) for mask in masks])
        member = base ^ raw ^ correction
        assert all(core.parity(mask & member) == bit for mask,bit in equations)
        offset = 128 if kind == "key" else 192
        member_keys = [(member >> (offset+64*i)) & core.MASK64 for i in range(5)]
        # Independently use the pre-existing complete-six-round contraction,
        # not this run's new 18-dimensional spectrum or quotient lookup.
        check = fixed.compute(member_keys, include_terms=False, verification=False)
        assert check["exact_probability"]["numerator"] == 0
        if kind == "joint":
            plaintext = member & core.MASK64
            whitening1 = (member >> 64) & core.MASK64
            x = plaintext ^ whitening1
            assert bool(model.evaluate(x,member_keys,block="head"))
        records.append({"index":index,"full_physical_member_hex":hex(member),
                        "all_equations_hold":True,"independent_old_middle_probability":check["exact_probability"],
                        "direct_head_H4_equals_one": True if kind == "joint" else None})
    assert len({r["full_physical_member_hex"] for r in records}) == members
    return {"scope":"Independent physical full-coset member checks with fresh old exact middle contractions; not Monte Carlo estimates.",
            "equation_rank":expected_rank,"members":members,"seed":202609150000+bits,
            "all_passed":True,"records":records}


def readable(document, kind, title, class_count):
    rank = 114 if kind == "key" else 206
    p = rational(document["probability"] if kind == "key" else document["joint_coset_constant_model_probability"])
    names = ["W1","W2","RK1","RK2","RK3","RK4","RK5"] if kind == "key" else ["P","W1","W2","RK1","RK2","RK3","RK4","RK5"]
    eqs = conditions(document,kind)
    assert len(eqs) == rank
    lines = [f"# {title}", "", "Model: scheme A with retained 4+6+4 internal differences, zero tweak, and two zero outer value-connector masks.", "",
             f"This file gives one complete {rank}-dimensional linear quotient class, not only a representative key or partial conditions. Its model probability is {p}; {class_count} cosets ({power_factor(class_count)}) attain this value in the full distribution.", "",
             f"Each coset contains 2^{448-rank if kind=='key' else 512-rank} {'448-bit master keys' if kind=='key' else '(P,K) pairs'}.", "",
             "Every variable is a 64-bit unsigned word. par(mask & V) is the xor of the selected bits of V, and xor denotes addition modulo two.", "",
             "All equations below must hold simultaneously. An omitted variable has a zero mask; it is not fixed to zero. Bit numbering starts at the least-significant bit, and masks use 16 hexadecimal digits.", ""]
    if kind == "joint":
        lines += ["Input whitening is included: X=P xor W1. Therefore the input mask occurs on both P and W1; the output-whitening W2 mask is zero.", ""]
    else:
        lines += ["W1 and W2 do not occur below: at zero tweak, endpoint whitening does not alter the input-averaged model probability, so both whitening words are free.", ""]
    lines += ["```text"]
    for index,(mask,rhs) in enumerate(eqs,1):
        terms = []
        for word,name in enumerate(names):
            local = (mask >> (64*word)) & core.MASK64
            if local:
                terms.append(f"par(0x{local:016x} & {name})")
        lines.append(f"{index:03d}: " + " xor ".join(terms) + f" = {rhs}")
    lines += ["```", "", "These conditions describe only the displayed coset. The global count does not duplicate this representative; the complete distribution and conditional selection records count the other classes attaining the same value.", ""]
    return "\n".join(lines)


def run():
    started = time.perf_counter()
    out = HERE / "output/final_join"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["outer_distribution_complete"] and summary["extremal_coset_conditions_complete"]
    cosets = out / "extremal_cosets"
    preserved_files = [cosets / f"{kind}_{which}.json" for kind in ("key","joint") for which in ("maximum","minimum_nonzero")]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in preserved_files}
    preserved_statistics = {kind:{which:summary[kind][which].copy() for which in ("maximum","minimum_nonzero")} for kind in ("key","joint")}
    kb,jb = KeyCosetBuilder(),WitnessBuilder()
    sat = [0x2874cc1e532a2747,0x1062f52829c95f66,0x18c078facb7bfa85,0x73ae068f503c26b9,0x90479091156bcc2d]
    shared8 = syndrome(kb.middle_basis,kb.pack_keys(sat)) & 255
    remainder = int(np.flatnonzero(kb.middle_table[shared8] == 0)[0])
    key_zero = kb.construct(sat[:2],sat[3:],shared8|(remainder<<8),checks=16)
    assert key_zero["factors"]["head"]["numerator"] > 0
    assert key_zero["factors"]["tail"]["numerator"] > 0
    assert key_zero["factors"]["middle"]["numerator"] == 0
    key_zero.update(scope="One globally attaining minimum-zero key coset of the complete scheme-A model",global_minimum_verified=True,
                    global_attaining_coset_count=summary["key"]["zero_cosets"],zero_cause="middle exact probability is zero; head and tail averaged probabilities remain positive")
    joint_zero = jb.construct(representative_keys=key_zero["physical_RK_representative_hex"],verification_members=16)
    assert joint_zero["middle_probability"]["numerator"] == 0 and joint_zero["tail_probability"]["numerator"] > 0
    joint_zero.update(scope="One globally attaining minimum-zero joint coset of the complete scheme-A model",global_minimum_verified=True,
                      global_attaining_coset_count=summary["joint"]["zero_cosets"],zero_cause="H4=1 throughout the coset and tail probability is positive, but the exact middle probability is zero",
                      global_extremum_status="Global minimum is exactly zero: the model is nonnegative and this complete affine coset attains zero.")
    checks = {}
    for kind,document in (("key",key_zero),("joint",joint_zero)):
        checks[kind] = independent_members(document,kind)
        document["independent_zero_coset_verification"] = {k:v for k,v in checks[kind].items() if k != "records"}
        write_json(cosets / f"{kind}_minimum_zero.json",document)
    write_json(out / "zero_coset_member_verification.json",checks)
    numeric = {"model":summary["model"],"minimum_convention":"minimum includes zero; minimum_nonzero remains a separate statistic",
               "mean_definition":"Arithmetic mean over all equally sized cosets of the complete quotient, including zero cosets; not the average of the extrema.",
               "all_statistics_exact":True,"zero_exponent_note":"0 has no finite log2 value; log2_extended=-infinity is an extended-real notation only.",
               "key_distribution_representation":"complete conditional factorization and exact value-frequency query; not flattened",
               "joint_distribution_representation":"all 163513 probability values and exact counts flattened",
               "objects":{}}
    docs = {"key":{"minimum":key_zero},"joint":{"minimum":joint_zero}}
    for kind in ("key","joint"):
        docs[kind].update({which:json.loads((cosets / f"{kind}_{which}.json").read_text(encoding="utf-8")) for which in ("maximum","minimum_nonzero")})
        rank = summary[kind]["quotient_rank"]
        summary[kind]["minimum"] = {"probability":{"numerator":0,"denominator":1,"text":"0"},
                                     "coset_count":summary[kind]["zero_cosets"],
                                     "explicit_coset_conditions_file":f"extremal_cosets/{kind}_minimum_zero.json",
                                     "readable_coset_conditions_file":f"extremal_cosets/{kind}_minimum_zero.md"}
        obj = {"quotient_rank":rank,"total_cosets_decimal":str(1<<rank),"total_cosets_power2":f"2^{rank}",
               "zero_cosets_decimal":str(summary[kind]["zero_cosets"]),"zero_cosets_factorization":power_factor(summary[kind]["zero_cosets"]),
               "positive_cosets_decimal":str(summary[kind]["positive_cosets"]),"positive_cosets_factorization":power_factor(summary[kind]["positive_cosets"]),
               "mean_over_all_cosets":describe(rational(summary[kind]["mean"])),"statistics":{}}
        for which,label in (("maximum","maximum probability"),("minimum","minimum probability including zero"),("minimum_nonzero","minimum nonzero probability")):
            document = docs[kind][which]
            probability = summary[kind][which]["probability"]
            count = summary[kind][which]["coset_count"]
            suffix = "minimum_zero" if which == "minimum" else which
            md_name = f"{kind}_{suffix}.md"
            scope = "Master-key" if kind == "key" else "Joint input-key"
            text = readable(document,kind,f"{scope} complete coset conditions for the {label}",count)
            (cosets / md_name).write_text(text,encoding="utf-8")
            # Parse every printed equation back to its physical bit mask;
            # readable documents must preserve all equations, not only length.
            names = ["W1","W2","RK1","RK2","RK3","RK4","RK5"] if kind=="key" else ["P","W1","W2","RK1","RK2","RK3","RK4","RK5"]
            printed = [line for line in text.splitlines() if re.match(r"^\d{3}: ",line)]
            exact_equations = conditions(document,kind)
            assert len(printed) == len(exact_equations) == rank
            for line,(mask,rhs) in zip(printed,exact_equations):
                reconstructed = 0
                for hexadecimal,word in re.findall(r"par\(0x([0-9a-f]{16}) & (\w+)\)",line):
                    reconstructed ^= int(hexadecimal,16) << (64*names.index(word))
                assert reconstructed == mask and int(line.rsplit(" = ",1)[1]) == rhs
            summary[kind][which]["readable_coset_conditions_file"] = "extremal_cosets/"+md_name
            factors = document["factors"] if kind == "key" else {"head_mean_at_representative":document["head_mean_at_representative"],"middle":document["middle_probability"],"tail":document["tail_probability"]}
            obj["statistics"][which] = {"probability":describe(rational(probability)),"coset_count_decimal":str(count),
                                          "coset_count_factorization":power_factor(count),"representative_factors":{k:describe(rational(v)) for k,v in factors.items()},
                                          "head_indicator_in_joint_coset":1 if kind=="joint" else None,
                                          "json_conditions_file":f"extremal_cosets/{kind}_{suffix}.json","readable_conditions_file":"extremal_cosets/"+md_name}
        numeric["objects"][kind] = obj
    summary["minimum_convention"] = "minimum is 0 and counts all zero-probability cosets; minimum_nonzero is retained separately"
    summary["numeric_probability_summary_file"] = "probability_summary_with_zero.json"
    summary["zero_coset_conditions_complete"] = True
    summary["all_six_extremum_condition_documents_complete"] = True
    for kind in ("key","joint"):
        for which in ("maximum","minimum_nonzero"):
            for field in ("probability","coset_count"):
                assert summary[kind][which][field] == preserved_statistics[kind][which][field]
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path,expected in hashes.items())
    numeric["verification"] = {"zero_coset_independent_members_per_object":16,
                               "constructor_members_per_object":16,"existing_four_extremal_json_files_unchanged":True,
                               "readable_equations_exactly_roundtrip_to_JSON":960,
                               "existing_maximum_and_positive_minimum_values_and_counts_unchanged":True,
                               "all_passed":True,"elapsed_seconds":time.perf_counter()-started}
    write_json(out / "probability_summary_with_zero.json",numeric)
    write_json(out / "summary.json",summary)
    status_path = out / "coset_finalization_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["state"] == "complete_verified"
        status.update(explicit_extremal_cosets=6,zero_minimum_extension_complete=True,
                      readable_complete_condition_documents=6,zero_coset_independent_member_checks_passed=True)
        write_json(status_path,status)
    print(json.dumps({"state":"complete_verified","minimum":0,"key_zero_cosets":str(summary["key"]["zero_cosets"]),
                      "joint_zero_cosets":str(summary["joint"]["zero_cosets"]),"readable_condition_documents":6,
                      "existing_extrema_preserved":True,"elapsed_seconds":time.perf_counter()-started},indent=2),flush=True)


if __name__ == "__main__":
    run()
