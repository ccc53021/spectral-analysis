"""Construct H4=1 inside any specified middle-tail key coset.

All 68 middle-tail syndromes are preserved. The construction enumerates at
most 108 nonzero first-Superbox key classes, not plaintexts or all keys, and
uses exact GF(2) solves plus complete local truth tables. It needs NumPy but
no SAT package. The final linear coset uses the certified minimal
206-dimensional joint quotient in (P,K), not a fixed-key probability claim.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
from engine import core
from engine.connected import Transfer
from engine.local import column_record
from engine import model
from ..outer.outer_model import OuterModel
from .assemble_quotient import full_joint_mask


def readint(value):
    return int(value, 0) if isinstance(value, str) else int(value)


def parity_syndrome(basis, packed):
    return sum(core.parity(mask & packed) << i for i, mask in enumerate(basis))


class WitnessBuilder:
    """Reusable complete construction for the certified scheme-A MT space."""

    def __init__(self):
        projection = json.loads((HERE / "output/projected_head_weight.json").read_text())
        self.mt_basis = [readint(m) for m in projection["retained_middle_tail_basis_hex"]]
        assert len(self.mt_basis) == 68
        self.mt_space = core.Space(self.mt_basis)
        quotient = json.loads((HERE / "output/joint_common_quotient.json").read_text())
        self.joint_basis = [readint(m) for m in quotient["joint_basis_internal_hex"]]
        self.key_basis = [readint(m) for m in quotient["key_only_basis_on_RK1_to_RK5_hex"]]
        self.rk1_forms = core.independent(m & core.MASK64 for m in self.mt_basis)
        self.rk2_forms = core.independent((m >> 64) & core.MASK64 for m in self.mt_basis)
        assert (len(self.rk1_forms), len(self.rk2_forms)) == (9, 3)
        head_projection = core.independent(m & ((1 << 128)-1) for m in self.mt_basis)
        assert len(head_projection) == 12
        self.tr = Transfer()
        assert core.Space(self.rk2_forms + self.tr.beta_basis).rank == 24
        self.local = []
        for c in range(4):
            record = column_record(core.extract(core.U,c), core.extract(core.T,c))
            masks = [core.embed(readint(m), c) for m in record["basis"]]
            good = sorted(s for row in record["histogram"] if row["probability"]["numerator"]
                          for s in row["syndromes"])
            self.local.append((masks, good))
        assert [len(good) for _,good in self.local] == [3,3,3,4]
        first_box_key_forms = [mask for masks,_ in self.local for mask in masks]
        assert len(first_box_key_forms) == 13
        assert core.Space(first_box_key_forms + self.rk1_forms).rank == 22
        self.outer = OuterModel()
        middle = json.loads((HERE.parent / "middle/output/complete_middle_fourier.json").read_text())
        self.middle_denominator = middle["denominator"]
        self.middle_terms = [(readint(row["mask"]), row["coefficient"]) for row in middle["terms"]]

    def validate_space(self, basis):
        basis = [readint(m) for m in basis]
        if len(basis) != 68 or core.Space(basis).rank != 68:
            raise ValueError("MT basis must contain exactly 68 independent physical five-RK masks")
        if core.Space(self.mt_basis + basis).rank != 68:
            raise ValueError("Supplied basis is not the complete certified middle-tail dependency space")
        if any(m >> 320 for m in basis):
            raise ValueError("MT masks use RK1 low64 through RK5 high64 only")
        return basis

    def probabilities(self, keys):
        packed = sum(k << (64*i) for i,k in enumerate(keys))
        middle = Fraction(sum((-n if core.parity(mask & packed) else n)
                              for mask,n in self.middle_terms), self.middle_denominator)
        head = self.outer.physical_head(keys[0],keys[1])
        tail = self.outer.physical_tail(keys[3],keys[4])
        return head, middle, tail

    def construct(self, basis=None, rhs=None, representative_keys=None,
                  verification_members=16, seed=202609150468):
        started = time.perf_counter()
        basis = self.validate_space(self.mt_basis if basis is None else basis)
        if representative_keys is not None:
            representative_keys = [readint(k) for k in representative_keys]
            if len(representative_keys) != 5 or any(k < 0 or k > core.MASK64 for k in representative_keys):
                raise ValueError("Representative must be five 64-bit physical round-key words")
            packed = sum(k << (64*i) for i,k in enumerate(representative_keys))
            expected_rhs = parity_syndrome(basis, packed)
            if rhs is not None and readint(rhs) != expected_rhs:
                raise ValueError("Representative keys do not satisfy supplied MT right-hand side")
            rhs = expected_rhs
        else:
            if rhs is None:
                raise ValueError("Supply either MT syndrome rhs or five representative round-key words")
            rhs = readint(rhs)
            if rhs < 0 or rhs >= 1 << 68:
                raise ValueError("MT syndrome must be an unsigned 68-bit integer")
            packed = core.solve([(m,(rhs >> i)&1) for i,m in enumerate(basis)])
            representative_keys = [(packed >> (64*i)) & core.MASK64 for i in range(5)]
        target_keys = list(representative_keys)
        target_mt = [core.parity(m & packed) for m in basis]
        fixed1 = [(m,core.parity(m & target_keys[0])) for m in self.rk1_forms]
        tried = 0
        selected = None
        for local_codes in product(*(good for _,good in self.local)):
            tried += 1
            equations = list(fixed1)
            for (masks,_), code in zip(self.local,local_codes):
                equations.extend((mask, ((code >> i)&1) ^ core.parity(mask & core.RC[0]))
                                 for i,mask in enumerate(masks))
            try:
                rk1 = core.solve(equations)
                selected = list(local_codes)
                break
            except ValueError:
                pass
        if selected is None:
            raise AssertionError("No compatible positive first-box key class: contradicts the certified conditional mean")
        x = 0
        for c in range(4):
            truth = core.local_indicator(core.extract(core.U,c), core.extract(core.T,c),
                                         core.extract(rk1 ^ core.RC[0],c))
            successes = np.flatnonzero(truth)
            assert len(successes)
            x |= core.embed(int(successes[0]),c)
        y = core.box(x, rk1 ^ core.RC[0])
        assert core.head_two(x,rk1 ^ core.RC[0])
        base_z = core.connector(y,core.RC[1])
        equations2 = [(m,core.parity(m & target_keys[1])) for m in self.rk2_forms]
        for c,quotient in enumerate(self.tr.g_quotients):
            successful = [row for row in quotient["histogram"] if row["probability"]["numerator"]]
            target = successful[0]["syndromes"][0]
            assert successful[0]["probability"]["numerator"] == successful[0]["probability"]["denominator"]
            for i,local_mask in enumerate(quotient["basis"]):
                a = core.embed(readint(local_mask),c)
                equations2.append((core.iperm(a), ((target >> i)&1) ^ core.parity(a & base_z)))
        rk2 = core.solve(equations2)
        keys = [rk1,rk2,target_keys[2],target_keys[3],target_keys[4]]
        result_packed = sum(k << (64*i) for i,k in enumerate(keys))
        assert [core.parity(m & result_packed) for m in basis] == target_mt
        assert bool(model.evaluate(x,keys,block="head"))
        head, middle, tail = self.probabilities(keys)
        original_h, original_m, original_t = self.probabilities(target_keys)
        assert head > 0 and middle == original_m and tail == original_t
        z = x | (result_packed << 64)
        syndrome = [core.parity(m & z) for m in self.joint_basis]
        rng = random.Random(seed)
        for _ in range(verification_members):
            raw = rng.getrandbits(384)
            correction = core.solve([(m,core.parity(m & raw)) for m in self.joint_basis])
            member = z ^ raw ^ correction
            assert all(core.parity(m & member) == b for m,b in zip(self.joint_basis,syndrome))
            xx = member & core.MASK64
            kk = [(member >> (64*(i+1))) & core.MASK64 for i in range(5)]
            assert bool(model.evaluate(xx,kk,block="head"))
            assert all(core.parity(m & (member >> 64)) == core.parity(m & result_packed)
                       for m in self.key_basis)
        return {
            "state": "complete_verified",
            "scope": "A constructed nonzero-head 206-dimensional joint coset inside the specified complete MT key coset; two outer value connectors zero",
            "method": "Choose a positive first-box key class and solve its 13 forms together with 9 MT forms (proved rank 22); exact local truth tables give X; a rank-24 RK2 solve makes g=1 while preserving its 3 MT forms. No SAT package.",
            "mt_input_basis_hex": [hex(m) for m in basis],
            "mt_input_rhs_lsb_first_hex": hex(rhs),
            "mt_input_representative_keys": [hex(k) for k in target_keys],
            "all_68_MT_syndromes_preserved": True,
            "rk1_preserved_form_count": len(self.rk1_forms),
            "rk2_preserved_form_count": len(self.rk2_forms),
            "first_box_key_and_MT_RK1_combined_rank": 22,
            "g_interface_and_MT_RK2_combined_rank": 24,
            "positive_first_box_key_combinations_tested": tried,
            "selected_first_box_positive_key_class_syndromes": selected,
            "X": hex(x), "paired_X": hex(x ^ core.U),
            "physical_round_keys": [hex(k) for k in keys],
            "head_mean_at_representative": core.fraction_record(head),
            "middle_probability": core.fraction_record(middle),
            "tail_probability": core.fraction_record(tail),
            "joint_coset_constant_model_probability": core.fraction_record(middle * tail),
            "joint_rank": len(self.joint_basis),
            "joint_rank_is_minimal_claimed": True,
            "joint_rank_certificate": str(HERE.parents[1] / "data/joint_rank_certificate.json"),
            "joint_full_physical_P_master_key_coset_size": "2^306",
            "joint_coset_basis_physical_hex": [hex(full_joint_mask(m)) for m in self.joint_basis],
            "joint_coset_rhs_bits": syndrome,
            "joint_coset_rhs_lsb_first_hex": hex(sum(b << i for i,b in enumerate(syndrome))),
            "physical_packing": "P|W1|W2|RK1|RK2|RK3|RK4|RK5; each 64-bit word, first least significant. Every equation is parity(mask & packed)=rhs.",
            "input_whitening_condition": "X=P xor W1; every input mask is duplicated on P and W1, W2 mask is zero",
            "direct_head_coset_member_checks": verification_members,
            "checks_seed": seed,
            "global_maximum_claimed": False,
            "global_extremum_status": "This constructor supplies feasibility and conditions only. A separate complete MT histogram or extremum certificate must establish global optimality.",
            "elapsed_seconds": time.perf_counter()-started,
        }


def self_test(builder, output):
    sat = model.CASES["sat_witness"]["keys"]
    records = [builder.construct(representative_keys=sat)]
    middle = json.loads((HERE.parent / "middle/output/summary.json").read_text())
    mb = [readint(m) for m in middle["physical_round_key_basis"]]
    audit = json.loads((HERE.parent / "output/factor_space_audit.json").read_text())
    tb = [readint(m) for m in audit["tail_basis_hex"]]
    packed_sat = sum(k << (64*i) for i,k in enumerate(sat))
    tail_eq = [(m,core.parity(m & packed_sat)) for m in tb]
    # Pick the highest M class compatible with the entire known-positive T
    # class. This is a complete 2^18 M scan, not a global joint maximum claim.
    dist = np.load(HERE.parent / "middle/output/middle_18d_distribution_numerators.npy")
    found = None
    for syndrome in np.argsort(dist)[::-1]:
        try:
            representative = core.solve(tail_eq + [(m,(int(syndrome) >> i)&1) for i,m in enumerate(mb)])
            found = representative, int(syndrome)
            break
        except ValueError:
            pass
    assert found is not None
    keys = [(found[0] >> (64*i)) & core.MASK64 for i in range(5)]
    result = builder.construct(representative_keys=keys)
    result["test_case"] = "Maximum middle class compatible with the full known-positive K_sat tail quotient; not unrestricted maximum"
    result["selected_middle_syndrome"] = found[1]
    assert result["middle_probability"]["numerator"] and result["tail_probability"]["numerator"]
    records.append(result)
    rng = random.Random(202609150469)
    for _ in range(16):
        records.append(builder.construct(rhs=rng.getrandbits(68), verification_members=4))
    output.mkdir(parents=True,exist_ok=True)
    for i,record in enumerate(records):
        (output / f"case_{i:02d}.json").write_text(json.dumps(record,indent=2)+"\n")
    report = {"state":"complete_verified", "cases":len(records),
              "arbitrary_MT_syndrome_cases":16,
              "all_cases_preserve_68_syndromes_and_have_H4_1":True,
              "max_positive_first_box_key_classes_tried":max(r["positive_first_box_key_combinations_tested"] for r in records),
              "nonzero_explicit_cases":[{"case":i,"probability":r["joint_coset_constant_model_probability"]} for i,r in enumerate(records[:2])],
              "no_global_extrema_claimed":True}
    (output / "self_test.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,help="JSON: basis(optional), rhs(hex or integer), representative_keys(optional)")
    parser.add_argument("--rhs",help="68-bit syndrome under default complete MT basis, little-endian")
    parser.add_argument("--keys",nargs=5,help="Five physical RK words specifying an MT class; they are not all fixed by the construction")
    parser.add_argument("--output",type=Path,default=HERE / "witness_output")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--background",action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.background:
        command = [sys.executable,"-B",str(Path(__file__).resolve())] + [arg for arg in sys.argv[1:] if arg != "--background"]
        options = {"start_new_session":True} if os.name != "nt" else {"creationflags":subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        with (args.output / "run.log").open("ab",buffering=0) as log:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,**options)
        launch = {"state":"background_running","pid":child.pid,"command":command}
        (args.output / "launch.json").write_text(json.dumps(launch,indent=2)+"\n")
        print(json.dumps(launch),flush=True)
        return
    (args.output / "status.json").write_text(json.dumps({"state":"running","pid":os.getpid()})+"\n")
    try:
        builder = WitnessBuilder()
        if args.self_test:
            self_test(builder,args.output)
        else:
            request = json.loads(args.request.read_text()) if args.request else {}
            if args.keys:
                request["representative_keys"] = args.keys
            if args.rhs is not None:
                request["rhs"] = args.rhs
            if not request:
                request["representative_keys"] = model.CASES["sat_witness"]["keys"]
            result = builder.construct(**request)
            (args.output / "witness.json").write_text(json.dumps(result,indent=2)+"\n")
            print(json.dumps({k:result[k] for k in ("state","X","physical_round_keys","joint_coset_constant_model_probability","elapsed_seconds")}),flush=True)
        (args.output / "status.json").write_text(json.dumps({"state":"complete_verified","pid":os.getpid()})+"\n")
    except BaseException as exc:
        (args.output / "status.json").write_text(json.dumps({"state":"failed","pid":os.getpid(),"error":str(exc)})+"\n")
        raise


if __name__ == "__main__":
    main()
