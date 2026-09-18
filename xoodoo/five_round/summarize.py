"""Aggregate two independent real full-five-round six-constraint experiments.

Read-only with respect to experiment inputs. Writes a new aggregate JSON and
Chinese Markdown report, with raw counts and all 64 conditional classes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import experiment

HERE = Path(__file__).resolve().parent
Z95 = NormalDist().inv_cdf(.975)
Z64 = NormalDist().inv_cdf(1 - .05 / 128)
MATCH_FIELDS = (
    "source_sha256", "source_theory_ddt_prefix_rounds", "constraints",
    "class_encoding", "constraint_count", "class_count", "sampling",
    "input_difference_lanes_y_major", "output_mask", "core_input_difference",
    "round_constants", "histogram_columns", "rng",
)


def add_rows(rows):
    rows = tuple(rows)
    return [sum(int(row[j]) for row in rows) for j in range(6)]


def signed(value):
    return experiment.signed_power(value)


def estimate(n, total):
    value = experiment.metric(n, total)
    if value is not None:
        lo, hi = value["simultaneous_64_confidence_95"]
        value["bonferroni_64_significant_nonzero"] = lo > 0 or hi < 0
    return value


def difference_test(value, variance, n):
    se = math.sqrt(max(0., variance) / n)
    z = value / se if se else (0. if value == 0 else None)
    return {
        "difference": value, "standard_error": se, "z": z,
        "two_sided_normal_p": math.erfc(abs(z) / math.sqrt(2)) if z is not None else 0.,
        "confidence_95": [value - Z95 * se, value + Z95 * se],
        "simultaneous_64_confidence_95": [value - Z64 * se, value + Z64 * se],
        "reject_zero_bonferroni_64": abs(value) > Z64 * se,
    }


def row_summary(row, model):
    n, s5, s4, ne, se, cross = map(int, row)
    if not n:
        raise ValueError("An aggregate class contains no samples")
    c5, c4, prod = s5 / n, s4 / n, cross / n
    five, control = estimate(n, s5), estimate(n, s4)
    diff_scaled, diff_equal = c5 - c4 / 16, c5 - c4
    return {
        "raw_histogram": list(map(int, row)),
        "core_4round_joint_U_1DDT_H64_model": model,
        "core_4round_model_signed_power": signed(model),
        "naive_core_model_div16_hypothesis_NOT_5round_theory": model / 16,
        "naive_hypothesis_signed_power": signed(model / 16),
        "five_round_experiment": five,
        "paired_four_round_core_experiment": control,
        "prefix_event_probability_estimate": ne / n,
        "prefix_event_probability_theory": 1 / 16,
        "on_prefix_event": estimate(ne, se),
        "off_prefix_event": estimate(n - ne, s5 - se),
        "event_contribution_to_five_round_correlation": se / n,
        "off_event_contribution_to_five_round_correlation": (s5 - se) / n,
        "paired_test_C5_equals_C4div16": difference_test(
            diff_scaled, 1 + 1 / 256 - prod / 8 - diff_scaled**2, n),
        "paired_test_C5_equals_C4": difference_test(
            diff_equal, 2 - 2 * prod - diff_equal**2, n),
        "C4_experiment_minus_model": difference_test(c4 - model, 1 - c4*c4, n),
        "C5_experiment_minus_naive_model_div16": difference_test(
            c5 - model / 16, 1 - c5*c5, n),
    }


def independent_seed_comparison(row_a, row_b, sign_column=1):
    na, nb = int(row_a[0]), int(row_b[0])
    ca, cb = row_a[sign_column] / na, row_b[sign_column] / nb
    variance = (1 - ca*ca) / na + (1 - cb*cb) / nb
    se = math.sqrt(max(0., variance))
    diff = ca - cb
    return {"run_a_correlation": ca, "run_b_correlation": cb,
            "difference": diff, "standard_error_difference": se,
            "z_difference": diff / se if se else None,
            "different_at_bonferroni_64": abs(diff) > Z64 * se}


def load_run(path):
    path = path.resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("complete") is not True or data["samples"] != data["target_samples"]:
        raise ValueError(f"Run not complete: {path}")
    hist = data["raw_histogram"]
    if len(hist) != 64 or any(len(row) != 6 for row in hist):
        raise ValueError(f"Unexpected histogram shape: {path}")
    if sum(row[0] for row in hist) != data["samples"]:
        raise ValueError(f"Count mismatch: {path}")
    for n, s5, s4, ne, se, cross in hist:
        if not (0 <= ne <= n and abs(s5) <= n and abs(s4) <= n
                and abs(se) <= ne and abs(s5-se) <= n-ne and abs(cross) <= n):
            raise ValueError(f"Invalid signed counts: {path}")
        if any((n - value) % 2 for value in (s5, s4, cross)) or (ne - se) % 2:
            raise ValueError(f"Invalid sign-sum parity: {path}")
    return data, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                  "seed": data["seed"], "samples": data["samples"],
                  "validation": data["validation"]}


def summarize(path_a, path_b):
    a, provenance_a = load_run(path_a)
    b, provenance_b = load_run(path_b)
    if a["seed"] == b["seed"]:
        raise ValueError("Two independent runs must use different seeds")
    for key in MATCH_FIELDS:
        if a.get(key) != b.get(key):
            raise ValueError(f"Run metadata mismatch: {key}")
    if a["source_sha256"] != experiment.SOURCE_HASH:
        raise ValueError("Unexpected source constraint/model hash")
    expected_delta = [f"0x{v:08x}" for v in experiment.DELTA]
    expected_constants = [f"0x{c:x}" for c in experiment.xo.reduced_round_constants(5)]
    if a["input_difference_lanes_y_major"] != expected_delta or a["round_constants"] != expected_constants:
        raise ValueError("Unexpected five-round boundary or constants")
    if a["output_mask"] != "full-round output bit e0 (col0,mask1)" or a["constraint_count"] != 6:
        raise ValueError("Unexpected output mask or constraints")
    source, bases, _ = experiment.source_data()
    from derive_constraints import rank
    event_bits = (129, 328, 88, 217)
    assert rank(bases) == 6 and rank(bases + [1 << j for j in event_bits]) == 10
    theory = source["capacities"][-1]["correlation_distribution"]
    hist = [add_rows((ra, rb)) for ra, rb in zip(a["raw_histogram"], b["raw_histogram"])]
    classes = []
    for i, row in enumerate(hist):
        result = row_summary(row, theory[i])
        result.update(index=i, assignment_a0_to_a5=[(i >> j) & 1 for j in range(6)],
                      display_a0_to_a5="".join(str((i >> j) & 1) for j in range(6)),
                      independent_seed_five_round=independent_seed_comparison(
                          a["raw_histogram"][i], b["raw_histogram"][i]))
        classes.append(result)
    groups = []
    for name, indices in (
        ("a0=1", [i for i in range(64) if i & 1]),
        ("a0=0,a1=0", [i for i in range(64) if i & 3 == 0]),
        ("a0=0,a1=1", [i for i in range(64) if i & 3 == 2]),
    ):
        group = row_summary(add_rows(hist[i] for i in indices),
                            sum(theory[i] for i in indices) / len(indices))
        group.update(name=name, class_indices=indices, class_count=len(indices))
        group["independent_seed_five_round"] = independent_seed_comparison(
            add_rows(a["raw_histogram"][i] for i in indices),
            add_rows(b["raw_histogram"][i] for i in indices))
        groups.append(group)
    overall = row_summary(add_rows(hist), sum(theory) / 64)
    overall["independent_seed_five_round"] = independent_seed_comparison(
        add_rows(a["raw_histogram"]), add_rows(b["raw_histogram"]))
    significant = [r for r in classes if r["five_round_experiment"]["bonferroni_64_significant_nonzero"]]
    extreme = {}
    for name, operation in (("strongest", max), ("weakest", min)):
        if significant:
            row = operation(significant, key=lambda r: abs(r["five_round_experiment"]["correlation"]))
            extreme[name] = {"index": row["index"], "display_a0_to_a5": row["display_a0_to_a5"],
                             **row["five_round_experiment"]}
    return {
        "version": 1, "complete": True, "provenance": [provenance_a, provenance_b],
        "source_sha256": experiment.SOURCE_HASH, "boundary_metadata": {k: a[k] for k in MATCH_FIELDS},
        "samples": overall["five_round_experiment"]["samples"], "constraint_count": 6,
        "theory_scope": "Only the four-round core column is joint-U complete 1DDT H64 theory; C4/16 is a hypothesis, not five-round theory.",
        "sampling_scope": "Uniform independent X; condition only on F(X)=a, never filter on event E.",
        "exact_event": {"definition": "R(X xor Delta5) xor R(X)=Delta_core; Y=R(X)",
                        "Y_flat_bit_conditions": [[129, 0], [328, 1], [88, 0], [217, 1]],
                        "rank_B": 6, "rank_B_plus_event": 10, "probability_in_every_class": 1 / 16,
                        "decomposition": "C5(a)=C4,E(a)/16+15*C5,notE(a)/16; event conditioning can change the core correlation."},
        "confidence_method": "Normal sign-mean intervals; per-class95 and Bonferroni64 familywise95, separately for each comparison family.",
        "paired_variance_formulas": {"C5_minus_C4div16": "1+1/256-E[S5*S4]/8-(C5-C4/16)^2",
                                     "C5_minus_C4": "2-2*E[S5*S4]-(C5-C4)^2"},
        "overall": overall, "groups": groups, "classes": classes,
        "raw_histogram": hist, "histogram_columns": a["histogram_columns"],
        "bonferroni_64_nonzero_class_count": len(significant),
        "bonferroni_64_nonzero_class_indices": [r["index"] for r in significant],
        "nonzero_point_estimate_extrema": extreme,
        "extrema_warning": "Selected point estimates among Bonferroni-significant classes; not proof of exact strongest/weakest ordering or exact zero elsewhere.",
        "independent_seed_different_class_indices": [r["index"] for r in classes if r["independent_seed_five_round"]["different_at_bonferroni_64"]],
        "paired_reject_C5_equals_C4div16_class_indices": [r["index"] for r in classes if r["paired_test_C5_equals_C4div16"]["reject_zero_bonferroni_64"]],
        "paired_reject_C5_equals_C4_class_indices": [r["index"] for r in classes if r["paired_test_C5_equals_C4"]["reject_zero_bonferroni_64"]],
    }


def fmt(value):
    return f"{value:+.10f} ({signed(value)})" if value else "0"


def interval(metric):
    return "[" + ", ".join(f"{v:+.8f}" for v in metric["confidence_95"]) + "]"


def render_markdown(data):
    overall = data["overall"]
    lines = ["# 完整五轮 Xoodoo：六个前推输入约束的真实实验", "",
             "只按六个 F_i(X)=B_i(R(X)) 分类，不额外要求首轮差分事件 E 成立。"
             "四轮模型来自联合全局 U、完整 1DDT、容量 H64；本次五轮实验直接计算真实置换，没有 DDT 近似。", "",
             "**关键区别：六个输入约束已精确前推，但四轮条件相关度不能随约束直接搬到五轮。表中 C4/16 仅为待检验假设，不是五轮理论。**", "",
             f"两次独立随机种子：{data['provenance'][0]['seed']}、{data['provenance'][1]['seed']}；"
             f"总样本 {data['samples']} = 2^{math.log2(data['samples']):g}；64 类。",
             "输入差分（每行一个平面，x=0..3）：", "", "```text"]
    delta = data["boundary_metadata"]["input_difference_lanes_y_major"]
    lines.extend(" ".join(delta[i:i+4]) for i in (0, 4, 8))
    lines += ["```", "", "输出掩码 e0；五轮常数 0x2c、0x380、0xf0、0x1a0、0x12。", "",
              "## 总体与分组", "",
              "分组理论值按等概率类取平均；实验按实际样本计数加权。所有点估计同时给出十进制值及带符号的 2 的幂次。", "",
              "| 组 | 类数 | 样本 | 四轮 core 模型 | 配对四轮实验 | C4模型/16 假设 | 真实五轮实验 | 五轮95%区间 |",
              "|---|---:|---:|---|---|---|---|---|"]
    for name, count, row in [("全部", 64, overall)] + [(g["name"], g["class_count"], g) for g in data["groups"]]:
        lines.append(f"| {name} | {count} | {row['five_round_experiment']['samples']} | "
                     f"{fmt(row['core_4round_joint_U_1DDT_H64_model'])} | {fmt(row['paired_four_round_core_experiment']['correlation'])} | "
                     f"{fmt(row['naive_core_model_div16_hypothesis_NOT_5round_theory'])} | {fmt(row['five_round_experiment']['correlation'])} | {interval(row['five_round_experiment'])} |")
    lines += ["", "## 首轮事件诊断", "",
              "记 Y=R(X)。E 等价于 Y[0,1,1]=0、Y[2,2,8]=1、Y[2,0,24]=0、Y[2,1,25]=1。"
              "六个 B 的秩为6，与这四个条件合并秩为10，因此每一类严格有 P(E|a)=1/16。", "",
              "精确分解为 C5(a)=C4,E(a)/16+15 C5,非E(a)/16。事件内真实五轮符号等于固定差分四轮 core 的符号，"
              "但 C4,E(a) 不一定等于未加 E 的 C4(a)，事件外贡献也不一定为零。下面贡献列采用实际样本比例，二者相加精确等于五轮实验值。", "",
              "| 组 | P(E)实验 | E内相关度 | E外相关度 | E贡献 | 非E贡献 |",
              "|---|---:|---|---|---|---|"]
    for name, row in [("全部", overall)] + [(g["name"], g) for g in data["groups"]]:
        lines.append(f"| {name} | {row['prefix_event_probability_estimate']:.10f} | {fmt(row['on_prefix_event']['correlation'])} | "
                     f"{fmt(row['off_prefix_event']['correlation'])} | {fmt(row['event_contribution_to_five_round_correlation'])} | "
                     f"{fmt(row['off_event_contribution_to_five_round_correlation'])} |")
    lines += ["", "## 全部64类", "",
              "六位串从左到右固定为 a0,a1,a2,a3,a4,a5；索引为 Σ a_i 2^i，不是六位串按通常二进制读取的值。"
              "“是”表示五轮相关度的64类 Bonferroni 同时95%区间不含0；“否”不等于证明相关度为0。单类95%区间用十进制列出。", "",
              "| 索引 | a0…a5 | 四轮 core 模型 | 配对四轮实验 | C4模型/16 假设 | 真实五轮实验 | 五轮95%区间 | 同时显著非零 |",
              "|---:|---|---|---|---|---|---|---|"]
    for row in data["classes"]:
        f = row["five_round_experiment"]
        lines.append(f"| {row['index']} | `{row['display_a0_to_a5']}` | {fmt(row['core_4round_joint_U_1DDT_H64_model'])} | "
                     f"{fmt(row['paired_four_round_core_experiment']['correlation'])} | {fmt(row['naive_core_model_div16_hypothesis_NOT_5round_theory'])} | "
                     f"{fmt(f['correlation'])} | {interval(f)} | {'是' if f['bonferroni_64_significant_nonzero'] else '否'} |")
    lines += ["", "## 统计比较与非零范围", "",
              f"64类同时95%标准下，显著非零类数为 {data['bonferroni_64_nonzero_class_count']}。"
              "以下最强/最弱只在这些类中按实验点估计选取；有限样本不能证明相近类别的精确排序，更不能把其他类证明为零。"]
    max_naive_z = max(abs(r["C5_experiment_minus_naive_model_div16"]["z"]) for r in data["classes"])
    max_paired_z = max(abs(r["paired_test_C5_equals_C4div16"]["z"]) for r in data["classes"])
    max_core_z = max(abs(r["C4_experiment_minus_model"]["z"]) for r in data["classes"])
    max_seed_z = max(abs(r["independent_seed_five_round"]["z_difference"]) for r in data["classes"])
    lines += ["", f"64类双侧 Bonferroni 95%临界值为 |z|>{Z64:.6f}。"
              f"逐类最大 |z|：五轮对 C4模型/16 为 {max_naive_z:.6f}，"
              f"配对五轮对 C4实验/16 为 {max_paired_z:.6f}，"
              f"四轮实验对四轮模型为 {max_core_z:.6f}，两个独立种子之间为 {max_seed_z:.6f}。"]
    if max_naive_z < Z64 and max_paired_z < Z64:
        lines += ["", "在当前精度和上述同时检验标准下，全部64类与 C4/16 近似关系兼容，实验支持这个近似；"
                  "这不是严格等式证明，也不能仅凭约束前推推出该关系。"
                  "特别是弱相关度16类中关于 a2…a5 的细微变化小于或接近当前单类误差，"
                  "不能声称已逐项确认全部细小理论差异。"]
    for name, label in (("strongest", "最强非零点估计"), ("weakest", "最弱非零点估计")):
        if name in data["nonzero_point_estimate_extrema"]:
            row = data["nonzero_point_estimate_extrema"][name]
            lines += ["", f"{label}：类 {row['index']}，a0…a5=`{row['display_a0_to_a5']}`，"
                      f"C5={fmt(row['correlation'])}，95%区间 {interval(row)}。"]
    lines += ["", "| 比较对象 | 全部样本 z | a0=1 z | a0=0,a1=0 z | a0=0,a1=1 z |",
              "|---|---:|---:|---:|---:|"]
    for key, name in (("paired_test_C5_equals_C4div16", "配对检验 C5=C4实验/16"),
                      ("paired_test_C5_equals_C4", "配对检验 C5=C4实验")):
        vals = [overall[key]["z"]] + [g[key]["z"] for g in data["groups"]]
        lines.append("| " + name + " | " + " | ".join(f"{v:+.4f}" if v is not None else "零方差" for v in vals) + " |")
    lines += ["", "上述配对检验使用同一样本两种符号的乘积估计协方差，不能把两列当成独立实验。"
              "C5−C4/16 的单样本方差为 1+1/256−E[S5S4]/8−(C5−C4/16)²；"
              "C5−C4 的方差为 2−2E[S5S4]−(C5−C4)²。", "",
              f"64类 Bonferroni 标准拒绝 C5=C4实验/16 的类：{data['paired_reject_C5_equals_C4div16_class_indices']}。", "",
              f"64类 Bonferroni 标准拒绝 C5=C4实验 的类：{data['paired_reject_C5_equals_C4_class_indices']}。", "",
              "独立种子比较采用 z=(C_A−C_B)/sqrt(SE_A²+SE_B²)。"
              f"总体 z={overall['independent_seed_five_round']['z_difference']:+.4f}；"
              f"64类 Bonferroni 标准下两次结果有差异的类：{data['independent_seed_different_class_indices']}。", "",
              "JSON保留每类原始计数、两种相关度、事件内外数据、单类及同时区间、配对检验和两个独立种子的差异统计；没有用缺省文字替代小的实验数值。", "",
              "## 可复现来源", ""]
    for item in data["provenance"]:
        lines += [f"- `{item['path']}`；SHA-256 `{item['sha256']}`；seed={item['seed']}；N={item['samples']}。"]
    lines += ["", f"原四轮六维模型 SHA-256：`{data['source_sha256']}`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path, nargs="?", default=HERE / "experiments" / "run_a_n30.json")
    parser.add_argument("run_b", type=Path, nargs="?", default=HERE / "experiments" / "run_b_n30.json")
    parser.add_argument("--output-json", type=Path, default=HERE / "experiments" / "combined_n31.json")
    parser.add_argument("--output-md", type=Path, default=HERE / "experiments" / "combined_n31.md")
    args = parser.parse_args()
    for output in (args.output_json, args.output_md):
        if output.exists():
            parser.error(f"Output exists; choose a new path: {output}")
        if output.resolve() in (args.run_a.resolve(), args.run_b.resolve()):
            parser.error("Output must not replace an experiment input")
    if args.output_json.resolve() == args.output_md.resolve():
        parser.error("JSON and Markdown output paths must differ")
    data = summarize(args.run_a, args.run_b)
    markdown = render_markdown(data)
    for output in (args.output_json, args.output_md):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"samples": data["samples"], "overall_C5": data["overall"]["five_round_experiment"],
                      "significant_classes": data["bonferroni_64_nonzero_class_count"],
                      "independent_seed_different_classes": data["independent_seed_different_class_indices"],
                      "output_json": str(args.output_json.resolve()), "output_md": str(args.output_md.resolve())}, indent=2))


if __name__ == "__main__":
    main()
