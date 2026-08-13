from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGGREGATE = REPO_ROOT / "experiments/results/stage16_diagnostic_r2.aggregate.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports/stage16-analysis"


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(values: list[float], *, seed: int = 20260812, draws: int = 20000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(statistics.mean(sample))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for cursor in range(index, end):
            ranks[indexed[cursor][0]] = rank
        index = end
    return ranks


def exact_sign_test(differences: list[float]) -> float:
    nonzero = [value for value in differences if value != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = sum(value > 0 for value in nonzero)
    tail = sum(math.comb(n, k) for k in range(0, min(positive, n - positive) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def exact_wilcoxon(differences: list[float]) -> tuple[float, float]:
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 0.0, 1.0
    absolute = [abs(value) for value in nonzero]
    ranks = rankdata(absolute)
    observed = sum(rank for value, rank in zip(nonzero, ranks) if value > 0)
    total = sum(ranks)
    distances = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        signed_sum = sum(rank * sign for rank, sign in zip(ranks, signs))
        distances.append(abs(signed_sum))
    observed_distance = abs(2.0 * observed - total)
    p_value = sum(distance >= observed_distance - 1e-12 for distance in distances) / len(distances)
    return observed, min(1.0, p_value)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for position, index in enumerate(order):
        value = min(1.0, (len(p_values) - position) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def load_rows(path: Path) -> list[dict[str, Any]]:
    aggregate = load_aggregate(path)
    return rows_from_aggregate(aggregate)


def load_aggregate(path: Path) -> dict[str, Any]:
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    if aggregate.get("accepted") is not True:
        raise ValueError("Stage16 aggregate is not accepted")
    if aggregate.get("matrix", {}).get("matched_triples") != 9:
        raise ValueError("Stage16 aggregate does not contain 9 matched triples")
    return aggregate


def rows_from_aggregate(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for triple in aggregate["triples"]:
        row: dict[str, Any] = {"capacity": triple["capacity"], "seed": triple["seed"]}
        for policy_name in ("pressure", "recompute", "bounded"):
            policy = triple[policy_name]
            prefix = policy_name
            row[f"{prefix}_actual_recompute_tokens"] = policy["actual_recompute_tokens"]
            row[f"{prefix}_preemption_count"] = policy["preemption_count"]
            row[f"{prefix}_ttft_p99_ms"] = policy["ttft_p99_ms"]
            row[f"{prefix}_itl_p99_ms"] = policy["itl_p99_ms"]
            fairness = policy["fairness"]
            row[f"{prefix}_post_token_progress_gap_p99_steps"] = fairness["post_token_progress_gap_steps"]["p99"]
            row[f"{prefix}_max_waiting_age_steps"] = fairness["oldest_waiting_age"]["max"]
        rows.append(row)
    return rows


def build_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("post_token_progress_gap_p99_steps", "recompute", "bounded", "lower", "fairness"),
        ("max_waiting_age_steps", "recompute", "bounded", "lower", "fairness"),
        ("itl_p99_ms", "recompute", "bounded", "lower", "latency"),
        ("actual_recompute_tokens", "pressure", "bounded", "lower", "resource"),
        ("preemption_count", "pressure", "bounded", "lower", "resource"),
        ("ttft_p99_ms", "pressure", "bounded", "lower", "latency_guardrail"),
    ]
    comparisons = []
    for metric, baseline, candidate, direction, family in specs:
        baseline_values = [float(row[f"{baseline}_{metric}"]) for row in rows]
        candidate_values = [float(row[f"{candidate}_{metric}"]) for row in rows]
        differences = [base - cand if direction == "lower" else cand - base for base, cand in zip(baseline_values, candidate_values)]
        reductions = [difference / base if base != 0 else float("nan") for difference, base in zip(differences, baseline_values)]
        wilcoxon_stat, wilcoxon_p = exact_wilcoxon(differences)
        mean_difference, sd_difference = mean_sd(differences)
        ci_low, ci_high = bootstrap_ci(differences, seed=20260812 + len(comparisons))
        comparisons.append(
            {
                "metric": metric,
                "family": family,
                "baseline": baseline,
                "candidate": candidate,
                "n": len(rows),
                "baseline_mean": statistics.mean(baseline_values),
                "baseline_sd": statistics.stdev(baseline_values),
                "candidate_mean": statistics.mean(candidate_values),
                "candidate_sd": statistics.stdev(candidate_values),
                "difference_mean": mean_difference,
                "difference_sd": sd_difference,
                "difference_ci95_low": ci_low,
                "difference_ci95_high": ci_high,
                "mean_reduction": statistics.mean(reductions),
                "improved_pairs": sum(value > 0 for value in differences),
                "sign_test_p": exact_sign_test(differences),
                "wilcoxon_w_plus": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
                "differences": differences,
            }
        )
    adjusted = holm_bonferroni([item["wilcoxon_p"] for item in comparisons])
    for item, value in zip(comparisons, adjusted):
        item["wilcoxon_holm_p"] = value
    return comparisons


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_pareto_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1100, 700
    left, top, plot_w, plot_h = 110, 60, 900, 530
    colors = {"pressure": "#0072B2", "recompute": "#D55E00", "bounded": "#009E73"}
    points = []
    for row in rows:
        for policy in colors:
            points.append((float(row[f"{policy}_actual_recompute_tokens"]), float(row[f"{policy}_itl_p99_ms"]), policy, row["capacity"], row["seed"]))
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    xmin, xmax = min(x_values) * 0.9, max(x_values) * 1.05
    ymin, ymax = 0.0, max(y_values) * 1.08
    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w
    def sy(value: float) -> float:
        return top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for tick in range(6):
        y = top + plot_h * tick / 5
        value = ymax - (ymax - ymin) * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="13" fill="#333">{value:.0f}</text>')
    for tick in range(6):
        x = left + plot_w * tick / 5
        value = xmin + (xmax - xmin) * tick / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+6}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-size="13" fill="#333">{value/1000:.0f}k</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    for x, y, policy, capacity, seed in points:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="7" fill="{colors[policy]}" stroke="white" stroke-width="1.5"><title>{policy}, KV={capacity}, seed={seed}: recompute={x:.0f}, ITL P99={y:.1f} ms</title></circle>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-28}" text-anchor="middle" font-size="16">Actual Recompute Tokens (lower is better)</text>')
    parts.append(f'<text x="25" y="{top+plot_h/2}" text-anchor="middle" font-size="16" transform="rotate(-90 25 {top+plot_h/2})">ITL P99 (ms, lower is better)</text>')
    for index, (label, color) in enumerate(colors.items()):
        x = left + plot_w - 230 + index * 78
        parts.append(f'<circle cx="{x}" cy="25" r="6" fill="{color}"/><text x="{x+10}" y="30" font-size="13">{label}</text>')
    parts.append('<text x="110" y="35" font-size="17" font-weight="bold">Stage16A matched Pareto view (n=9 triples)</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_reduction_svg(path: Path, comparisons: list[dict[str, Any]]) -> None:
    selected = [item for item in comparisons if item["metric"] != "ttft_p99_ms"]
    width, height = 1100, 620
    left, top, plot_w, plot_h = 290, 70, 700, 430
    colors = {"fairness": "#0072B2", "latency": "#D55E00", "resource": "#009E73"}
    labels = {"post_token_progress_gap_p99_steps": "Post-token progress-gap P99", "max_waiting_age_steps": "Max waiting age", "itl_p99_ms": "ITL P99", "actual_recompute_tokens": "Actual recompute tokens", "preemption_count": "Preemption count"}
    maximum = max(float(item["mean_reduction"]) for item in selected)
    row_h = plot_h / len(selected)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="80" y="35" font-size="17" font-weight="bold">Stage16A mean paired reduction (bounded vs designated baseline)</text>']
    for i, item in enumerate(selected):
        y = top + i * row_h + row_h * 0.22
        reduction = float(item["mean_reduction"])
        bar_w = max(0.0, reduction) / maximum * plot_w
        color = colors[item["family"]]
        parts.append(f'<text x="{left-15}" y="{y+18:.1f}" text-anchor="end" font-size="14">{labels[item["metric"]]}</text>')
        parts.append(f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="{row_h*0.55:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{left+bar_w+10:.1f}" y="{y+18:.1f}" font-size="14">{reduction*100:.1f}% ({int(item["improved_pairs"])}/{int(item["n"])})</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h+4}" x2="{left+plot_w}" y2="{top+plot_h+4}" stroke="#333"/>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-size="15">Mean reduction across {int(selected[0]["n"])} matched triples</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _png_font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def write_pareto_png(path: Path, rows: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    scale = 1.5
    width, height = 1100, 700
    left, top, plot_w, plot_h = 110, 60, 900, 530
    colors = {"pressure": "#0072B2", "recompute": "#D55E00", "bounded": "#009E73"}
    points = [
        (
            float(row[f"{policy}_actual_recompute_tokens"]),
            float(row[f"{policy}_itl_p99_ms"]),
            policy,
        )
        for row in rows
        for policy in colors
    ]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    xmin, xmax = min(x_values) * 0.9, max(x_values) * 1.05
    ymin, ymax = 0.0, max(y_values) * 1.08

    def px(value: float) -> int:
        return round(value * scale)

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - ymin) / (ymax - ymin) * plot_h

    image = Image.new("RGB", (px(width), px(height)), "white")
    draw = ImageDraw.Draw(image)
    small = _png_font(px(13))
    label = _png_font(px(16))
    title = _png_font(px(17), bold=True)
    for tick in range(6):
        y = top + plot_h * tick / 5
        value = ymax - (ymax - ymin) * tick / 5
        draw.line((px(left), px(y), px(left + plot_w), px(y)), fill="#dddddd", width=px(1))
        draw.text((px(left - 12), px(y)), f"{value:.0f}", fill="#333333", font=small, anchor="rm")
    for tick in range(6):
        x = left + plot_w * tick / 5
        value = xmin + (xmax - xmin) * tick / 5
        draw.line((px(x), px(top + plot_h), px(x), px(top + plot_h + 6)), fill="#333333", width=px(1))
        draw.text((px(x), px(top + plot_h + 10)), f"{value / 1000:.0f}k", fill="#333333", font=small, anchor="ma")
    draw.line((px(left), px(top + plot_h), px(left + plot_w), px(top + plot_h)), fill="#333333", width=px(1))
    draw.line((px(left), px(top), px(left), px(top + plot_h)), fill="#333333", width=px(1))
    for x, y, policy in points:
        radius = px(7)
        center_x, center_y = px(sx(x)), px(sy(y))
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=colors[policy],
            outline="white",
            width=px(1.5),
        )
    draw.text((px(left + plot_w / 2), px(height - 28)), "Actual Recompute Tokens (lower is better)", fill="#111111", font=label, anchor="mm")
    y_label = Image.new("RGBA", (px(210), px(42)), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((y_label.width // 2, y_label.height // 2), "ITL P99 (ms, lower is better)", fill="#111111", font=label, anchor="mm")
    y_label = y_label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(y_label, (px(12), px(top + plot_h / 2) - y_label.height // 2), y_label)
    for index, (policy, color) in enumerate(colors.items()):
        x = left + plot_w - 230 + index * 78
        radius = px(6)
        draw.ellipse((px(x) - radius, px(25) - radius, px(x) + radius, px(25) + radius), fill=color)
        draw.text((px(x + 10), px(25)), policy, fill="#111111", font=small, anchor="lm")
    draw.text((px(110), px(35)), "Stage16A matched Pareto view (n=9 triples)", fill="#111111", font=title, anchor="ls")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(450, 450), optimize=True)


def write_reduction_png(path: Path, comparisons: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    selected = [item for item in comparisons if item["metric"] != "ttft_p99_ms"]
    scale = 1.5
    width, height = 1100, 620
    left, top, plot_w, plot_h = 290, 70, 700, 430
    colors = {"fairness": "#0072B2", "latency": "#D55E00", "resource": "#009E73"}
    labels = {
        "post_token_progress_gap_p99_steps": "Post-token progress-gap P99",
        "max_waiting_age_steps": "Max waiting age",
        "itl_p99_ms": "ITL P99",
        "actual_recompute_tokens": "Actual recompute tokens",
        "preemption_count": "Preemption count",
    }
    maximum = max(float(item["mean_reduction"]) for item in selected)
    row_h = plot_h / len(selected)

    def px(value: float) -> int:
        return round(value * scale)

    image = Image.new("RGB", (px(width), px(height)), "white")
    draw = ImageDraw.Draw(image)
    body = _png_font(px(14))
    axis = _png_font(px(15))
    title = _png_font(px(17), bold=True)
    draw.text((px(80), px(35)), "Stage16A mean paired reduction (bounded vs designated baseline)", fill="#111111", font=title, anchor="ls")
    for index, item in enumerate(selected):
        y = top + index * row_h + row_h * 0.22
        reduction = float(item["mean_reduction"])
        bar_w = max(0.0, reduction) / maximum * plot_w
        draw.text((px(left - 15), px(y + 18)), labels[item["metric"]], fill="#222222", font=body, anchor="rm")
        draw.rectangle((px(left), px(y), px(left + bar_w), px(y + row_h * 0.55)), fill=colors[item["family"]])
        draw.text((px(left + bar_w + 10), px(y + 18)), f"{reduction * 100:.1f}% ({int(item['improved_pairs'])}/{int(item['n'])})", fill="#222222", font=body, anchor="lm")
    draw.line((px(left), px(top + plot_h + 4), px(left + plot_w), px(top + plot_h + 4)), fill="#333333", width=px(1))
    draw.text((px(left + plot_w / 2), px(height - 25)), f"Mean reduction across {int(selected[0]['n'])} matched triples", fill="#111111", font=axis, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(450, 450), optimize=True)


def write_analysis_figures(
    output: Path,
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    write_pareto_svg(output / "stage16_pareto.svg", rows)
    write_pareto_png(output / "stage16_pareto.png", rows)
    write_reduction_svg(output / "stage16_reductions.svg", comparisons)
    write_reduction_png(output / "stage16_reductions.png", comparisons)


def markdown_report(
    aggregate: dict[str, Any],
    aggregate_path: Path,
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> str:
    del comparisons
    matrix = aggregate["matrix"]
    gates = aggregate["aggregate_gates"]
    runs = int(matrix["runs"])
    matched = int(matrix["matched_triples"])

    def gate(metric: str) -> dict[str, Any]:
        value = gates.get(metric)
        if not isinstance(value, dict):
            raise ValueError(f"Stage16 aggregate is missing gate: {metric}")
        return value

    def observation(metric: str, label: str) -> str:
        item = gate(metric)
        reduction = float(item["reduction"])
        direction = "降低" if reduction >= 0 else "增加"
        return (
            f"{label} aggregate 由 {float(item['baseline_total']):,.2f} 变为 "
            f"{float(item['candidate_total']):,.2f}，{direction} {abs(reduction) * 100:.2f}%，"
            f"{int(item['improved_pairs'])}/{matched} matched pairs 改善，gate "
            f"{'通过' if item['passed'] else '未通过'}。"
        )

    bounded_waiting_max = max(
        float(row["bounded_max_waiting_age_steps"]) for row in rows
    )
    accepted = aggregate.get("accepted") is True
    decision = str(aggregate.get("decision", "missing_decision"))
    data_source = aggregate_path.as_posix()
    lines = [
        "# Stage16A 严格结果分析",
        "",
        "## 分析问题",
        "",
        f"在冻结的 {runs}-cell matched matrix 中，`recompute_aware_bounded` 是否在保留资源收益的同时修复 Stage15 `recompute_aware` 的 drain fairness / tail-latency collapse？",
        "",
        "## 数据与比较单位",
        "",
        f"- 数据源：`{data_source}`。",
        f"- 比较单位：{matched} 个 `capacity × seed` matched triples；每个 triple 同时包含 pressure、unbounded recompute、bounded 三种 policy。",
        f"- aggregate 声明 completion gates {'已' if aggregate.get('completion_gates_validated') else '未'}验证、raw metrics {'已' if aggregate.get('raw_metrics_recomputed') else '未'}重算；矩阵为 {runs} cells。具体终态、请求数与 KV 释放结论以 strict aggregate 的上游 manifest/raw 校验为准。",
        f"- 统计结果用于描述 {matched} 个冻结配置点，不外推到生产流量总体。",
        "",
        "## 主要观察",
        "",
        f"1. {observation('post_token_progress_gap_p99', 'Bounded Drain 相比 unbounded recompute 的 post-token progress-gap P99')}",
        f"2. {observation('max_waiting_age', '最大 waiting age')} bounded 的 cell-level max waiting age 最大为 {bounded_waiting_max:,.0f} steps。",
        f"3. {observation('itl_p99_ms', 'ITL P99')}",
        f"4. {observation('actual_recompute_tokens', '相比 pressure baseline，Actual Recompute')} {observation('preemption_count', 'Preemption')}",
        f"5. {observation('ttft_p99_ms', 'TTFT P99 相比 pressure baseline')}",
        "",
        "## 机制解释与边界",
        "",
        "结果支持如下机制链：硬性 drain budget、waiting-age bound 与 episode-entry SLO watch 将原本以 resident exhaustion 为隐式终点的长 drain episode 切分为短 episode，使 waiting request 和已产生 token 的 resident request 获得有限 progress window；同时仍沿用 Stage15 的 recompute-aware victim selection，因此资源收益没有退回 pressure baseline。该解释由 episode length/exit reason、per-request progress gap、KV/终态守恒和 matched 指纹共同支持，但不等同于证明所有生产 workload 都有相同收益。",
        "",
        "## 阶段决策",
        "",
        f"Stage16A aggregate 判定为 `accepted={str(accepted).lower()}`、`decision={decision}`。" + (
            "可以进入 Stage16B 的最小 Admission/Backpressure coupling；Stage16A 结果本身已足以将项目叙事从‘资源优化但在线服务失败’升级为‘在资源效率与在线 SLO 之间实现可验证 Pareto 控制’。"
            if accepted and decision == "pareto_success_enter_stage16b"
            else "当前 aggregate 未授权进入 Stage16B，应先处理未通过 gate。"
        ),
        "",
        "## 下一步",
        "",
        "构建 steady-state open-loop overload workload，先估计 service capacity，再比较无 backpressure 与 KV+queue+SLO-aware backpressure 的 offered/admitted/completed/rejected/SLO-goodput trade-off。Stage16B 不应修改 bounded drain 的冻结参数。",
        "",
        "## 限制",
        "",
        f"- n={matched} 个 matched triples，且 capacity 与 seed 组合来自单一冻结 workload；不做大样本生产显著性外推。",
        "- `interactive_slo_goodput_rps` 在多数矩阵 cell 为 0，Stage16A 的主结论依赖 progress-gap、waiting-age、ITL 与资源指标，而不是 Goodput 提升。",
        "- 绝对 wall-clock 延迟受本机 GPU、模型与运行时影响；对外报告百分比和 matched pair 数时，必须同时保留 raw artifact 与环境元数据。",
        "",
    ]
    return "\n".join(lines)


def markdown_stats(comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage16A 统计附录",
        "",
        "## 方法",
        "",
        "分析单位为冻结的 matched configuration points，不把各 policy cell 当作独立样本。每项比较使用配对差值（lower-is-better 指标定义为 baseline - bounded），报告均值、样本标准差、matched-point bootstrap percentile 95% 区间、精确 sign test、精确 Wilcoxon signed-rank p 值及 Holm 校正后的 p 值。capacity 之间可能共享同一 seed/trace，因此这些区间和 p 值只作为描述性敏感性分析，不视为来自独立生产总体的推断；正式 gate 仍以预注册 ratio-of-totals reduction 与 direction criterion 为准。",
        "",
        "## 配对比较",
        "",
        "| 指标 | baseline → candidate | baseline mean±SD | candidate mean±SD | mean reduction | improved | Sign-test p | Wilcoxon p | Holm p | difference 95% interval |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        baseline = f"{item['baseline']} → {item['candidate']}"
        lines.append(
            f"| `{item['metric']}` | {baseline} | {item['baseline_mean']:,.3f} ± {item['baseline_sd']:,.3f} | {item['candidate_mean']:,.3f} ± {item['candidate_sd']:,.3f} | {item['mean_reduction']*100:.2f}% | {item['improved_pairs']}/{item['n']} | {item['sign_test_p']:.4f} | {item['wilcoxon_p']:.4f} | {item['wilcoxon_holm_p']:.4f} | [{item['difference_ci95_low']:,.3f}, {item['difference_ci95_high']:,.3f}] |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "Wilcoxon、sign test 与 bootstrap 均把冻结配置点作为 matched observations；由于相同 seed 跨 capacity 共享 trace，这里不声称观测独立或外推总体显著性。多重校正覆盖本附录 6 项主要对比。即使某个 p 值未达到传统阈值，也不推翻预注册 gate；相反，若 aggregate gate 与统计检验方向冲突，应优先保留完整 pair-level 数据并降级结论。",
            "",
        ]
    )
    return "\n".join(lines)


def markdown_catalog(aggregate_path: Path, matched: int) -> str:
    return f"""# Stage16A 图表目录

| 图 | 文件 | 目的 | 数据源 | 主要观察 | 解读边界 |
|---|---|---|---|---|---|
| Figure 1 | `figures/stage16_pareto.svg` / `.png` | 展示 Actual Recompute 与 ITL P99 的系统 Pareto 关系 | `{aggregate_path.as_posix()}`，{matched} triples × 3 policies | bounded 点整体位于 pressure 与 unbounded 之间：资源成本低于 pressure，ITL 尾延迟低于 unbounded | 每点是一个冻结 matched cell；不代表连续 workload 曲线 |
| Figure 2 | `figures/stage16_reductions.svg` / `.png` | 汇总 bounded 相对指定 baseline 的 5 项主要降低比例 | 同上 | fairness、ITL、resource 指标均为正向降低，且 9/9 direction 一致 | 柱形为 9 个 pair 的均值，不替代逐 pair 数据 |

## 图表解释清单

### Figure 1

- Purpose：回答 bounded 是否形成资源效率与在线 tail latency 之间的中间 Pareto 点。
- Observation：bounded 的 ITL P99 明显低于 unbounded，同时 Actual Recompute 低于 pressure baseline；具体数值以 tooltip/CSV 和 aggregate 为准。
- Interpretation：支持 bounded episode 将长期 phase-exclusive drain 截断，同时保留部分 recompute-aware victim selection 的机制解释。
- Implication：Stage16A 可以进入 Admission/Backpressure overload 实验，而不是继续 sweep drain 参数。

### Figure 2

- Purpose：将预注册 gate 对应的 reduction 统一展示。
- Observation：progress-gap、waiting-age、ITL、recompute、preemption 均为正向降低，aggregate 和 9/9 direction gate 均通过。
- Interpretation：bounded 不是只优化一个指标；它同时改变 fairness、latency 和 resource 三个维度。
- Implication：该结果支持跨层 Scheduler + KV resource + SLO guard 的系统结论，但必须保留 benchmark 口径。

## 复现

CSV 由 `experiments/analyze_stage16_results.py` 从 strict aggregate 重新导出；SVG 与 PNG 直接消费同一份 rows/comparisons，PNG 以 450 DPI metadata 输出。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    aggregate_path = args.aggregate.resolve()
    aggregate = load_aggregate(aggregate_path)
    rows = rows_from_aggregate(aggregate)
    comparisons = build_comparisons(rows)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(output / "data/stage16_metrics.csv", rows)
    (output / "analysis-report.md").write_text(markdown_report(aggregate, aggregate_path, rows, comparisons), encoding="utf-8")
    (output / "stats-appendix.md").write_text(markdown_stats(comparisons), encoding="utf-8")
    (output / "figure-catalog.md").write_text(markdown_catalog(aggregate_path, len(rows)), encoding="utf-8")
    write_analysis_figures(output / "figures", rows, comparisons)
    (output / "analysis.json").write_text(json.dumps({"aggregate": str(args.aggregate.resolve()), "rows": rows, "comparisons": comparisons}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "comparisons": len(comparisons)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
