from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def append_experiment_log(
    log_path: str | Path,
    experiment_name: str,
    config_path: str | Path,
    run_dir: str | Path,
    metrics: dict[str, Any],
) -> None:
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_config = _load_run_config(run_dir)
    lines = [
        "",
        f"## {timestamp} - {experiment_name}",
        "",
        f"- 配置文件：`{config_path}`",
        f"- 结果目录：`{run_dir}`",
        f"- 数据集：`{metrics['dataset']['source']}`；节点数={metrics['dataset']['num_nodes']}；"
        f"训练超边={metrics['dataset']['train_edges']}；验证超边={metrics['dataset']['val_edges']}；"
        f"测试超边={metrics['dataset']['test_edges']}",
        "",
        "### 本次执行操作",
        "",
        "1. 读取配置文件并构建本轮实验数据集。",
        f"2. 使用 `{_model_description(run_config)}` 作为超边预测 backbone，分别训练不同负采样器对应的模型。",
        "3. 使用统一的 size-matched 测试负样本评估预测性能，避免评估负样本差异直接影响 AUC/AUPR。",
        "4. 使用各自负采样器生成测试负样本，计算难度、Jaccard、最近正样本相似度和通用结构风险分数。",
        f"5. 候选生成与选择设置：{_sampling_description(run_config)}",
        "6. 将完整结果写入 `metrics.json`，并把摘要追加到本实验记录。",
        "",
        "### 得到的结果",
        "",
        "| 负采样器 | AUC | AUPR | 难度均值 | hard ratio | future hit | selected in-band | candidate in-band | Jaccard | nearest-positive | nearest-other | closure | co-walk | hitting | residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for sampler_name, sampler_metrics in metrics["samplers"].items():
        lines.append(
            "| {name} | {auc:.4f} | {aupr:.4f} | {hardness:.4f} | {hard_ratio:.4f} | {future_hit:.4f} | {selected_band:.4f} | {candidate_band:.4f} | {jaccard:.4f} | {nearest:.4f} | {nearest_other:.4f} | {closure_risk:.4f} | {cowalk_risk:.4f} | {hitting_risk:.4f} | {residual_risk:.4f} |".format(
                name=sampler_name,
                auc=sampler_metrics["auc"],
                aupr=sampler_metrics["aupr"],
                hardness=sampler_metrics.get("hardness_mean", 0.0),
                hard_ratio=sampler_metrics.get("hard_negative_ratio", 0.0),
                future_hit=sampler_metrics.get("future_positive_hit_rate", 0.0),
                selected_band=sampler_metrics.get("rerank_selected_in_band_rate", 0.0),
                candidate_band=sampler_metrics.get("rerank_candidate_in_band_rate", 0.0),
                jaccard=sampler_metrics.get("jaccard_mean", 0.0),
                nearest=sampler_metrics.get("nearest_positive_similarity_mean", 0.0),
                nearest_other=sampler_metrics.get("nearest_other_positive_similarity_mean", 0.0),
                closure_risk=sampler_metrics.get("closure_risk_mean", 0.0),
                cowalk_risk=sampler_metrics.get("cowalk_risk_mean", 0.0),
                hitting_risk=sampler_metrics.get("hitting_risk_mean", 0.0),
                residual_risk=sampler_metrics.get("residual_risk_mean", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "### 结合研究内容的解释",
            "",
            _build_research_interpretation(metrics),
            "",
            "### 学到什么",
            "",
            "- 自动记录：优先判断候选是否同时满足风险可行性和边界困难性。若 `selected in-band` 明显高于 `candidate in-band`，说明选择器有效；若二者都低，瓶颈在候选生成分布。",
            "",
            "### 问题或失败点",
            "",
            "- 自动记录：如果所有方法 AUC 接近 0.5，应先检查 backbone 和数据划分，而不是直接否定负采样策略。",
            "",
            "### 下一步",
            "",
            "- 自动记录：若 future hit 低但 hard ratio 不足，下一步应改进候选生成 proposal；若 hard ratio 高但 future hit 高，下一步应收紧或重构结构风险约束。",
            "",
        ]
    )

    with target.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))


def _build_research_interpretation(metrics: dict[str, Any]) -> str:
    samplers = metrics["samplers"]
    if "anchored_random" not in samplers:
        sampler_name, sampler_metrics = next(iter(samplers.items()))
        neighbor_rate = sampler_metrics.get("neighbor_replacement_rate")
        rerank_multiplier = sampler_metrics.get("rerank_candidate_multiplier")
        rerank_top_k = sampler_metrics.get("rerank_top_k")
        lines = [
            f"本轮主要观察 `{sampler_name}` 在当前配置下的预测性能、负样本难度与结构风险。"
            f"AUC={sampler_metrics['auc']:.4f}，AUPR={sampler_metrics['aupr']:.4f}，"
            f"hardness_mean={sampler_metrics.get('hardness_mean', 0.0):.4f}，"
            f"hard_negative_ratio={sampler_metrics.get('hard_negative_ratio', 0.0):.4f}，"
            f"future_positive_hit_rate={sampler_metrics.get('future_positive_hit_rate', 0.0):.4f}，"
            f"nearest_other_positive_similarity_mean={sampler_metrics.get('nearest_other_positive_similarity_mean', 0.0):.4f}。",
        ]
        if sampler_metrics.get("weighted_negative_loss_enabled", 0.0) > 0.0:
            lines.append(
                "本轮启用了可靠性加权负例损失："
                f"负例平均权重为 {sampler_metrics.get('negative_loss_weight_mean', 1.0):.4f}，"
                f"最小权重为 {sampler_metrics.get('negative_loss_weight_min', 1.0):.4f}，"
                f"平均 positive-support risk 为 {sampler_metrics.get('negative_loss_positive_support_risk_mean', 0.0):.4f}。"
                "该指标用于判断高风险 hard negatives 是否被降低负标签梯度，而不是被完全删除。"
            )
        if sampler_metrics.get("negative_label_loss_gce_enabled", 0.0) > 0.0:
            lines.append(
                "本轮启用了 generalized cross entropy 负标签目标："
                f"q={sampler_metrics.get('negative_label_loss_q', 0.0):.4f}，"
                f"起始轮次为 {sampler_metrics.get('negative_label_loss_start_epoch', 1.0):.0f}。"
                "该目标用于限制疑似假负例造成的无界 BCE 梯度，并检验是否能在降低 future hit 的同时减少 AUPR 损失。"
            )
        if sampler_metrics.get("negative_label_loss_nnpu_enabled", 0.0) > 0.0:
            lines.append(
                "本轮启用了 non-negative PU 负标签目标："
                f"估计正例污染率为 {sampler_metrics.get('negative_label_loss_positive_prior', 0.0):.4f}，"
                f"起始轮次为 {sampler_metrics.get('negative_label_loss_start_epoch', 1.0):.0f}。"
                "该目标把采样负例视作 open-world unlabeled 集合，并用可靠性权重的互补量估计潜在假负例比例。"
            )
        if neighbor_rate is not None:
            lines.append(
                f"邻域替换比例为 {neighbor_rate:.4f}，表示候选生成确实使用了 anchor 条件下的局部共现邻域；"
                "若 hardness 同时提高且 nearest-other 相似度不升高，则说明邻域 proposal 与风险控制可以形成有效互补。"
            )
        residual_safe_rate = sampler_metrics.get("residual_safe_replacement_rate")
        if residual_safe_rate is not None and residual_safe_rate > 0.0:
            lines.append(
                f"residual-safe 替换比例为 {residual_safe_rate:.4f}。"
                "该 proposal 优先选择与 anchor 有中等结构关系、但 degree-corrected residual risk 较低的候选节点，"
                "用于提高风险可行域内的 boundary-candidate ratio，而不是放松 positive-support risk 约束。"
            )
        if sampler_metrics.get("residual_safe_controller_enabled", 0.0) > 0.0:
            lines.append(
                "本轮启用了 residual-safe 混合比例闭环控制："
                f"最终 residual-safe proposal 概率为 {sampler_metrics.get('residual_safe_controller_final_probability', 0.0):.4f}，"
                f"增加次数为 {sampler_metrics.get('residual_safe_controller_increase_updates', 0.0):.0f}，"
                f"降低次数为 {sampler_metrics.get('residual_safe_controller_decrease_updates', 0.0):.0f}。"
                "该控制器在 hard-negative 覆盖不足时提高 residual-safe 占比，在 fallback、采样成本或验证性能退化时降低占比。"
            )
        if sampler_metrics.get("primal_dual_enabled", 0.0) > 0.0:
            lines.append(
                "本轮启用了 primal-dual 风险约束边界选择："
                f"平均目标函数值为 {sampler_metrics.get('primal_dual_objective_mean', 0.0):.4f}，"
                f"平均边界难度项为 {sampler_metrics.get('primal_dual_hardness_mean', 0.0):.4f}，"
                f"平均风险惩罚项为 {sampler_metrics.get('primal_dual_risk_penalty_mean', 0.0):.4f}。"
                f"对偶乘子均值为 nearest={sampler_metrics.get('primal_dual_lambda_nearest_mean', 0.0):.4f}，"
                f"closure={sampler_metrics.get('primal_dual_lambda_closure_mean', 0.0):.4f}，"
                f"co-walk={sampler_metrics.get('primal_dual_lambda_cowalk_mean', 0.0):.4f}，"
                f"hitting={sampler_metrics.get('primal_dual_lambda_hitting_mean', 0.0):.4f}，"
                f"residual={sampler_metrics.get('primal_dual_lambda_residual_mean', 0.0):.4f}。"
                "这些乘子只由训练阶段的 positive-support risk 预算违反更新，用于避免按数据集手工调采样规则。"
            )
        if rerank_multiplier is not None:
            lines.append(
                f"本轮 rerank 候选倍数为 {rerank_multiplier:.0f}，top-k 为 {rerank_top_k:.0f}。"
                "候选倍数越大，越接近 boundary-aware candidate selection，但计算成本也会线性增加。"
            )
        selected_band = sampler_metrics.get("rerank_selected_in_band_rate")
        candidate_band = sampler_metrics.get("rerank_candidate_in_band_rate")
        if selected_band is not None and candidate_band is not None:
            lines.append(
                f"边界区间命中率：候选池为 {candidate_band:.4f}，最终选中样本为 {selected_band:.4f}。"
                "若选中率明显高于候选池，说明 boundary selector 在工作；若绝对值仍低，则主要瓶颈是风险可行域内缺少边界附近候选。"
            )
        adversarial_enabled = sampler_metrics.get("adversarial_proposal_enabled", 0.0) > 0.0
        if adversarial_enabled:
            lines.append(
                "本轮启用了风险约束下的对抗式边界 proposal。"
                f"平均探测候选数为 {sampler_metrics.get('adversarial_probe_candidates_mean', 0.0):.4f}，"
                f"平均可行候选数为 {sampler_metrics.get('adversarial_feasible_candidates_mean', 0.0):.4f}，"
                f"风险可行候选比例为 {sampler_metrics.get('adversarial_feasible_candidate_rate', 0.0):.4f}，"
                f"proposal 选中样本 in-band rate 为 {sampler_metrics.get('adversarial_selected_in_band_rate', 0.0):.4f}。"
                "该指标用于判断瓶颈是否已经从候选生成转移到最终训练或风险约束。"
            )
            if sampler_metrics.get("adversarial_elite_enabled", 0.0) > 0.0:
                lines.append(
                    "本轮启用了 elite feedback 对抗式候选搜索："
                    f"平均 elite 更新次数为 {sampler_metrics.get('adversarial_elite_updates_mean', 0.0):.4f}，"
                    f"elite 替换节点使用率为 {sampler_metrics.get('adversarial_elite_replacement_rate', 0.0):.4f}。"
                    "该指标用于确认候选生成本身是否根据边界收益和风险可行性发生了自适应偏移。"
                )
        return "\n\n".join(lines)

    anchored = samplers["anchored_random"]
    random_metrics = samplers.get("random")
    size_metrics = samplers.get("size_matched")

    lines = [
        "`anchored_random` 是早期 MVP 采样器，只用于 smoke test 或历史对照；它不再代表当前主方法。",
        f"本轮 `anchored_random` 的难度均值为 {anchored['hardness_mean']:.4f}，Jaccard 均值为 {anchored['jaccard_mean']:.4f}，最近正样本相似度为 {anchored['nearest_positive_similarity_mean']:.4f}。",
    ]

    if random_metrics is not None:
        lines.append(
            "相对 `random`，`anchored_random` 的 Jaccard 变化为 "
            f"{anchored['jaccard_mean'] - random_metrics['jaccard_mean']:+.4f}，"
            "难度变化为 "
            f"{anchored['hardness_mean'] - random_metrics['hardness_mean']:+.4f}。"
            "如果二者为正，说明 anchor 机制确实让负样本更接近对应正超边，也更难被当前模型区分。"
        )

    if size_metrics is not None:
        lines.append(
            "相对 `size_matched`，`anchored_random` 的预测 AUC 变化为 "
            f"{anchored['auc'] - size_metrics['auc']:+.4f}，AUPR 变化为 "
            f"{anchored['aupr'] - size_metrics['aupr']:+.4f}。"
            "这可以作为第一阶段性能信号，但仍需要多随机种子和真实数据集确认。"
        )

    lines.append(
        "需要注意的是，早期 anchored-only 结果不能作为正文主结果。当前主线应比较风险可行域、边界选择、future hit 和 hard ratio，而不是只看 anchor 是否提高 Jaccard。"
    )
    return "\n\n".join(lines)


def _load_run_config(run_dir: str | Path) -> dict[str, Any]:
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def _model_description(config: dict[str, Any]) -> str:
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        return "unknown"
    model_type = model_config.get("type", "mean")
    if model_type == "bipartite":
        layers = model_config.get("message_passing_layers", 1)
        return f"bipartite message-passing, layers={layers}"
    return str(model_type)


def _sampling_description(config: dict[str, Any]) -> str:
    sampling_config = config.get("sampling")
    if not isinstance(sampling_config, dict):
        return "未读取到采样配置。"
    replacement_strategy = sampling_config.get("replacement_strategy", "random")
    rerank_strategy = sampling_config.get("rerank_selection_strategy", "none")
    rerank_multiplier = sampling_config.get("rerank_candidate_multiplier", 1)
    return (
        f"replacement_strategy=`{replacement_strategy}`，"
        f"rerank_strategy=`{rerank_strategy}`，"
        f"candidate_multiplier={rerank_multiplier}。"
    )
