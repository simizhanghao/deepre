# Qwen3-30B 三线总计划

## 目标与边界

本目录把后续工作拆成三条互不阻塞的线。三条线共享代码、数据和最终模型，但不共享叙事与验收标准。

| 目录 | 任务 | 当前状态 | 交付物 |
|---|---|---|---|
| `xiangmu/` | 强最终 DeepResearch Agent | **当前唯一执行线** | 30B Base / SFT / Evidence-GRPO、冻结 dev 选模、可复现实验表 |
| `web/` | 真实 Web 与演示 | 暂停，等待项目模型冻结 | Web adapter、BrowseComp/GAIA-text、Demo |
| `lunwen/` | 因果诊断论文 | 暂停，等待项目模型冻结 | Counterfactual Search Audit、2×2 泛化矩阵、论文 |

## 冻结的项目方法

```text
Qwen3-30B-A3B-Instruct-2507
  → 原 coldstart_v1 数据做 LoRA SFT
  → 合并为完整 HF checkpoint
  → 原 Evidence GRPO（lambda_e=0.5）
  → 原 Candidate-BM25
  → Exact VeXact + EcaSearchAgentLoop
  → 200/400/600/800 frozen-dev 选模
```

本线禁止引入 CUR、DSSR、Step Gate、Root Pivot、cost routing、reward 改造和 optimizer sweep。它们属于 `lunwen/` 的研究证据，不属于最终产品。

## 总体阶段门

1. **P0 静态预检**：路径、8×GPU、磁盘、数据 hash、框架版本通过。
2. **P1 Base 兼容性**：Qwen3-MoE 能被 LlamaFactory、HF、VeOmni/VeXact 正确识别；Exact smoke 通过。
3. **P2 SFT**：只使用冻结 `coldstart_v1`；协议标签和 observation role 不变；合并模型健康。
4. **P3 Evidence-GRPO**：先 1-step smoke；再依次训练到 200、400、600、800。每段结束立即做 frozen-dev，不碰 Test。
5. **P4 冻结 checkpoint**：按预先写定的选择规则选唯一 best；记录模型、tokenizer、数据与代码 hash。
6. **P5 Controlled final evaluation**：best 冻结后才允许第一次打开 sealed HotpotQA Test。
7. **P6 项目包装**：最终主表、案例、环境、README；之后 `web/` 与 `lunwen/` 才进入执行。

## 纪律

- 训练和评测命令由用户亲自执行；Codex 只维护脚本、配置、命令与结果分析。
- smoke 失败不得直接启动长训练。
- frozen dev 是唯一选模集；sealed Test 不参与调参、早停或 checkpoint 选择。
- 任何超出冻结方案的改动先写成显式变更，不在运行命令中临时追加。

