# DeepResearch-Agent-RL

Evidence-Cost-Aware Deep Research Agent：HotpotQA 上的 Search-R1 风格 Agentic RL（SFT 冷启动 → GRPO）。

## 状态

🟡 **3D1 λ=0.40 FAIL**（search→0、KL 爆炸，已停 @250）— 下一步降 λ 探针或 3D2。  
结果：[RESULTS_BOARD](docs/RESULTS_BOARD.md) · [PHASE3D1](docs/PHASE3D1.md)

| 阶段 | 状态 | 产物 |
|------|------|------|
| SFT-v1 | frozen | `outputs/sft_qwen25_3b_coldstart_v1_merged` |
| 3B Answer-only GRPO | closed @100 | `outputs/rl/grpo_sftv1_smoke/global_step_100` |
| 3C Evidence GRPO | **closed @400** | `outputs/rl/grpo_sftv1_evidence_3c/global_step_400` |
| 3C-GEN | **PASS** | dev-200 Agent EM 0.54 |
| 3D0 | **DONE** | λ_s=**0.40** |
| 3D1 Cost | **next** | fresh SFT-v1 @400 |
| 5M Multimodal | deferred | after text ECA |

### 3C 一句话结果（smoke128）

相对 3B（search=0, answer≈0.20）：3C 将 **search→~1**，**Evidence F1≈0.62**，**Answer≈0.61**；后期 zero_std↑ / 必搜 → 交给 3D。

## 方法概要

- **范式:** 数据 → SFT 冷启动 → 多轮检索 GRPO
- **检索:** Candidate-BM25 + `sample_id`（非开放网页）
- **Reward 演进:** Answer+Format（3B）→ +Evidence F1（3C）→ +Cost（3D）
- **框架:** veRL + SGLang（容器 `eca-verl`）
- **模型:** Qwen2.5-**3B**-Instruct → SFT-v1

## 文档

| Doc | 内容 |
|-----|------|
| [ROADMAP.md](docs/ROADMAP.md) | **主线 v2（门禁触发 3D2/CIPO）** |
| [PHASE3C_GEN.md](docs/PHASE3C_GEN.md) | **当前：val-200 泛化门禁** |
| [NEXT_STEPS.md](docs/NEXT_STEPS.md) | 当前待办顺序 |
| [RESULTS_BOARD.md](docs/RESULTS_BOARD.md) | 全部实验结果总表 |
| [PHASE3C.md](docs/PHASE3C.md) | 3C 结案与窗口指标 |
| [PHASE3B2.md](docs/PHASE3B2.md) | 3B 结案（no-search） |
| [PHASE2_CLOSED.md](docs/PHASE2_CLOSED.md) | SFT-v1 freeze |
| [RUN_LOG.md](docs/RUN_LOG.md) | 逐次跑数日志 |
| [PROJECT_MAP.md](docs/PROJECT_MAP.md) | 仓库地图 |

## 目录结构

见 [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)

## Cursor Agent Skills

本项目使用 `.cursor/skills/` 下的专用 skills（勿装 marketplace skills）。

## 工程约束

- 大权重 / `outputs/` / 多数 `results/` 默认 gitignore；结案 audit 用 `git add -f results/...` 入库
- ckpt 不每 step 保存（GRPO 默认 `SAVE_FREQ=50`）
- 详见 `.cursor/rules/00-deepresearch-guardrails.mdc`
