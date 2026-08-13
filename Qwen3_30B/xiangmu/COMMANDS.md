# 用户执行命令

以下命令按阶段执行。Codex 不会代替用户启动下载、训练或评测。

## 0. 提取冻结输入并预检

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/00_stage_frozen_data.sh
bash scripts/00_preflight.sh
```

第一条命令是整个提纯项目对旧仓库的唯一数据读取：把冻结 SFT、RL parquet、BM25 index 和 dev 快照复制到 `xiangmu/data/` 并写 hash。之后主线只使用本地副本。P0 必须看到 `P0_PREFLIGHT_PASS`。预检会硬性要求至少 450 GiB 可用空间，因为 30B base、SFT merged 与一套可 resume 的 optimizer state 都很大。

## 1. 下载模型并准备冻结 SFT 数据

直接在终端安装 ModelScope 并下载，不需要运行下载脚本：

```bash
python3 -m pip install --user -U modelscope
python3 -c "import modelscope; print(modelscope.__version__)"
mkdir -p /data1/hcc/deepresearch/Qwen3_30B/model
modelscope download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir /data1/hcc/deepresearch/Qwen3_30B/model \
  --max-workers 8
```

下载结束后注册本地冻结 SFT 数据：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/02_prepare_sft_data.sh 2>&1 | tee logs/prepare_sft_data.log
```

模型使用前一节给出的 ModelScope 终端命令下载。下载中断后重复相同命令即可续传；模型固定保存在 `/data1/hcc/deepresearch/Qwen3_30B/model`。`01_download_model.sh` 只是可选封装，不是必须入口。

先做 Base 的 VeXact compatibility；该命令只占一张卡：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
COMPAT_GPU=0 bash scripts/05_check_vexact_model.sh \
  /data1/hcc/deepresearch/Qwen3_30B/model qwen3_base
```

只有出现 `VEXACT_MODEL_COMPAT_PASS` 才进入 SFT。

## 2. 4 卡 SFT（GPU 0–3）

建议在 tmux 中启动：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
tmux new-session -d -s q30_sft \
  "cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu && bash scripts/03_train_sft.sh"
tmux attach -t q30_sft
```

查看日志/GPU而不干扰训练：

```bash
tail -F /data1/hcc/deepresearch/Qwen3_30B/xiangmu/logs/sft_*.log
watch -n 1 nvidia-smi
```

训练成功后合并；merge 使用 CPU，不要在已有目标上覆盖：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/04_merge_sft.sh
COMPAT_GPU=0 bash scripts/05_check_vexact_model.sh \
  artifacts/models/qwen3_30b_sft_merged qwen3_sft
```

## 3. 冻结 dev 的 Base / SFT 基线

每次评测占一张 80G GPU，顺序执行：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh qwen3_base \
  /data1/hcc/deepresearch/Qwen3_30B/model
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh qwen3_sft \
  artifacts/models/qwen3_30b_sft_merged
```

## 4. 启动 Candidate-BM25

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
tmux new-session -d -s q30_retriever \
  "cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu && bash scripts/06_start_retriever.sh 2>&1 | tee logs/retriever.log"
curl -s http://127.0.0.1:8001/health
```

## 5. 4 卡 Exact RL smoke（GPU 0–3）

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
tmux new-session -d -s q30_rl_smoke \
  "cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu && bash scripts/07_run_evidence_grpo.sh smoke"
tmux attach -t q30_rl_smoke
```

必须看到 `GRPO_SEGMENT_PASS step=1`，并确认没有 OOM、NaN、Agent loop error、reward 全零或格式崩溃，才允许正式训练。

## 6. 200/400/600/800 分段训练与评测

每段严格执行同一循环。以下先以 step 200 为例：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
tmux new-session -d -s q30_grpo_200 \
  "cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu && bash scripts/07_run_evidence_grpo.sh segment 200"
tmux attach -t q30_grpo_200

EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh step200 \
  artifacts/evidence_grpo_ckpt/global_step_200/actor/huggingface
python3 scripts/09_select_best.py --allow-partial
ALLOW_BEST_REPLACE=1 bash scripts/10_promote_best.sh
python3 scripts/12_build_result_table.py
```

确认当前 checkpoint 过 health gate 且 `artifacts/best_hf/FROZEN_GRPO_STEP` 已写好，才继续下一段：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu

tmux new-session -d -s q30_grpo_400 \
  "cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu && bash scripts/07_run_evidence_grpo.sh segment 400"
tmux attach -t q30_grpo_400
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh step400 \
  artifacts/evidence_grpo_ckpt/global_step_400/actor/huggingface
python3 scripts/09_select_best.py --allow-partial
ALLOW_BEST_REPLACE=1 bash scripts/10_promote_best.sh

tmux new-session -d -s q30_grpo_600 \
  "cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu && bash scripts/07_run_evidence_grpo.sh segment 600"
tmux attach -t q30_grpo_600
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh step600 \
  artifacts/evidence_grpo_ckpt/global_step_600/actor/huggingface
python3 scripts/09_select_best.py --allow-partial
ALLOW_BEST_REPLACE=1 bash scripts/10_promote_best.sh

tmux new-session -d -s q30_grpo_800 \
  "cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu && bash scripts/07_run_evidence_grpo.sh segment 800"
tmux attach -t q30_grpo_800
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh step800 \
  artifacts/evidence_grpo_ckpt/global_step_800/actor/huggingface
python3 scripts/09_select_best.py
ALLOW_BEST_REPLACE=1 bash scripts/10_promote_best.sh
python3 scripts/12_build_result_table.py
```

## 7. 状态检查

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/11_status.sh
cat results/checkpoint_selection.json
cat results/FROZEN_DEV_TABLE.md
```

到这里先停。不得自行打开 sealed Test；先由我们审计 selection、hash、完整 dev 表和训练曲线，再给 final Test 命令。
