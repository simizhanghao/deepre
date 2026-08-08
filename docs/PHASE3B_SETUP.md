# Phase 3B0 — veRL Docker environment setup

> **Status (2026-08-08):** image pinned + veRL installed; **3B1 GRPO micro-smoke PASS**.  
> Prefer tag `eca-verl:sgl055-smokeok-20260808`. Pin: [`PHASE3B_ENV_PIN.md`](PHASE3B_ENV_PIN.md).

## Host paths

```text
REPO=/data1/hcc/deepresearch
HF=/data1/hcc/.hf_home
SFT_V1=$REPO/outputs/sft_qwen25_3b_coldstart_v1_merged
GPUS=4,5,6,7
IMAGE=eca-verl:sgl055-pinned-20260808   # preferred (local pin of sgl055.latest)
# Hub source (already pulled): verlai/verl:sgl055.latest
# Digest: sha256:9eeb7f5323c67fcb00975bcb45d64d9e96837f990939df8057912a240e6a7bc7
```

## 1) Pull image

```bash
docker pull verlai/verl:sgl055.latest
```

## 2) Create persistent container (do not use lf-sft)

```bash
docker rm -f eca-verl 2>/dev/null

docker create \
  --runtime=nvidia \
  --gpus '"device=4,5,6,7"' \
  --network=host \
  --ipc=host \
  --shm-size=32g \
  --cap-add=SYS_ADMIN \
  -e NVIDIA_VISIBLE_DEVICES=4,5,6,7 \
  -v /data1/hcc/deepresearch:/workspace/deepresearch \
  -v /data1/hcc/.hf_home:/root/.cache/huggingface \
  --name eca-verl \
  verlai/verl:sgl055.latest \
  sleep infinity

docker start eca-verl
docker exec -it eca-verl bash
```

## 3) Inside container — freeze versions (paste back)

```bash
echo "=== IMAGE ==="
# from host:
# docker inspect eca-verl --format '{{.Image}}'
# docker image inspect verlai/verl:sgl055.latest --format '{{index .RepoDigests 0}}'

python - <<'PY'
import torch, transformers, sys
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
try:
    import sglang
    print("sglang", getattr(sglang, "__version__", "?"))
except Exception as e:
    print("sglang", e)
try:
    import verl
    print("verl", getattr(verl, "__version__", verl.__file__))
except Exception as e:
    print("verl", e)
print("gpu_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

ls -lh /workspace/deepresearch/outputs/sft_qwen25_3b_coldstart_v1_merged/config.json
```

Also on **host**:

```bash
docker inspect eca-verl --format 'ImageID={{.Image}}'
docker image inspect verlai/verl:sgl055.latest --format 'Digest={{index .RepoDigests 0}}'
```

Save these into `docs/PHASE3B_ENV_PIN.md` after first success (next commit).

## 4) Sanity: see SFT-v1 from inside

```bash
# still in eca-verl
test -f /workspace/deepresearch/outputs/sft_qwen25_3b_coldstart_v1_merged/model.safetensors.index.json && echo SFT_V1_OK
```

## Stop / resume later

```bash
docker stop eca-verl
docker start eca-verl
docker exec -it eca-verl bash
```

## 5) Install veRL into the running container (required)

Hub image ships SGLang/torch/ray but **not** the `verl` Python package:

```bash
docker exec -it eca-verl bash
# inside:
cd /workspace
git clone https://github.com/verl-project/verl.git
cd verl
pip3 install --no-deps -e .
python -c "import verl; print(verl.__file__)"
git rev-parse HEAD
```

Then on **host**, commit a ready image and (optionally) re-save tar:

```bash
docker commit eca-verl eca-verl:sgl055-ready-20260808
# optional second backup after verl install
# docker save eca-verl:sgl055-ready-20260808 -o /data1/hcc/docker_backups/eca-verl_sgl055_ready_20260808.tar
```

Record the verl git SHA into `PHASE3B_ENV_PIN.md`.

## Next after verl import OK

See **[`PHASE3B0.md`](PHASE3B0.md)** — scaffolding is in-repo:

```bash
python scripts/build_grpo_smoke_dataset.py --n-train 128
python scripts/start_candidate_retrieval_server.py --port 8001
python scripts/audit_response_mask.py
# only after gates: bash scripts/run_grpo_smoke.sh   # inside eca-verl
```
