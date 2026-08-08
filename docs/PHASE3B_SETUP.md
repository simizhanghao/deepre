# Phase 3B0 — veRL Docker environment setup

> First executable step only: image + container. No GRPO train yet.

## Host paths

```text
REPO=/data1/hcc/deepresearch
HF=/data1/hcc/.hf_home
SFT_V1=$REPO/outputs/sft_qwen25_3b_coldstart_v1_merged
GPUS=4,5,6,7
IMAGE=verlai/verl:sgl055.latest
```

If pull of `sgl055.latest` fails, try `verlai/verl:sgl059.latest` (newer stable on Docker Hub) and pin that instead.

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

## Next after env OK (not this step)

- Candidate-BM25 retrieve server + veRL `BaseTool`
- Train subset parquet / agent data
- GRPO yaml: 4 GPU, n=4, EM+0.1 format, 2–5 steps
- Mask verification dump
