# Phase 3B — Environment pin (`eca-verl`)

> Frozen after first successful pull on `lyg0250` (2026-08-08).  
> Use the **pinned local tag** for all formal runs; do not re-pull floating `sgl055.latest` blindly.

## Image pin

| Field | Value |
|-------|-------|
| Hub tag (source) | `verlai/verl:sgl055.latest` |
| Repo digest | `sha256:9eeb7f5323c67fcb00975bcb45d64d9e96837f990939df8057912a240e6a7bc7` |
| Image ID | `sha256:569073d8c8e7def24dc12762d986c0aaf70b4f0baef8129c82fb85f5403057de` |
| Local pin tag | `eca-verl:sgl055-pinned-20260808` |
| Approx size | 32.7GB |
| Image created | 2025-11-12 |

```bash
# Prefer pinned local tag
docker images eca-verl:sgl055-pinned-20260808
```

## Offline / restore backup (host)

```text
META=/data1/hcc/docker_backups/eca-verl_sgl055_20260808.json
TAR_GZ=/data1/hcc/docker_backups/eca-verl_sgl055_pinned_20260808.tar.gz
LOG=/data1/hcc/docker_backups/docker_save.log
README=/data1/hcc/docker_backups/README.md
```

本机日常复用：**不依赖 tar**，Docker 本地已有 pin 标签 + 容器 `eca-verl`。  
离线 tar.gz 在后台导出（约 15–40 分钟，体积 ~十几 GB）。

```bash
# 看导出进度
ls -lh /data1/hcc/docker_backups/eca-verl_sgl055_pinned_20260808.tar.gz
tail -f /data1/hcc/docker_backups/docker_save.log

# 恢复（另一台机器 / 清镜像后）
gunzip -c /data1/hcc/docker_backups/eca-verl_sgl055_pinned_20260808.tar.gz | docker load
# 再按 PHASE3B_SETUP.md 用 eca-verl:sgl055-pinned-20260808 重建容器
```

## Persistent container

| Field | Value |
|-------|-------|
| Name | `eca-verl` |
| GPUs | `4,5,6,7` |
| Mounts | `/data1/hcc/deepresearch` → `/workspace/deepresearch` |
| | `/data1/hcc/.hf_home` → `/root/.cache/huggingface` |

```bash
docker start eca-verl
docker exec -it eca-verl bash
```

## Versions observed inside container (base image, before `pip install verl`)

```text
python        3.12.12
torch         2.8.0+cu129  cuda 12.9
transformers  4.57.1
sglang        0.5.5
ray           2.51.1
flash_attn    2.8.1
gpu_count     4 × A100-SXM4-80GB
SFT-v1 mount  OK
verl package  NOT preinstalled  ← required next step (official docs)
```

Official flow for `verlai/verl:sgl*.latest`: image = SGLang/train stack only; then:

```bash
git clone https://github.com/verl-project/verl && cd verl
pip3 install --no-deps -e .
```

After that succeeds, optionally:

```bash
docker commit eca-verl eca-verl:sgl055-ready-20260808
docker save eca-verl:sgl055-ready-20260808 -o /data1/hcc/docker_backups/eca-verl_sgl055_ready_20260808.tar
```

and record the verl git SHA in this file.

## veRL source pin (installed 2026-08-08)

| Field | Value |
|-------|-------|
| Path in container | `/workspace/verl` (editable) |
| Package | `verl==0.9.0.dev0` |
| Git SHA | `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71` |
| Ready image | `eca-verl:sgl055-ready-20260808` (docker commit after install) |
| Smoke-ok image | `eca-verl:sgl055-smokeok-20260808` (after 3B1 GRPO 5-step pass + TransferQueue + sgl055 patch) |

```bash
docker images eca-verl:sgl055-smokeok-20260808
# recreate later from smokeok tag if needed; or start existing container:
docker start eca-verl && docker exec -it eca-verl bash
```

Extra packages baked into smokeok (beyond ready): `TransferQueue`, runtime sgl055 compat patch applied on each launch via `scripts/patch_verl_sgl055_compat.py`.

Offline tar (background `docker save | gzip`):

```text
/data1/hcc/docker_backups/eca-verl_sgl055_smokeok_20260808.tar.gz
```

3B0/3B1 code lives under `src/rl/` + `configs/rl/` + `docs/PHASE3B0.md` (host mount `/workspace/deepresearch`).
