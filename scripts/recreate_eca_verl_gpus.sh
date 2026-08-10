#!/usr/bin/env bash
# Recreate eca-verl with --gpus all AND keep /workspace/verl + host network.
#
# Why commit-from-backup:
#   /workspace/verl often lives in the container writable layer (not a bind mount).
#   Recreating from the base image alone drops verl → PYTHONPATH break.
#
# Usage:
#   DRY_RUN=1 bash scripts/recreate_eca_verl_gpus.sh
#   bash scripts/recreate_eca_verl_gpus.sh
#   FROM_BACKUP=eca-verl-pre8gpu bash scripts/recreate_eca_verl_gpus.sh
set -euo pipefail

CONTAINER=${CONTAINER:-eca-verl}
BACKUP=${BACKUP:-${CONTAINER}-pre8gpu}
FROM_BACKUP=${FROM_BACKUP:-$BACKUP}   # prefer old container as filesystem source
GPUS=${GPUS:-all}
DRY_RUN=${DRY_RUN:-0}
COMMIT_TAG=${COMMIT_TAG:-eca-verl:sgl055-8gpu-$(date +%Y%m%d)}

if ! docker info >/dev/null 2>&1; then
  if sg docker -c 'docker info' >/dev/null 2>&1; then
    exec sg docker -c "cd \"${PWD}\" && DRY_RUN=${DRY_RUN} GPUS=${GPUS@Q} CONTAINER=${CONTAINER} BACKUP=${BACKUP} FROM_BACKUP=${FROM_BACKUP} bash \"${BASH_SOURCE[0]}\""
  fi
  echo "ERROR: cannot talk to docker" >&2
  exit 1
fi

SRC="$CONTAINER"
if docker inspect "$FROM_BACKUP" >/dev/null 2>&1; then
  SRC="$FROM_BACKUP"
elif ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "ERROR: neither $CONTAINER nor $FROM_BACKUP found" >&2
  exit 1
fi

# Prefer image that still has /workspace/verl
HAS_VERL=0
if docker exec "$SRC" bash -lc 'test -d /workspace/verl' 2>/dev/null; then
  HAS_VERL=1
fi

IMAGE=$(docker inspect -f '{{.Config.Image}}' "$SRC")
WORKDIR=$(docker inspect -f '{{.Config.WorkingDir}}' "$SRC")
SHM=$(docker inspect -f '{{.HostConfig.ShmSize}}' "$SRC")
NETWORK=$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$SRC")
SHM_ARG=()
[[ "$SHM" != "0" && -n "$SHM" ]] && SHM_ARG=(--shm-size "$SHM")

mapfile -t BIND_LINES < <(docker inspect -f '{{range .Mounts}}{{if eq .Type "bind"}}{{.Source}}|{{.Destination}}|{{.RW}}{{println}}{{end}}{{end}}' "$SRC")
MOUNT_ARGS=()
for line in "${BIND_LINES[@]}"; do
  [[ -z "$line" ]] && continue
  IFS='|' read -r src dst rw <<<"$line"
  if [[ "$rw" == "true" ]]; then
    MOUNT_ARGS+=(-v "${src}:${dst}")
  else
    MOUNT_ARGS+=(-v "${src}:${dst}:ro")
  fi
done

mapfile -t ENV_LINES < <(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$SRC")
ENV_ARGS=()
for line in "${ENV_LINES[@]}"; do
  case "$line" in
    ""|PATH=*|HOSTNAME=*|HOME=*|TERM=*|NVIDIA_*) continue ;;
  esac
  ENV_ARGS+=(-e "$line")
done

# Always host network so container can hit host Candidate-BM25 :8001
NET_ARGS=(--network host)
# Keep --ipc host for NCCL
IPC_ARGS=(--ipc host)

echo "=== recreate plan ==="
echo "filesystem_source=$SRC has_verl=$HAS_VERL"
echo "base_image=$IMAGE"
echo "commit_tag_if_needed=$COMMIT_TAG"
echo "gpus=$GPUS network=host"
echo "bind mounts:"
printf '  %s\n' "${MOUNT_ARGS[@]:-(none)}"

CREATE_IMAGE="$IMAGE"
if [[ "$HAS_VERL" == "1" ]]; then
  echo "Will docker commit $SRC → $COMMIT_TAG (preserves /workspace/verl)"
  CREATE_IMAGE="$COMMIT_TAG"
else
  echo "WARNING: $SRC has no /workspace/verl — recreate may still fail PYTHONPATH"
fi

CREATE=(
  docker create
  --name "$CONTAINER"
  --gpus "$GPUS"
  "${SHM_ARG[@]}"
  "${NET_ARGS[@]}"
  "${IPC_ARGS[@]}"
  -w "${WORKDIR:-/workspace}"
  "${MOUNT_ARGS[@]}"
  "${ENV_ARGS[@]}"
  "$CREATE_IMAGE"
  sleep infinity
)

printf 'CMD: '
printf '%q ' "${CREATE[@]}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 — no changes"
  exit 0
fi

# Stop current working container if present
docker stop "$CONTAINER" >/dev/null 2>&1 || true

if [[ "$HAS_VERL" == "1" ]]; then
  docker commit "$SRC" "$COMMIT_TAG"
fi

# Rotate names: keep SRC as backup if it is the live container
if [[ "$SRC" == "$CONTAINER" ]]; then
  if docker inspect "$BACKUP" >/dev/null 2>&1; then
    docker rm -f "$BACKUP" >/dev/null
  fi
  docker rename "$CONTAINER" "$BACKUP"
elif docker inspect "$CONTAINER" >/dev/null 2>&1; then
  # live container is broken; remove it, keep FROM_BACKUP
  docker rm -f "$CONTAINER" >/dev/null
fi

"${CREATE[@]}"
docker start "$CONTAINER" >/dev/null

echo "=== verify ==="
docker exec "$CONTAINER" bash -lc 'nvidia-smi -L | wc -l; ls -d /workspace/verl /workspace/deepresearch; curl -sf http://127.0.0.1:8001/health || echo NEED_RETRIEVER_OR_START'
docker exec "$CONTAINER" bash -lc 'python -c "import torch; print(\"torch.cuda.device_count=\", torch.cuda.device_count())"'
echo "OK. Backup container (if kept): $BACKUP / source was $SRC"
