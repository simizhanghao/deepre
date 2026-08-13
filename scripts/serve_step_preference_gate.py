#!/usr/bin/env python3
"""Batched local HTTP inference service for the frozen Step Preference Gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Work:
    payload: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BaseException | None = None


class GateRuntime:
    def __init__(self, args):
        import numpy as np
        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM

        self.np, self.torch = np, torch
        self.args = args
        self.device = torch.device("cuda")
        self.queue: queue.Queue[Work] = queue.Queue()
        self.root_cache: dict[str, float] = {}
        if args.root_score_map:
            root_rows = json.loads(args.root_score_map.read_text())
            self.root_cache = {str(key): float(value) for key, value in root_rows["scores"].items()}
        self.threshold = float(json.loads((args.gate_dir / "threshold.json").read_text())["threshold"])

        class MLP(nn.Module):
            def __init__(self, width, outputs=1, sigmoid=False):
                super().__init__()
                layers = [nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, outputs)]
                if sigmoid:
                    layers.append(nn.Sigmoid())
                self.network = nn.Sequential(*layers)

            def forward(self, values):
                return self.network(values)

        self.model = AutoModelForCausalLM.from_pretrained(
            str(args.model), dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
        ).cuda().eval()
        self.base = self.model.model
        self.pad_id = int(self.model.config.pad_token_id)

        root_pca = np.load(args.root_models.parent / "pca27.npz")
        root_scaler = np.load(args.root_models / "scaler.npz")
        self.root_pca_mean = torch.tensor(root_pca["mean"], dtype=torch.float32, device=self.device)
        self.root_pca_components = torch.tensor(root_pca["components"], dtype=torch.float32, device=self.device)
        self.root_scale_mean = torch.tensor(root_scaler["mean"], dtype=torch.float32, device=self.device)
        self.root_scale = torch.tensor(root_scaler["scale"], dtype=torch.float32, device=self.device)
        self.root_models = []
        for seed in (1, 2, 3):
            model = MLP(64, outputs=2, sigmoid=True).cuda().eval()
            model.load_state_dict(torch.load(args.root_models / f"seed{seed}.pt", map_location=self.device))
            self.root_models.append(model)

        gate_pca = np.load(args.gate_dir / "pca_l27.npz")
        gate_scaler = np.load(args.gate_dir / "scaler.npz")
        self.gate_pca_mean = torch.tensor(gate_pca["mean"], dtype=torch.float32, device=self.device)
        self.gate_pca_components = torch.tensor(gate_pca["components"], dtype=torch.float32, device=self.device)
        self.gate_scale_mean = torch.tensor(gate_scaler["mean"], dtype=torch.float32, device=self.device)
        self.gate_scale = torch.tensor(gate_scaler["scale"], dtype=torch.float32, device=self.device)
        self.gate_models = []
        for seed in (1, 2, 3):
            model = MLP(72).cuda().eval()
            model.load_state_dict(torch.load(args.gate_dir / f"seed{seed}.pt", map_location=self.device))
            self.gate_models.append(model)

        self.freeze_manifest = self._freeze_manifest()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _freeze_manifest(self):
        files = [
            self.args.model / "config.json", self.args.model / "tokenizer.json",
            self.args.model / "tokenizer_config.json", self.args.model / "chat_template.jinja",
            self.args.gate_dir / "pca_l27.npz", self.args.gate_dir / "scaler.npz",
            self.args.gate_dir / "threshold.json", self.args.gate_dir / "summary.json",
            *(self.args.gate_dir / f"seed{seed}.pt" for seed in (1, 2, 3)),
            self.args.root_models.parent / "pca27.npz", self.args.root_models / "scaler.npz",
            *(self.args.root_models / f"seed{seed}.pt" for seed in (1, 2, 3)),
        ]
        if self.args.root_score_map:
            files.append(self.args.root_score_map)
        manifest = {
            "gate": "STEP_GATE_DEPLOYMENT_FREEZE_PASS",
            "threshold": self.threshold,
            "feature_dim": 72,
            "feature_schema": [
                "query_final_token_L27_PCA64", "step_mean_logp", "step_p10_logp",
                "checkpoint_mean_entropy", "step_index", "previous_searches",
                "query_token_length", "max_previous_query_jaccard", "frozen_root_B3",
            ],
            "artifact_sha256": {str(path): sha256_file(path) for path in files},
            "root_score_mode": "frozen_exact_batch32_map" if self.args.root_score_map else "online_native_hf",
            "val3_outcomes_read": False,
            "test_read": False,
        }
        path = self.args.gate_dir / "deployment_freeze.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["manifest_sha256"] = sha256_file(path)
        return manifest

    def submit(self, payload):
        item = Work(payload)
        self.queue.put(item)
        if not item.event.wait(self.args.timeout):
            raise TimeoutError("Gate inference queue timed out")
        if item.error:
            raise item.error
        return item.result

    def _worker(self):
        while True:
            first = self.queue.get()
            batch = [first]
            deadline = time.perf_counter() + self.args.batch_wait_ms / 1000
            while len(batch) < self.args.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                results = self._infer([item.payload for item in batch])
                for item, result in zip(batch, results, strict=True):
                    item.result = result
            except BaseException as exc:
                for item in batch:
                    item.error = exc
            finally:
                for item in batch:
                    item.event.set()

    def _infer(self, payloads):
        torch = self.torch
        sequences, metadata = [], []
        for index, payload in enumerate(payloads):
            sample_id = str(payload["sample_id"])
            if sample_id not in self.root_cache:
                sequences.append([int(x) for x in payload["canonical_prompt_ids"]])
                metadata.append(("root", index))
            sequences.append([int(x) for x in payload["state_prompt_ids"]])
            metadata.append(("state", index))
        lengths = [len(values) for values in sequences]
        width = max(lengths)
        ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for index, values in enumerate(sequences):
            ids[index, :len(values)] = torch.tensor(values, dtype=torch.long, device=self.device)
            mask[index, :len(values)] = 1
        captured = []
        handle = self.base.layers[26].register_forward_hook(
            lambda _module, _inputs, output: captured.append(output[0] if isinstance(output, tuple) else output)
        )
        with torch.inference_mode():
            output = self.base(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        handle.remove()
        h27, final = captured[0], output.last_hidden_state
        state_h27: dict[int, Any] = {}
        for seq_index, (kind, payload_index) in enumerate(metadata):
            payload = payloads[payload_index]
            if kind == "root":
                value = h27[seq_index, lengths[seq_index] - 1].float()
                z = (value - self.root_pca_mean) @ self.root_pca_components.T
                z = (z - self.root_scale_mean) / self.root_scale
                with torch.inference_mode():
                    outcomes = torch.stack([model(z) for model in self.root_models]).mean(0)
                self.root_cache[str(payload["sample_id"])] = float((outcomes[1] - outcomes[0]).cpu())
            else:
                position = int(payload["query_position"])
                if position < 0 or position >= lengths[seq_index]:
                    raise ValueError("query_position outside state prefix")
                state_h27[payload_index] = h27[seq_index, position].float()

        results = []
        for seq_index, (kind, payload_index) in enumerate(metadata):
            if kind != "state":
                continue
            payload = payloads[payload_index]
            start, end = int(payload["checkpoint_abs_start"]), int(payload["checkpoint_abs_end"])
            if not 0 < start < end <= lengths[seq_index]:
                raise ValueError("invalid checkpoint entropy span")
            states = final[seq_index, start - 1:end - 1]
            entropy = []
            with torch.inference_mode():
                for begin in range(0, len(states), self.args.entropy_chunk):
                    logits = self.model.lm_head(states[begin:begin + self.args.entropy_chunk]).float()
                    logz = torch.logsumexp(logits, dim=-1)
                    entropy.append(logz - (torch.softmax(logits, dim=-1) * logits).sum(dim=-1))
                mean_entropy = torch.cat(entropy).mean()
                hidden = state_h27[payload_index]
                z = (hidden - self.gate_pca_mean) @ self.gate_pca_components.T
                scalars = torch.tensor([
                    float(payload["mean_logp"]), float(payload["p10_logp"]), float(mean_entropy),
                    float(payload["step_index"]), float(payload["previous_searches"]),
                    float(payload["query_length"]), float(payload["duplicate_similarity"]),
                    self.root_cache[str(payload["sample_id"])],
                ], dtype=torch.float32, device=self.device)
                features = torch.cat([z, scalars])
                features = (features - self.gate_scale_mean) / self.gate_scale
                probability = torch.stack([torch.sigmoid(model(features)).squeeze() for model in self.gate_models]).mean()
            p = float(probability.cpu())
            results.append({
                "sample_id": str(payload["sample_id"]),
                "probability_search": p,
                "threshold": self.threshold,
                "action": "search" if p >= self.threshold else "continue",
                "bundle_sha256": self.freeze_manifest["manifest_sha256"],
                "feature_audit": {
                    "root_b3": self.root_cache[str(payload["sample_id"])],
                    "mean_entropy": float(mean_entropy.cpu()),
                    "l27_norm": float(torch.linalg.vector_norm(hidden).cpu()),
                    "l27_first8": [float(value) for value in hidden[:8].cpu()],
                },
            })
        # Metadata ordering can interleave root/state but states preserve payload order.
        if len(results) != len(payloads):
            raise RuntimeError("Gate result cardinality mismatch")
        return results


RUNTIME: GateRuntime | None = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send(200, {"status": "ok", "threshold": RUNTIME.threshold, "bundle_sha256": RUNTIME.freeze_manifest["manifest_sha256"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/decide":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            self._send(200, RUNTIME.submit(payload))
        except BaseException as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--gate-dir", type=Path, required=True)
    ap.add_argument("--root-models", type=Path, required=True)
    ap.add_argument("--root-score-map", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8007)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--batch-wait-ms", type=float, default=10)
    ap.add_argument("--entropy-chunk", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()
    global RUNTIME
    RUNTIME = GateRuntime(args)
    print(json.dumps(RUNTIME.freeze_manifest, indent=2), flush=True)
    print(f"STEP_GATE_SERVER_READY=http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
