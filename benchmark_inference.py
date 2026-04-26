#!/usr/bin/env python3
"""Benchmark inference performance across PyTorch/JIT/ONNX/TensorRT paths."""

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO


def _load_actor_critic_barlow_twins() -> Any:
    """Load ActorCriticBarlowTwins without importing modules/__init__.py."""
    project_root = Path(__file__).resolve().parent
    modules_dir = project_root / "modules"
    package_name = "modules"

    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(modules_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    def _load_submodule(name: str, file_path: Path) -> None:
        if name in sys.modules:
            return
        spec = importlib.util.spec_from_file_location(name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to create spec for {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    _load_submodule("modules.common_modules", modules_dir / "common_modules.py")
    _load_submodule("modules.transformer_modules", modules_dir / "transformer_modules.py")
    _load_submodule("modules.actor_critic", modules_dir / "actor_critic.py")
    actor_module = sys.modules["modules.actor_critic"]
    return getattr(actor_module, "ActorCriticBarlowTwins")


def _class_attrs_to_dict(cls: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key, value in cls.__dict__.items():
        if key.startswith("__"):
            continue
        if callable(value):
            continue
        data[key] = value
    return data


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _latency_stats_ms(latencies_ms: List[float]) -> Dict[str, Optional[float]]:
    if not latencies_ms:
        return {
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p99_ms": None,
            "std_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return None
    return float(np.dot(a_flat, b_flat) / denom)


def _accuracy_metrics(reference: Optional[np.ndarray], candidate: Optional[np.ndarray]) -> Dict[str, Any]:
    if reference is None or candidate is None:
        return {"available": False, "reason": "missing_reference_or_candidate"}
    if reference.shape != candidate.shape:
        return {
            "available": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    diff = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "available": True,
        "max_abs_error": float(np.max(diff)),
        "mean_abs_error": float(np.mean(diff)),
        "cosine_similarity": _cosine_similarity(reference, candidate),
    }


def _parse_device_index(device: str) -> int:
    if ":" in device:
        _, idx = device.split(":", 1)
        try:
            return int(idx)
        except ValueError:
            return 0
    return 0


class NvidiaSmiSampler:
    """Poll nvidia-smi for util/power/memory during benchmark windows."""

    def __init__(self, gpu_index: int, interval_ms: int) -> None:
        self.gpu_index = gpu_index
        self.interval_s = max(interval_ms, 20) / 1000.0
        self.samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    first = result.stdout.strip().splitlines()[0]
                    values = [v.strip() for v in first.split(",")]
                    if len(values) == 3:
                        util = _safe_float(values[0])
                        mem = _safe_float(values[1])
                        power = _safe_float(values[2])
                        if util is not None and mem is not None and power is not None:
                            self.samples.append(
                                {
                                    "timestamp_s": time.time(),
                                    "gpu_util_percent": util,
                                    "memory_used_mb": mem,
                                    "power_w": power,
                                }
                            )
            except Exception:
                # Keep benchmark robust even if nvidia-smi sampling fails intermittently.
                pass
            time.sleep(self.interval_s)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def summary(self) -> Dict[str, Optional[float]]:
        if not self.samples:
            return {
                "samples": 0,
                "gpu_util_mean": None,
                "gpu_util_peak": None,
                "memory_used_mean_mb": None,
                "memory_used_peak_mb": None,
                "power_mean_w": None,
                "power_peak_w": None,
            }
        util = [s["gpu_util_percent"] for s in self.samples]
        mem = [s["memory_used_mb"] for s in self.samples]
        power = [s["power_w"] for s in self.samples]
        return {
            "samples": len(self.samples),
            "gpu_util_mean": float(np.mean(util)),
            "gpu_util_peak": float(np.max(util)),
            "memory_used_mean_mb": float(np.mean(mem)),
            "memory_used_peak_mb": float(np.max(mem)),
            "power_mean_w": float(np.mean(power)),
            "power_peak_w": float(np.max(power)),
        }


@dataclass
class InputPack:
    obs_prop_torch: torch.Tensor
    obs_hist_torch: torch.Tensor
    obs_prop_numpy: np.ndarray
    obs_hist_numpy: np.ndarray


class InProcessBackend:
    """Backend with callable inference in current Python process."""

    def __init__(self, name: str, device: str, func, torch_memory_stats: bool = False):
        self.name = name
        self.device = device
        self.func = func
        self.torch_memory_stats = torch_memory_stats

    def infer_once(self, inputs: InputPack) -> np.ndarray:
        output = self.func(inputs)
        if isinstance(output, torch.Tensor):
            return output.detach().cpu().numpy()
        return np.asarray(output)

    def benchmark(
        self,
        inputs: InputPack,
        warmup: int,
        iters: int,
        batch_size: int,
        cuda_sync: bool,
        sampler: Optional[NvidiaSmiSampler],
        csv_rows: List[Dict[str, Any]],
        repeat_idx: int,
    ) -> Dict[str, Any]:
        if self.torch_memory_stats and torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        for _ in range(warmup):
            _ = self.func(inputs)
            if cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(self.device))

        latencies_ms: List[float] = []
        torch_allocated: List[float] = []
        torch_reserved: List[float] = []
        started = time.perf_counter()
        if sampler is not None:
            sampler.start()
        for step in range(iters):
            t0 = time.perf_counter()
            _ = self.func(inputs)
            if cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(self.device))
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            latencies_ms.append(ms)
            csv_rows.append(
                {
                    "backend": self.name,
                    "repeat": repeat_idx,
                    "iter": step,
                    "latency_ms": ms,
                    "source": "in_process",
                }
            )
            if self.torch_memory_stats and torch.cuda.is_available() and self.device.startswith("cuda"):
                dev = torch.device(self.device)
                torch_allocated.append(torch.cuda.memory_allocated(dev) / (1024 ** 2))
                torch_reserved.append(torch.cuda.memory_reserved(dev) / (1024 ** 2))
        ended = time.perf_counter()
        if sampler is not None:
            sampler.stop()

        duration_s = max(ended - started, 1e-9)
        stats = _latency_stats_ms(latencies_ms)
        repeat_result: Dict[str, Any] = {
            "duration_s": duration_s,
            "throughput_qps": float(batch_size * iters / duration_s),
            "latency": stats,
            "resource": sampler.summary() if sampler is not None else None,
        }
        if self.torch_memory_stats:
            repeat_result["torch_memory_mb"] = {
                "allocated_peak_mb": float(torch.cuda.max_memory_allocated(torch.device(self.device)) / (1024 ** 2))
                if (torch.cuda.is_available() and self.device.startswith("cuda"))
                else None,
                "reserved_peak_mb": float(torch.cuda.max_memory_reserved(torch.device(self.device)) / (1024 ** 2))
                if (torch.cuda.is_available() and self.device.startswith("cuda"))
                else None,
                "allocated_mean_mb": float(np.mean(torch_allocated)) if torch_allocated else None,
                "reserved_mean_mb": float(np.mean(torch_reserved)) if torch_reserved else None,
            }
        return repeat_result


class TrtexecBackend:
    """TensorRT benchmark by parsing trtexec output."""

    def __init__(self, name: str, engine_path: str, trtexec_path: str):
        self.name = name
        self.engine_path = engine_path
        self.trtexec_path = trtexec_path

    @staticmethod
    def _parse_duration_to_ms(value: str, unit: str) -> Optional[float]:
        number = _safe_float(value)
        if number is None:
            return None
        unit = unit.lower()
        if unit == "ms":
            return number
        if unit == "us":
            return number / 1000.0
        if unit == "s":
            return number * 1000.0
        return None

    @classmethod
    def _parse_stats_line(cls, output: str, prefix: str) -> Dict[str, Optional[float]]:
        line = None
        for raw in output.splitlines():
            if prefix in raw:
                line = raw
        if line is None:
            return {
                "mean_ms": None,
                "p50_ms": None,
                "p90_ms": None,
                "p99_ms": None,
                "std_ms": None,
                "min_ms": None,
                "max_ms": None,
            }

        def _extract(label: str) -> Optional[float]:
            match = re.search(rf"{label}\s*=\s*([0-9]*\.?[0-9]+)\s*(us|ms|s)", line)
            if not match:
                return None
            return cls._parse_duration_to_ms(match.group(1), match.group(2))

        return {
            "mean_ms": _extract("mean"),
            "p50_ms": _extract("median"),
            "p90_ms": None,
            "p99_ms": _extract(r"percentile\(99%\)"),
            "std_ms": None,
            "min_ms": _extract("min"),
            "max_ms": _extract("max"),
        }

    @classmethod
    def _parse_latency_line(cls, output: str) -> Dict[str, Optional[float]]:
        return cls._parse_stats_line(output, "Latency:")

    @classmethod
    def _parse_gpu_compute_line(cls, output: str) -> Dict[str, Optional[float]]:
        return cls._parse_stats_line(output, "GPU Compute Time:")

    @staticmethod
    def _parse_throughput(output: str) -> Optional[float]:
        match = re.search(r"Throughput:\s*([0-9]*\.?[0-9]+)\s*qps", output)
        return _safe_float(match.group(1)) if match else None

    def benchmark(
        self,
        warmup: int,
        iters: int,
        sampler: Optional[NvidiaSmiSampler],
        device_index: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        cmd = [
            self.trtexec_path,
            f"--loadEngine={self.engine_path}",
            f"--warmUp={warmup}",
            f"--iterations={iters}",
            "--duration=0",
            "--noDataTransfers",
            "--useCudaGraph",
            f"--device={device_index}",
        ]
        started = time.perf_counter()
        if sampler is not None:
            sampler.start()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ended = time.perf_counter()
        if sampler is not None:
            sampler.stop()
        combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        latency = self._parse_latency_line(combined_output)
        if latency.get("mean_ms") in (None, 0.0):
            gpu_compute = self._parse_gpu_compute_line(combined_output)
            if gpu_compute.get("mean_ms") not in (None, 0.0):
                latency = gpu_compute
        throughput = self._parse_throughput(combined_output)
        if latency.get("mean_ms") in (None, 0.0) and throughput not in (None, 0.0):
            implied_ms = float(batch_size * 1000.0 / throughput)
            latency["mean_ms"] = implied_ms
            latency["p50_ms"] = implied_ms if latency.get("p50_ms") in (None, 0.0) else latency.get("p50_ms")
            latency["p99_ms"] = implied_ms if latency.get("p99_ms") in (None, 0.0) else latency.get("p99_ms")
            latency["min_ms"] = implied_ms if latency.get("min_ms") in (None, 0.0) else latency.get("min_ms")
            latency["max_ms"] = implied_ms if latency.get("max_ms") in (None, 0.0) else latency.get("max_ms")

        repeat_result = {
            "duration_s": ended - started,
            "throughput_qps": throughput,
            "latency": latency,
            "resource": sampler.summary() if sampler is not None else None,
            "raw_exit_code": proc.returncode,
        }
        if repeat_result["throughput_qps"] is None and repeat_result["latency"]["mean_ms"] not in (None, 0.0):
            mean_ms = repeat_result["latency"]["mean_ms"]
            repeat_result["throughput_qps"] = float(batch_size * 1000.0 / mean_ms)
        if proc.returncode != 0:
            repeat_result["error"] = combined_output[-4000:]
        return repeat_result


def _aggregate_repeat_metrics(repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(path: Tuple[str, ...]) -> List[float]:
        values: List[float] = []
        for item in repeats:
            cur: Any = item
            ok = True
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    ok = False
                    break
                cur = cur[key]
            if ok:
                val = _safe_float(cur)
                if val is not None and not math.isnan(val):
                    values.append(val)
        return values

    summary: Dict[str, Any] = {"repeats": len(repeats)}
    fields = {
        "mean_latency_ms": ("latency", "mean_ms"),
        "p50_latency_ms": ("latency", "p50_ms"),
        "p90_latency_ms": ("latency", "p90_ms"),
        "p99_latency_ms": ("latency", "p99_ms"),
        "std_latency_ms": ("latency", "std_ms"),
        "throughput_qps": ("throughput_qps",),
        "gpu_util_mean": ("resource", "gpu_util_mean"),
        "gpu_util_peak": ("resource", "gpu_util_peak"),
        "memory_used_mean_mb": ("resource", "memory_used_mean_mb"),
        "memory_used_peak_mb": ("resource", "memory_used_peak_mb"),
        "power_mean_w": ("resource", "power_mean_w"),
        "power_peak_w": ("resource", "power_peak_w"),
        "torch_allocated_peak_mb": ("torch_memory_mb", "allocated_peak_mb"),
        "torch_reserved_peak_mb": ("torch_memory_mb", "reserved_peak_mb"),
    }
    for out_name, path in fields.items():
        vals = collect(path)
        summary[out_name] = float(np.mean(vals)) if vals else None
    return summary


def _build_inputs(batch_size: int, num_prop: int, num_hist: int, seed: int, device: str) -> InputPack:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    obs_prop_cpu = torch.randn(batch_size, num_prop, generator=gen, dtype=torch.float32)
    obs_hist_cpu = torch.randn(batch_size, num_hist, num_prop, generator=gen, dtype=torch.float32)
    obs_prop_torch = obs_prop_cpu.to(device)
    obs_hist_torch = obs_hist_cpu.to(device)
    return InputPack(
        obs_prop_torch=obs_prop_torch,
        obs_hist_torch=obs_hist_torch,
        obs_prop_numpy=obs_prop_cpu.numpy(),
        obs_hist_numpy=obs_hist_cpu.numpy(),
    )


def _resolve_trtexec() -> Optional[str]:
    by_path = shutil.which("trtexec")
    if by_path:
        return by_path
    default = "/usr/src/tensorrt/bin/trtexec"
    return default if os.path.exists(default) else None


def _collect_meta(device: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device": device,
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        dev = torch.device(device)
        idx = dev.index or 0
        meta["gpu_name"] = torch.cuda.get_device_name(idx)
        meta["gpu_capability"] = list(torch.cuda.get_device_capability(idx))
        cmd = [
            "nvidia-smi",
            "-i",
            str(idx),
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                meta["driver_version"] = result.stdout.strip()
        except Exception:
            pass
    return meta


def _maybe_load_checkpoint_backend(
    ckpt_path: str,
    device: str,
    num_prop: int,
    num_scan: int,
    num_obs: int,
    num_priv_latent: int,
    num_hist: int,
    num_actions: int,
) -> InProcessBackend:
    policy_kwargs = _class_attrs_to_dict(TitaConstraintRoughCfgPPO.policy)
    actor_critic_cls = _load_actor_critic_barlow_twins()
    model = actor_critic_cls(
        num_prop,
        num_scan,
        num_obs,
        num_priv_latent,
        num_hist,
        num_actions,
        **policy_kwargs,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return InProcessBackend(
        name="pytorch_checkpoint",
        device=device,
        func=lambda inputs: model.actor_teacher_backbone(inputs.obs_prop_torch, inputs.obs_hist_torch),
        torch_memory_stats=True,
    )


def _maybe_load_torchscript_backend(jit_path: str, device: str) -> InProcessBackend:
    model = torch.jit.load(jit_path, map_location=device)
    model.eval()
    return InProcessBackend(
        name="torchscript_jit",
        device=device,
        func=lambda inputs: model(inputs.obs_prop_torch, inputs.obs_hist_torch),
        torch_memory_stats=True,
    )


def _maybe_load_onnx_backend(onnx_path: str) -> Tuple[InProcessBackend, Dict[str, Any]]:
    import onnxruntime as ort  # lazy import

    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_names = [i.name for i in session.get_inputs()]
    if len(input_names) < 2:
        raise RuntimeError(f"ONNX model expects >=2 inputs, got {len(input_names)}")

    def _run(inputs: InputPack) -> np.ndarray:
        outputs = session.run(
            None,
            {
                input_names[0]: inputs.obs_prop_numpy.astype(np.float32),
                input_names[1]: inputs.obs_hist_numpy.astype(np.float32),
            },
        )
        return np.asarray(outputs[0])

    backend = InProcessBackend(name="onnxruntime", device="cpu", func=_run, torch_memory_stats=False)
    info = {
        "available_providers": available,
        "requested_providers": providers,
        "active_providers": session.get_providers(),
    }
    return backend, info


def _run_backend_repeats(
    runner,
    repeats: int,
    sampler_enabled: bool,
    gpu_index: int,
    nvidia_interval_ms: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for repeat_idx in range(repeats):
        sampler = NvidiaSmiSampler(gpu_index=gpu_index, interval_ms=nvidia_interval_ms) if sampler_enabled else None
        results.append(runner(sampler, repeat_idx))
    return results


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = ["backend", "repeat", "iter", "latency_ms", "source"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def _write_markdown(path: str, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Inference Benchmark Report")
    lines.append("")
    lines.append(f"- 时间: `{payload['metadata'].get('timestamp')}`")
    lines.append(f"- 设备: `{payload['metadata'].get('device')}`")
    lines.append(f"- GPU: `{payload['metadata'].get('gpu_name', 'N/A')}`")
    lines.append(f"- 批大小: `{payload['config']['batch_size']}`")
    lines.append(f"- warmup/iters/repeats: `{payload['config']['warmup']}/{payload['config']['iters']}/{payload['config']['repeats']}`")
    lines.append("")
    baseline = payload.get("baseline_backend", "pytorch_checkpoint")
    baseline_latency = (
        payload["backends"].get(baseline, {}).get("aggregate", {}).get("mean_latency_ms")
        if payload["backends"].get(baseline)
        else None
    )
    lines.append("## Backend Summary")
    lines.append("")
    for backend_name, item in payload["backends"].items():
        lines.append(f"### `{backend_name}`")
        if item.get("status") != "ok":
            lines.append(f"- 状态: `{item.get('status')}`")
            lines.append(f"- 原因: `{item.get('reason', 'unknown')}`")
            lines.append("")
            continue
        aggr = item.get("aggregate", {})
        mean_latency = aggr.get("mean_latency_ms")
        throughput = aggr.get("throughput_qps")
        speedup = None
        if baseline_latency and mean_latency:
            speedup = baseline_latency / mean_latency
        lines.append(f"- 平均时延(ms): `{mean_latency}`")
        lines.append(f"- p50/p90/p99(ms): `{aggr.get('p50_latency_ms')}` / `{aggr.get('p90_latency_ms')}` / `{aggr.get('p99_latency_ms')}`")
        lines.append(f"- 吞吐(qps): `{throughput}`")
        lines.append(f"- 相对 `{baseline}` 加速比: `{speedup}`")
        lines.append(f"- GPU利用率均值/峰值(%): `{aggr.get('gpu_util_mean')}` / `{aggr.get('gpu_util_peak')}`")
        lines.append(f"- 显存占用均值/峰值(MB): `{aggr.get('memory_used_mean_mb')}` / `{aggr.get('memory_used_peak_mb')}`")
        lines.append(f"- 功耗均值/峰值(W): `{aggr.get('power_mean_w')}` / `{aggr.get('power_peak_w')}`")
        acc = item.get("accuracy_vs_baseline", {})
        if acc.get("available"):
            lines.append(
                "- 精度误差(max/mean abs, cosine): "
                f"`{acc.get('max_abs_error')}` / `{acc.get('mean_abs_error')}` / `{acc.get('cosine_similarity')}`"
            )
        else:
            lines.append(f"- 精度误差: `不可用 ({acc.get('reason', 'unknown')})`")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark across PT/JIT/ONNX/TRT")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="model_stairs_10000.pt")
    parser.add_argument("--torchscript", default="model.pt")
    parser.add_argument("--onnx", default="test.onnx")
    parser.add_argument("--engine-fp32", default="test_fp32.engine")
    parser.add_argument("--engine-fp16", default="test_fp16.engine")
    parser.add_argument("--nvidia-smi-interval-ms", type=int, default=100)
    parser.add_argument("--output-json", default="benchmark_results.json")
    parser.add_argument("--output-md", default="benchmark_results.md")
    parser.add_argument("--output-csv", default="benchmark_samples.csv")
    parser.add_argument(
        "--disable-resource-sampling",
        action="store_true",
        help="Disable nvidia-smi GPU util/memory/power sampling",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    env_cfg = TitaConstraintRoughCfg.env
    num_prop = int(env_cfg.n_proprio)
    num_hist = int(env_cfg.history_len)
    num_scan = int(env_cfg.n_scan)
    num_priv_latent = int(env_cfg.n_priv_latent)
    num_obs = int(env_cfg.num_observations)
    num_actions = int(env_cfg.num_actions)

    inputs = _build_inputs(
        batch_size=args.batch_size,
        num_prop=num_prop,
        num_hist=num_hist,
        seed=args.seed,
        device=args.device,
    )
    gpu_index = _parse_device_index(args.device)
    sampler_enabled = (not args.disable_resource_sampling) and args.device.startswith("cuda")

    payload: Dict[str, Any] = {
        "metadata": _collect_meta(args.device),
        "config": {
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
            "device": args.device,
            "nvidia_smi_interval_ms": args.nvidia_smi_interval_ms,
            "resource_sampling_window": "measured_iterations_only",
        },
        "input_spec": {
            "obs_prop_shape": [args.batch_size, num_prop],
            "obs_hist_shape": [args.batch_size, num_hist, num_prop],
            "num_actions": num_actions,
            "num_obs": num_obs,
        },
        "baseline_backend": "pytorch_checkpoint",
        "backends": {},
    }
    csv_rows: List[Dict[str, Any]] = []
    baseline_output: Optional[np.ndarray] = None

    # PyTorch checkpoint backend
    if os.path.exists(args.checkpoint):
        try:
            backend = _maybe_load_checkpoint_backend(
                args.checkpoint,
                args.device,
                num_prop=num_prop,
                num_scan=num_scan,
                num_obs=num_obs,
                num_priv_latent=num_priv_latent,
                num_hist=num_hist,
                num_actions=num_actions,
            )
            baseline_output = backend.infer_once(inputs)
            repeats = _run_backend_repeats(
                runner=lambda sampler, repeat_idx: backend.benchmark(
                    inputs=inputs,
                    warmup=args.warmup,
                    iters=args.iters,
                    batch_size=args.batch_size,
                    cuda_sync=args.device.startswith("cuda"),
                    sampler=sampler,
                    csv_rows=csv_rows,
                    repeat_idx=repeat_idx,
                ),
                repeats=args.repeats,
                sampler_enabled=sampler_enabled,
                gpu_index=gpu_index,
                nvidia_interval_ms=args.nvidia_smi_interval_ms,
            )
            payload["backends"][backend.name] = {
                "status": "ok",
                "repeats": repeats,
                "aggregate": _aggregate_repeat_metrics(repeats),
            }
            payload["backends"][backend.name]["accuracy_vs_baseline"] = _accuracy_metrics(baseline_output, baseline_output)
        except Exception as exc:  # noqa: BLE001
            payload["backends"]["pytorch_checkpoint"] = {"status": "error", "reason": str(exc)}
    else:
        payload["backends"]["pytorch_checkpoint"] = {"status": "skipped", "reason": f"missing_file:{args.checkpoint}"}

    # TorchScript backend
    if os.path.exists(args.torchscript):
        try:
            backend = _maybe_load_torchscript_backend(args.torchscript, args.device)
            candidate = backend.infer_once(inputs)
            repeats = _run_backend_repeats(
                runner=lambda sampler, repeat_idx: backend.benchmark(
                    inputs=inputs,
                    warmup=args.warmup,
                    iters=args.iters,
                    batch_size=args.batch_size,
                    cuda_sync=args.device.startswith("cuda"),
                    sampler=sampler,
                    csv_rows=csv_rows,
                    repeat_idx=repeat_idx,
                ),
                repeats=args.repeats,
                sampler_enabled=sampler_enabled,
                gpu_index=gpu_index,
                nvidia_interval_ms=args.nvidia_smi_interval_ms,
            )
            payload["backends"][backend.name] = {
                "status": "ok",
                "repeats": repeats,
                "aggregate": _aggregate_repeat_metrics(repeats),
                "accuracy_vs_baseline": _accuracy_metrics(baseline_output, candidate),
            }
        except Exception as exc:  # noqa: BLE001
            payload["backends"]["torchscript_jit"] = {"status": "error", "reason": str(exc)}
    else:
        payload["backends"]["torchscript_jit"] = {"status": "skipped", "reason": f"missing_file:{args.torchscript}"}

    # ONNX Runtime backend
    if os.path.exists(args.onnx):
        try:
            backend, onnx_info = _maybe_load_onnx_backend(args.onnx)
            candidate = backend.infer_once(inputs)
            repeats = _run_backend_repeats(
                runner=lambda sampler, repeat_idx: backend.benchmark(
                    inputs=inputs,
                    warmup=args.warmup,
                    iters=args.iters,
                    batch_size=args.batch_size,
                    cuda_sync=False,
                    sampler=sampler,
                    csv_rows=csv_rows,
                    repeat_idx=repeat_idx,
                ),
                repeats=args.repeats,
                sampler_enabled=sampler_enabled,
                gpu_index=gpu_index,
                nvidia_interval_ms=args.nvidia_smi_interval_ms,
            )
            payload["backends"][backend.name] = {
                "status": "ok",
                "repeats": repeats,
                "aggregate": _aggregate_repeat_metrics(repeats),
                "accuracy_vs_baseline": _accuracy_metrics(baseline_output, candidate),
                "notes": onnx_info,
            }
        except Exception as exc:  # noqa: BLE001
            payload["backends"]["onnxruntime"] = {"status": "error", "reason": str(exc)}
    else:
        payload["backends"]["onnxruntime"] = {"status": "skipped", "reason": f"missing_file:{args.onnx}"}

    # TensorRT via trtexec fallback.
    trtexec_path = _resolve_trtexec()
    for name, engine_path in [("tensorrt_fp32", args.engine_fp32), ("tensorrt_fp16", args.engine_fp16)]:
        if not os.path.exists(engine_path):
            payload["backends"][name] = {"status": "skipped", "reason": f"missing_file:{engine_path}"}
            continue
        if trtexec_path is None:
            payload["backends"][name] = {"status": "error", "reason": "trtexec_not_found"}
            continue
        try:
            backend = TrtexecBackend(name=name, engine_path=engine_path, trtexec_path=trtexec_path)
            repeats = _run_backend_repeats(
                runner=lambda sampler, _repeat_idx: backend.benchmark(
                    warmup=args.warmup,
                    iters=args.iters,
                    sampler=sampler,
                    device_index=gpu_index,
                    batch_size=args.batch_size,
                ),
                repeats=args.repeats,
                sampler_enabled=sampler_enabled,
                gpu_index=gpu_index,
                nvidia_interval_ms=args.nvidia_smi_interval_ms,
            )
            payload["backends"][name] = {
                "status": "ok",
                "repeats": repeats,
                "aggregate": _aggregate_repeat_metrics(repeats),
                "accuracy_vs_baseline": {
                    "available": False,
                    "reason": "trtexec_mode_no_output_tensor",
                },
                "notes": "TensorRT benchmark uses trtexec --loadEngine fallback.",
            }
        except Exception as exc:  # noqa: BLE001
            payload["backends"][name] = {"status": "error", "reason": str(exc)}

    _write_json(args.output_json, payload)
    _write_markdown(args.output_md, payload)
    if args.output_csv:
        _write_csv(args.output_csv, csv_rows)

    print(f"[done] json: {args.output_json}")
    print(f"[done] markdown: {args.output_md}")
    if args.output_csv:
        print(f"[done] csv: {args.output_csv}")


if __name__ == "__main__":
    main()
