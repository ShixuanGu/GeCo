#!/usr/bin/env python3
"""
Backend contract checker.

Builds every requested geometry / flow backend, runs it on a few frames of an
example video and verifies the output contract (``GeometryOutput.validate`` /
``FlowOutput.validate``: shapes, dtypes, valid masks, camera conventions).  The
outputs are then cross-fed into ``geco.metric.compute_pair_errors`` so that a
backend which *looks* right but does not fit the metric (wrong flow units, flipped
pose convention, ...) shows up as an implausible error statistic.  Timing and
peak GPU memory are reported per backend and everything is written to a JSON
report.

Usage (repo root, GPU node)::

    python scripts/check_backends.py --geometry vggt,vggt_omega --flow ufm,sea_raft,raft \\
        --report reports/check.json [--strict]

Flow entries take the form ``backend[:model[:ckpt]]``, e.g.
``ptlflow:dpflow:sintel`` or ``sea_raft:sintel-M:/path/to/sintel-M.pth``;
geometry entries ``backend[:ckpt]``.  ``--geometry_ckpt`` applies when a single
geometry backend is listed.  Without ``--strict`` a failing backend is recorded
(traceback in the report) and the remaining checks continue; with ``--strict``
the exit status is non-zero as soon as any backend failed.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

import geco  # noqa: E402,F401  (puts the vendored third-party packages on sys.path)
from geco.cli import default_device, describe_backend  # noqa: E402
from geco.flow import build_flow, list_flow_backends  # noqa: E402
from geco.flow.base import FlowOutput, make_flow_output  # noqa: E402
from geco.geometry import build_geometry, list_geometry_backends  # noqa: E402
from geco.geometry.base import GeometryOutput  # noqa: E402
from geco.image_io import check_same_size, list_frame_files, load_image_rgb  # noqa: E402
from geco.metric import compute_pair_errors, source_confidence_mask  # noqa: E402

DEFAULT_FRAMES = "examples/deform_house"
DEFAULT_GEOMETRY = "vggt,vggt_omega"
DEFAULT_FLOW = "ufm,sea_raft,raft"
# confidence-mask settings of the scripts' defaults
CONF_PERCENTILE = 20.0
CONF_MIN = 0.2
QUANTILE_LIMIT = 16_000_000  # torch.quantile input-size limit


# --------------------------------------------------------------------------- small helpers
def parse_spec(entry: str, max_parts: int) -> List[Optional[str]]:
    """Split ``a:b:c`` into at most ``max_parts`` fields, padding with ``None`` (empty fields -> ``None``)."""
    parts = entry.strip().split(":")
    if len(parts) > max_parts or not parts[0]:
        raise ValueError(f"malformed backend entry {entry!r}")
    fields: List[Optional[str]] = [p if p else None for p in parts]
    return fields + [None] * (max_parts - len(fields))


def finite(x: float) -> Optional[float]:
    """JSON-safe float (``None`` for NaN / inf)."""
    x = float(x)
    return x if math.isfinite(x) else None


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**30


def fmt_gb(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f} GB"


class Stopwatch:
    """``with Stopwatch(device) as sw: ...; sw.seconds`` -- synchronises CUDA before reading the clock."""

    def __init__(self, device: torch.device):
        self.device = device
        self.seconds = float("nan")

    def __enter__(self) -> "Stopwatch":
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        sync(self.device)
        self.seconds = time.perf_counter() - self._t0


def release(obj: Any, device: torch.device) -> None:
    """Drop a backend and give its memory back to the allocator."""
    del obj
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def param_count_m(backend: Any) -> Optional[float]:
    model = getattr(backend, "model", None)
    if not isinstance(model, torch.nn.Module):
        return None
    return sum(p.numel() for p in model.parameters()) / 1e6


def quantiles(values: torch.Tensor, qs: Tuple[float, ...]) -> List[float]:
    """Quantiles of a 1-D tensor (strided sub-sample above torch.quantile's size limit)."""
    v = values.flatten().float()
    if v.numel() == 0:
        return [float("nan")] * len(qs)
    if v.numel() > QUANTILE_LIMIT:
        v = v[:: v.numel() // QUANTILE_LIMIT + 1]
    return torch.quantile(v, torch.tensor(qs, device=v.device)).tolist()


def min_median_max(values: torch.Tensor) -> Dict[str, Optional[float]]:
    if values.numel() == 0:
        return {"min": None, "median": None, "max": None}
    med = quantiles(values, (0.5,))[0]
    return {"min": finite(values.min()), "median": finite(med), "max": finite(values.max())}


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    return finite(x[mask].mean()) if bool(mask.any()) else None


def fraction(mask: torch.Tensor) -> float:
    return float(mask.float().mean())


def zero_flow(hw: Tuple[int, int], device: torch.device) -> FlowOutput:
    """Dummy flow: no motion, everything covisible -- isolates the geometry side of the metric."""
    H, W = hw
    return make_flow_output(torch.zeros(1, 2, H, W, device=device), torch.ones(1, H, W, device=device)).validate(hw)


def metric_summary(geo: GeometryOutput, tgt: int, flow_out: FlowOutput) -> Dict[str, Any]:
    """Feed a (geometry, flow) pair (0 -> tgt) through the metric and summarise the masks / errors."""
    conf_src = source_confidence_mask(geo, 0, CONF_PERCENTILE, CONF_MIN)
    pe = compute_pair_errors(geo, 0, tgt, flow_out, conf_src, depth_check="bidirectional")
    return {
        "pair": [0, tgt],
        "conf_mask_fraction": fraction(conf_src),
        "mot_valid_fraction": fraction(pe.mot_valid),
        "dep_mask_fraction": fraction(pe.dep_mask),
        "fus_mask_fraction": fraction(pe.fus_mask),
        "wrong_occ_fraction": fraction(pe.wrong_occ),
        "mean_e_xy": masked_mean(pe.e_xy, pe.mot_valid),
        "mean_e_z": masked_mean(pe.e_z, pe.dep_mask),
        "mean_fused": masked_mean(pe.p_fus, pe.fus_mask),
    }


# --------------------------------------------------------------------------- geometry check
def check_geometry(
    spec: str,
    ckpt_override: Optional[str],
    files: List[str],
    hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[Dict[str, Any], Optional[GeometryOutput]]:
    name, ckpt = parse_spec(spec, 2)
    if ckpt is None:
        ckpt = ckpt_override
    tag = f"[geometry/{spec}]"
    rep: Dict[str, Any] = {"spec": spec, "name": name, "ckpt": ckpt, "status": "error"}
    geo: Optional[GeometryOutput] = None
    backend = None
    try:
        reset_peak(device)
        with Stopwatch(device) as sw:
            backend = build_geometry(name, device=device, ckpt=ckpt)
        rep.update(
            build_s=sw.seconds,
            description=describe_backend(backend, ckpt=ckpt),
            supports_grad=bool(backend.supports_grad),
            params_M=param_count_m(backend),
        )
        print(f"{tag} built in {sw.seconds:.1f} s :: {rep['description']}")

        with Stopwatch(device) as sw:
            geo = backend.run(files, output_hw=hw)
        geo.validate(image_hw=hw, num_frames=len(files))
        rep["run_s"] = sw.seconds
        rep["peak_mem_gb"] = peak_gb(device)

        Fn, (H, W) = geo.num_frames, geo.image_hw
        depth_valid = geo.depth[..., 0][geo.valid]
        conf_valid = geo.conf[..., 0][geo.valid]
        K0 = geo.intrinsic[0]
        t1 = geo.extrinsic[1, :, 3]
        rep.update(
            num_frames=Fn,
            image_hw=[H, W],
            model_hw=list(geo.model_hw),
            depth=min_median_max(depth_valid),
            conf=min_median_max(conf_valid),
            valid_fraction=fraction(geo.valid),
            K0={"fx": float(K0[0, 0]), "fy": float(K0[1, 1]), "cx": float(K0[0, 2]), "cy": float(K0[1, 2])},
            t1_norm=float(torch.linalg.vector_norm(t1)),
            extras=sorted(geo.extras.keys()),
        )
        print(
            f"{tag} F={Fn} H={H} W={W} model_hw={geo.model_hw} valid={rep['valid_fraction']:.3f} "
            f"run={sw.seconds:.2f} s peak_mem={fmt_gb(rep['peak_mem_gb'])}"
        )
        d, c = rep["depth"], rep["conf"]
        print(
            f"{tag} depth min/median/max = {d['min']:.4g}/{d['median']:.4g}/{d['max']:.4g} | "
            f"conf min/median/max = {c['min']:.4g}/{c['median']:.4g}/{c['max']:.4g}"
        )
        print(
            f"{tag} K0 fx={rep['K0']['fx']:.2f} fy={rep['K0']['fy']:.2f} cx={rep['K0']['cx']:.2f} cy={rep['K0']['cy']:.2f} | "
            f"|t1|={rep['t1_norm']:.4f}"
        )

        # metric cross-check: zero flow, everything covisible -> tests the geometry side only
        ms = metric_summary(geo, 1, zero_flow(hw, device))
        rep["metric_zero_flow"] = ms
        print(
            f"{tag} metric(zero flow, pair 0->1): mot_valid={ms['mot_valid_fraction']:.3f} "
            f"dep_mask={ms['dep_mask_fraction']:.3f} mean e_xy={ms['mean_e_xy']} mean e_z={ms['mean_e_z']}"
        )
        rep["status"] = "ok"
    except Exception:
        rep["error"] = traceback.format_exc()
        print(f"{tag} FAILED\n{rep['error']}", file=sys.stderr)
        geo = None
    finally:
        release(backend, device)
    return rep, geo


# --------------------------------------------------------------------------- flow check
def check_flow(
    spec: str,
    files: List[str],
    hw: Tuple[int, int],
    device: torch.device,
    geo: Optional[GeometryOutput],
    geo_name: Optional[str],
) -> Dict[str, Any]:
    name, model, ckpt = parse_spec(spec, 3)
    tag = f"[flow/{spec}]"
    rep: Dict[str, Any] = {"spec": spec, "name": name, "model": model, "ckpt": ckpt, "status": "error", "pairs": {}}
    backend = None
    try:
        reset_peak(device)
        with Stopwatch(device) as sw:
            backend = build_flow(name, device=device, ckpt=ckpt, model=model)
        rep.update(
            build_s=sw.seconds,
            description=describe_backend(backend, ckpt=ckpt),
            supports_grad=bool(backend.supports_grad),
            native_covisibility=bool(backend.native_covisibility),
            params_M=param_count_m(backend),
        )
        print(f"{tag} built in {sw.seconds:.1f} s :: {rep['description']}")

        src = load_image_rgb(files[0])
        targets = sorted({1, min(4, len(files) - 1)})
        for tgt in targets:
            key = f"0-{tgt}"
            with Stopwatch(device) as sw:
                fo = backend.predict(src, load_image_rgb(files[tgt]))
            fo.validate(image_hw=hw)
            mag = torch.linalg.vector_norm(fo.flow, dim=-1)[fo.valid]
            p50, p90, p99 = quantiles(mag, (0.5, 0.9, 0.99))
            cov = fo.covisibility[fo.valid]
            pair: Dict[str, Any] = {
                "predict_s": sw.seconds,
                "flow_mag": {"p50": finite(p50), "p90": finite(p90), "p99": finite(p99), "max": finite(mag.max()) if mag.numel() else None},
                "cov_mean": finite(cov.mean()) if cov.numel() else None,
                "cov_frac_gt_0.5": fraction(cov > 0.5) if cov.numel() else None,
                "valid_fraction": fraction(fo.valid),
                "extras": sorted(fo.extras.keys()),
            }
            print(
                f"{tag} pair {key}: |flow| p50/p90/p99/max = {p50:.2f}/{p90:.2f}/{p99:.2f}/{pair['flow_mag']['max']:.2f} px | "
                f"cov mean={pair['cov_mean']:.3f} frac>0.5={pair['cov_frac_gt_0.5']:.3f} | valid={pair['valid_fraction']:.3f} | "
                f"extras={pair['extras']} | {sw.seconds:.2f} s"
            )
            if geo is not None and tgt < geo.num_frames:
                ms = metric_summary(geo, tgt, fo)
                ms["geometry"] = geo_name
                pair["metric"] = ms
                print(
                    f"{tag} metric(vs {geo_name}, pair {key}): mean e_xy over mot_valid={ms['mean_e_xy']} "
                    f"(mot_valid={ms['mot_valid_fraction']:.3f}) mean e_z={ms['mean_e_z']} fused={ms['mean_fused']}"
                )
            rep["pairs"][key] = pair
        rep["peak_mem_gb"] = peak_gb(device)
        print(f"{tag} peak_mem={fmt_gb(rep['peak_mem_gb'])}")
        rep["status"] = "ok"
    except Exception:
        rep["error"] = traceback.format_exc()
        print(f"{tag} FAILED\n{rep['error']}", file=sys.stderr)
    finally:
        release(backend, device)
    return rep


# --------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check every geometry / flow backend against the GeCo contract.")
    p.add_argument("--frames", type=str, default=DEFAULT_FRAMES,
                   help="Directory of frames (relative paths are also tried from the repo root).")
    p.add_argument("--num_frames", type=int, default=8, help="Number of leading frames fed to the geometry backends.")
    p.add_argument("--geometry", type=str, default=DEFAULT_GEOMETRY,
                   help=f"Comma list of geometry backends, entries 'name[:ckpt]' (available: {list_geometry_backends()}); "
                        "empty string skips the geometry checks.")
    p.add_argument("--flow", type=str, default=DEFAULT_FLOW,
                   help=f"Comma list of flow backends, entries 'name[:model[:ckpt]]' (available: {list_flow_backends()}); "
                        "empty string skips the flow checks.")
    p.add_argument("--geometry_ckpt", type=str, default=None,
                   help="Checkpoint for the (single) listed geometry backend; use 'name:ckpt' entries for several.")
    p.add_argument("--device", type=str, default=default_device(), help="Torch device (default: cuda if available).")
    p.add_argument("--report", type=str, default=None, help="Path of the JSON report (default: no report).")
    p.add_argument("--strict", action="store_true", help="Exit with status 1 if any backend failed.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if args.num_frames < 2:
        raise ValueError("--num_frames must be >= 2")

    frames_dir = args.frames
    if not os.path.isdir(frames_dir):
        frames_dir = os.path.join(REPO_ROOT, args.frames)
    files = list_frame_files(frames_dir)[: args.num_frames]
    if len(files) < 2:
        raise ValueError(f"need at least 2 frames in {frames_dir}, found {len(files)}")
    hw = check_same_size(files)

    geometry_specs = [s for s in args.geometry.split(",") if s.strip()]
    flow_specs = [s for s in args.flow.split(",") if s.strip()]
    if args.geometry_ckpt is not None and len(geometry_specs) != 1:
        raise ValueError("--geometry_ckpt applies to a single geometry backend; use 'name:ckpt' entries for several")

    report: Dict[str, Any] = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames": frames_dir,
        "num_frames": len(files),
        "image_hw": list(hw),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "geometry": {},
        "flow": {},
    }

    def write_report() -> None:
        if args.report:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
            with open(args.report, "w") as f:
                json.dump(report, f, indent=2)

    print(f"[check] {len(files)} frames of {frames_dir} at {hw[0]}x{hw[1]} on {device} ({report['gpu'] or 'cpu'})")

    # the first successful geometry output is reused by every flow check (computed once)
    geo_ref: Optional[GeometryOutput] = None
    geo_ref_name: Optional[str] = None
    with torch.no_grad():
        for spec in geometry_specs:
            rep, geo = check_geometry(spec, args.geometry_ckpt, files, hw, device)
            report["geometry"][spec] = rep
            if geo is not None and geo_ref is None:
                geo_ref, geo_ref_name = geo, rep["name"]
            write_report()

        if flow_specs and geo_ref is None:
            print("[check] no geometry output available: flow checks skip the metric cross-check")
        for spec in flow_specs:
            report["flow"][spec] = check_flow(spec, files, hw, device, geo_ref, geo_ref_name)
            write_report()

    failed = [f"geometry/{k}" for k, v in report["geometry"].items() if v["status"] != "ok"]
    failed += [f"flow/{k}" for k, v in report["flow"].items() if v["status"] != "ok"]
    report["ok"] = not failed
    report["failed"] = failed
    write_report()

    n_total = len(report["geometry"]) + len(report["flow"])
    print(f"[check] {n_total - len(failed)}/{n_total} backends passed" + (f"; failed: {failed}" if failed else ""))
    if args.report:
        print(f"[check] report written to {args.report}")
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
