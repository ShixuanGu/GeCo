#!/usr/bin/env python3
"""CPU unit tests for the backend contracts and the shared metric (no model weights needed).

Run:  python tests/test_contracts.py        (plain asserts, no pytest required; pytest also works)
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import geco  # noqa: F401
from geco.flow.base import FlowOutput, image_to_bchw01, make_flow_output, rescale_flow, resize_to_max_side
from geco.flow.consistency import forward_backward_covisibility
from geco.geometry.base import GeometryOutput, PreparedImages, map_to_output_grid, resize_hw_to_multiple
from geco.metric import ErrorAccumulator, compute_pair_errors, source_confidence_mask


def _synthetic_geometry(Fn=2, H=48, W=64, tx=0.0, depth_val=2.0, device="cpu") -> GeometryOutput:
    K = torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]]).expand(Fn, 3, 3).clone()
    E = torch.zeros(Fn, 3, 4)
    E[:, :, :3] = torch.eye(3)
    E[1, 0, 3] = tx  # second camera translated along x (world->cam t)
    depth = torch.full((Fn, H, W, 1), depth_val)
    conf = torch.full((Fn, H, W, 1), 2.0)
    valid = torch.ones(Fn, H, W, dtype=torch.bool)
    return GeometryOutput(K, E, depth, conf, valid, model_hw=(H, W)).validate(image_hw=(H, W), num_frames=Fn)


def test_resize_hw_to_multiple():
    assert resize_hw_to_multiple((720, 1280), 518, 14) == (294, 518)
    assert resize_hw_to_multiple((1280, 720), 518, 14) == (518, 294)
    assert resize_hw_to_multiple((480, 480), 518, 14) == (518, 518)
    try:
        resize_hw_to_multiple((720, 1280), 512, 14)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_map_to_output_grid_no_crop():
    Fn, h, w, H, W = 2, 42, 56, 90, 120
    depth = torch.rand(Fn, h, w, 1) + 0.5
    conf = torch.rand(Fn, h, w, 1) + 1.0
    K = torch.tensor([[70.0, 0, w / 2], [0, 65.0, h / 2], [0, 0, 1]]).expand(Fn, 3, 3).clone()
    prep = PreparedImages(images=torch.zeros(Fn, 3, h, w), source_hw=(H, W), crop_xywh=None)
    K2, d2, c2, v2 = map_to_output_grid(depth, conf, K, prep, (H, W))
    assert d2.shape == (Fn, H, W, 1) and c2.shape == (Fn, H, W, 1) and v2.shape == (Fn, H, W)
    assert v2.all()
    ref = F.interpolate(depth.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    assert torch.allclose(d2, ref)
    assert math.isclose(K2[0, 0, 0].item(), 70.0 * W / w, rel_tol=1e-6)
    assert math.isclose(K2[0, 1, 1].item(), 65.0 * H / h, rel_tol=1e-6)
    assert math.isclose(K2[0, 0, 2].item(), (w / 2) * W / w, rel_tol=1e-6)
    assert math.isclose(K2[0, 1, 2].item(), (h / 2) * H / h, rel_tol=1e-6)


def test_map_to_output_grid_with_crop():
    Fn, h, w, H, W = 1, 40, 40, 90, 120
    depth = torch.ones(Fn, h, w, 1)
    conf = torch.ones(Fn, h, w, 1) * 1.5
    K = torch.tensor([[50.0, 0, w / 2], [0, 50.0, h / 2], [0, 0, 1]]).expand(Fn, 3, 3).clone()
    crop = (15, 0, 90, 90)  # centre square crop of the 120x90 source
    prep = PreparedImages(images=torch.zeros(Fn, 3, h, w), source_hw=(H, W), crop_xywh=crop)
    K2, d2, c2, v2 = map_to_output_grid(depth, conf, K, prep, (H, W))
    assert v2[:, :, 15:105].all() and not v2[:, :, :15].any() and not v2[:, :, 105:].any()
    assert (d2[:, :, 15:105] > 0).all() and (d2[:, :, :15] == 0).all()
    assert math.isclose(K2[0, 0, 2].item(), (w / 2) * (90 / w) + 15, rel_tol=1e-6)
    assert math.isclose(K2[0, 0, 0].item(), 50.0 * 90 / w, rel_tol=1e-6)
    GeometryOutput(K2, torch.cat([torch.eye(3), torch.zeros(3, 1)], 1)[None], d2, c2, v2, model_hw=(h, w)).validate((H, W), Fn)


def test_forward_backward_covisibility():
    H, W = 32, 48
    f = torch.zeros(1, 2, H, W)
    f[:, 0] = 3.0  # shift right by 3 px
    b = torch.zeros(1, 2, H, W)
    b[:, 0] = -3.0
    cov, err, inb = forward_backward_covisibility(f, b)
    assert cov.shape == (1, H, W) and err.shape == (1, H, W)
    assert torch.allclose(cov[:, :, : W - 3], torch.ones(1, H, W - 3))  # consistent -> 1
    assert (cov[:, :, W - 3 :] == 0).all()  # x+f leaves the image -> 0
    assert not inb[:, :, W - 3 :].any()
    b0 = torch.zeros_like(b)  # inconsistent backward flow
    cov0, err0, _ = forward_backward_covisibility(f, b0)
    assert (cov0 == 0).all() and torch.allclose(err0[:, :, : W - 3], torch.full((1, H, W - 3), 3.0))
    # threshold 0.5 <=> classic criterion |c|^2 = 0.01(|f|^2+|b|^2) + 0.5.
    # Perturb b perpendicularly: c = (0, d), |f|^2 = 9, |b|^2 = 9 + d^2  ->  boundary d^2 = 0.68 / 0.99
    d_star = math.sqrt(0.68 / 0.99)
    for factor, expect_visible in ((0.99, True), (1.01, False)):
        b1 = b.clone()
        b1[:, 1] = d_star * factor
        cov1, _, _ = forward_backward_covisibility(f, b1)
        inside = cov1[:, :, : W - 3]
        assert ((inside > 0.5).all() if expect_visible else (inside < 0.5).all()), (factor, inside.min(), inside.max())


def test_flow_helpers():
    img = (torch.rand(3, 20, 30) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    t = image_to_bchw01(img, torch.device("cpu"))
    assert t.shape == (1, 3, 20, 30) and t.dtype == torch.float32 and t.max() <= 1.0
    tf = image_to_bchw01(torch.rand(20, 30, 3, requires_grad=True), torch.device("cpu"))
    assert tf.requires_grad
    x, (h, w) = resize_to_max_side(torch.rand(1, 3, 720, 1280), 1024, 8)
    assert max(h, w) <= 1024 and h % 8 == 0 and w % 8 == 0 and x.shape == (1, 3, h, w)
    x2, hw2 = resize_to_max_side(torch.rand(1, 3, 240, 320), None, 8)
    assert hw2 == (240, 320)
    fl = torch.ones(1, 2, 24, 32)
    up = rescale_flow(fl, (48, 64))
    assert up.shape == (1, 2, 48, 64) and torch.allclose(up, torch.full_like(up, 2.0))
    out = make_flow_output(torch.zeros(1, 2, 24, 32), torch.ones(1, 24, 32)).validate((24, 32))
    assert out.valid.all()


def test_compute_pair_errors_static_and_translation():
    H, W = 48, 64
    geo = _synthetic_geometry(H=H, W=W, tx=0.0)
    flow = make_flow_output(torch.zeros(1, 2, H, W), torch.ones(1, H, W))
    conf_mask = source_confidence_mask(geo, 0, conf_percentile=20, conf_min=0.2)
    assert conf_mask.all()
    pe = compute_pair_errors(geo, 0, 1, flow, conf_mask, covis_thresh=0.5, tau_z=0.02, depth_check="bidirectional")
    assert pe.mot_valid.all() and pe.dep_mask.all() and pe.fus_mask.all()
    assert torch.allclose(pe.e_xy, torch.zeros(H, W), atol=1e-5) and torch.allclose(pe.e_z, torch.zeros(H, W), atol=1e-5)
    # camera 2 translated by tx=0.1 (world->cam t): ego flow du = -fx * tx / Z = -60*0.1/2 = -3 px (points move left)
    geo2 = _synthetic_geometry(H=H, W=W, tx=0.1)
    pe2 = compute_pair_errors(geo2, 0, 1, flow, conf_mask, depth_check="unidirectional")
    ego_u = pe2.ego_flow[..., 0][pe2.mask_rp]
    assert torch.allclose(ego_u, torch.full_like(ego_u, 3.0), atol=1e-4), ego_u[:3]
    # zero measured flow vs 3 px ego flow -> e_xy = 3/60 = 0.05
    assert torch.allclose(pe2.e_xy[pe2.mot_valid], torch.full((int(pe2.mot_valid.sum()),), 0.05), atol=1e-5)
    # pixels whose projection leaves the target image are invalid
    assert not pe2.mask_rp[:, W - 2 :].any() and pe2.mask_rp[:, : W - 4].all()
    acc = ErrorAccumulator((H, W), "cpu")
    acc.add(pe)
    acc.add(pe2)
    m = acc.maps()
    assert m["motion_cnt"].max() == 2 and math.isclose(m["motion"][0, 0].item(), 0.025, abs_tol=1e-6)
    s = acc.scalars()
    assert all(math.isfinite(v) for v in s)
    # covisibility threshold masks out the motion term
    flow_occ = make_flow_output(torch.zeros(1, 2, H, W), torch.zeros(1, H, W))
    pe3 = compute_pair_errors(geo, 0, 1, flow_occ, conf_mask)
    assert not pe3.mot_valid.any() and pe3.wrong_occ.all() and pe3.fus_mask.all()


def test_build_flow_max_side_plumbing():
    # build_flow must not override a backend's max_side default when the caller leaves it unset
    # (UFM defaults to native resolution, the RAFT family to 1024); an explicit None means "no cap".
    import geco.flow as flow_registry

    captured = {}

    class _DummyFlow:
        def __init__(self, device="cpu", ckpt=None, model=None, max_side=1024):
            captured["max_side"] = max_side

    flow_registry._DummyFlow = _DummyFlow
    flow_registry.register_flow_backend("_dummy", "geco.flow", "_DummyFlow")
    try:
        flow_registry.build_flow("_dummy", device="cpu")
        assert captured["max_side"] == 1024
        flow_registry.build_flow("_dummy", device="cpu", max_side=None)
        assert captured["max_side"] is None
        flow_registry.build_flow("_dummy", device="cpu", max_side=512)
        assert captured["max_side"] == 512
    finally:
        del flow_registry._REGISTRY["_dummy"]
        del flow_registry._DummyFlow


def test_validate_rejects_bad_outputs():
    geo = _synthetic_geometry()
    bad = GeometryOutput(geo.intrinsic, geo.extrinsic * 2, geo.depth, geo.conf, geo.valid, geo.model_hw)
    try:
        bad.validate()
        raise AssertionError("non-orthonormal rotation accepted")
    except ValueError:
        pass
    try:
        FlowOutput(torch.zeros(4, 4, 2), torch.full((4, 4), 2.0), torch.ones(4, 4, dtype=torch.bool)).validate()
        raise AssertionError("covisibility > 1 accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
    print(f"{len(tests)} contract tests passed")
