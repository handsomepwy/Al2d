"""Compare two exported models numerically through Cubism Core.

Usage: python compare_models.py <reference model3.json> <candidate model3.json> <output folder>

For a list of poses, both models are posed identically and every drawable's deformed vertices are
compared in canvas pixels.  Both are also rendered, and side-by-side sheets are written.
"""
import json
import os
import sys

import cv2
import numpy as np

from al2d import l2dcore


def moc_of(model3_path):
    manifest = json.load(open(model3_path, encoding="utf-8"))
    return os.path.join(os.path.dirname(model3_path), manifest["FileReferences"]["Moc"])


def poses_for(model):
    """Default, each parameter at min and max alone, and a few combined poses."""
    poses = [("default", {})]
    for name, low, high, default in zip(model.param_ids, model.param_min, model.param_max, model.param_default):
        if low != default:
            poses.append((f"{name}=min", {name: float(low)}))
        if high != default:
            poses.append((f"{name}=max", {name: float(high)}))
        middle = (low + high) / 2 if default in (low, high) else (default + high) / 2
        poses.append((f"{name}=mid", {name: float(middle)}))
    rng = np.random.default_rng(7)
    for number in range(8):
        combo = {name: float(rng.uniform(low, high))
                 for name, low, high in zip(model.param_ids, model.param_min, model.param_max)}
        poses.append((f"random{number}", combo))
    return poses


def main():
    reference_json, candidate_json, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    reference = l2dcore.Model(moc_of(reference_json))
    candidate = l2dcore.Model(moc_of(candidate_json))
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    for label, model in (("reference", reference), ("candidate", candidate)):
        say(f"{label}: {model.path}")
        say(f"  moc version {model.file_version} (Core reads up to {model.core_latest}), consistency OK={model.consistent}")
        say(f"  canvas {model.canvas} origin {model.origin} ppu {model.ppu}")
        say(f"  parameters {model.param_count}, parts {model.part_count}, drawables {model.drawable_count}, "
            f"vertices {int(model.vertex_counts.sum())}, triangles {int(model.index_counts.sum()) // 3}")

    # ---- static structure ----
    say()
    say("== structure ==")
    problems = 0
    if reference.param_ids != candidate.param_ids:
        only_ref = [p for p in reference.param_ids if p not in candidate.param_ids]
        only_cand = [p for p in candidate.param_ids if p not in reference.param_ids]
        say(f"parameter ids differ: only in reference {only_ref}, only in candidate {only_cand}, "
            f"same set in different order={not only_ref and not only_cand}")
        problems += bool(only_ref or only_cand)
    else:
        say("parameter ids: identical, same order")
    for name in reference.param_ids:
        if name in candidate.param_ids:
            i, j = reference.param_ids.index(name), candidate.param_ids.index(name)
            ref_range = (reference.param_min[i], reference.param_max[i], reference.param_default[i])
            cand_range = (candidate.param_min[j], candidate.param_max[j], candidate.param_default[j])
            if ref_range != cand_range:
                say(f"  range differs for {name}: {ref_range} vs {cand_range}")
                problems += 1
    say(f"part ids: {'identical' if reference.part_ids == candidate.part_ids else 'DIFFER ' + str(reference.part_ids) + ' vs ' + str(candidate.part_ids)}")
    common = [d for d in reference.drawable_ids if d in candidate.drawable_ids]
    say(f"drawables: {len(common)} shared, only in reference "
        f"{[d for d in reference.drawable_ids if d not in candidate.drawable_ids]}, only in candidate "
        f"{[d for d in candidate.drawable_ids if d not in reference.drawable_ids]}")
    say(f"drawable order in file identical: {reference.drawable_ids == candidate.drawable_ids}")
    pairs = [(reference.drawable_ids.index(d), candidate.drawable_ids.index(d)) for d in common]
    for i, j in pairs:
        name = reference.drawable_ids[i]
        if reference.vertex_counts[i] != candidate.vertex_counts[j]:
            say(f"  {name}: vertex count {reference.vertex_counts[i]} vs {candidate.vertex_counts[j]}")
            problems += 1
        if reference.index_counts[i] != candidate.index_counts[j] or not np.array_equal(reference.indices[i], candidate.indices[j]):
            say(f"  {name}: triangle list differs")
            problems += 1
        if reference.constant_flags[i] != candidate.constant_flags[j]:
            say(f"  {name}: constant flags {reference.constant_flags[i]} vs {candidate.constant_flags[j]}")
            problems += 1
        ref_masks = sorted(reference.drawable_ids[k] for k in reference.masks[i] if k >= 0)
        cand_masks = sorted(candidate.drawable_ids[k] for k in candidate.masks[j] if k >= 0)
        if ref_masks != cand_masks:
            say(f"  {name}: clipping masks {ref_masks} vs {cand_masks}")
            problems += 1
    say(f"structure problems: {problems}")

    # ---- poses ----
    say()
    say("== deformed vertex positions, candidate vs reference, in canvas pixels ==")
    ref_textures = l2dcore.load_textures(reference_json)
    cand_textures = l2dcore.load_textures(candidate_json)
    worst_overall = 0.0
    rows = []
    render_these = {"default", "ParamAngleX=min", "ParamAngleX=max", "ParamAngleY=min", "ParamAngleY=max",
                    "ParamEyeLOpen=min", "ParamEyeROpen=min", "ParamMouthOpenY=max", "ParamEyeBallX=max",
                    "ParamEyeBallY=max", "ParamHairFront=max", "ParamHairSide=max", "ParamHairBack=max",
                    "random0", "random1", "random2"}
    image_diffs = []
    for pose_name, pose in poses_for(reference):
        reference.pose(pose)
        candidate.pose({k: v for k, v in pose.items() if k in candidate.param_ids})
        ref_vertices, cand_vertices = reference.vertices_px(), candidate.vertices_px()
        ref_orders, cand_orders = reference.render_orders(), candidate.render_orders()
        ref_opacity, cand_opacity = reference.opacities(), candidate.opacities()
        worst, worst_name, total_moved = 0.0, "", 0.0
        order_same = [reference.drawable_ids[i] for i in np.argsort(ref_orders, kind="stable")] == \
                     [candidate.drawable_ids[j] for j in np.argsort(cand_orders, kind="stable")]
        opacity_gap = 0.0
        for i, j in pairs:
            if ref_vertices[i].shape != cand_vertices[j].shape:
                continue
            gap = float(np.abs(ref_vertices[i] - cand_vertices[j]).max()) if len(ref_vertices[i]) else 0.0
            if gap > worst:
                worst, worst_name = gap, reference.drawable_ids[i]
            opacity_gap = max(opacity_gap, abs(float(ref_opacity[i]) - float(cand_opacity[j])))
        worst_overall = max(worst_overall, worst)
        rows.append((pose_name, worst, worst_name, order_same, opacity_gap))
        if pose_name in render_these:
            ref_image = l2dcore.render(reference, ref_textures)
            cand_image = l2dcore.render(candidate, cand_textures)
            difference = np.abs(ref_image.astype(np.int16) - cand_image.astype(np.int16)).max(axis=2)
            image_diffs.append((pose_name, float(difference.mean()), int((difference > 24).sum())))
            heat = np.repeat(np.clip(difference * 4, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
            sheet = np.concatenate([ref_image, cand_image, heat], axis=1)
            cv2.putText(sheet, f"{pose_name}   left: Editor export   middle: Umamo convert   right: difference x4",
                        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            safe = pose_name.replace("=", "_")
            l2dcore.save_rgb(os.path.join(out_dir, f"cmp_{safe}.jpg"), sheet)
    for pose_name, worst, worst_name, order_same, opacity_gap in rows:
        flag = "" if worst < 0.05 and order_same and opacity_gap < 0.004 else "   <-- look"
        say(f"{pose_name:28s} max vertex gap {worst:8.4f} px ({worst_name:12s}) draw order same={order_same} "
            f"opacity gap {opacity_gap:.4f}{flag}")
    say()
    say(f"WORST vertex gap over all poses: {worst_overall:.4f} px")
    say()
    say("== rendered image difference (0..255 per channel) ==")
    for pose_name, mean, strong in image_diffs:
        say(f"{pose_name:28s} mean {mean:6.3f}   pixels differing by more than 24: {strong}")
    open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
