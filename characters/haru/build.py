"""Build Haru's head (Live2D sample, receptionist version) from the artist's PSD with our recipes,
and compare it with Live2D's own rig of the same art.  Private research only: never sold.

    python characters/haru/prep.py                            (once: artist's PSD -> out/haru_prep/haru_head.psd)
    python characters/haru/build.py [output folder] [--install]

This is the first character whose layers were drawn by an artist for Live2D rather than split by
See-through, so two recipes have an artist-style variant: the blink (an invisible eye-opening mask
clips the eye; rig_blink_masked) and the mouth (skin patches over a drawn open mouth;
rig_mouth_patches).  The rest is the recipes Qwen娘 used, scaled to this face.
"""
import json
import os
import sys

import numpy as np

from al2d import l2dcore
from al2d import config, physics
from al2d import rigkit
from prep import HEAD_BOX, OUT_PSD
from al2d.rigkit import (Head, Tuning, drawn_box, rig_blink_masked, rig_brows, rig_mouth_form, rig_mouth_patches,
                    rig_pupils, rig_sway, rig_tilt, rig_turn)

NAME = 'haru_code'

# Haru's PSD has the smiling mouth, so Mouth Form rests at +1 (as in Live2D's rig); she also has
# the brows' sideways, angle and form parameters.
PARAMETERS = [(pid, label, low, high, 1 if pid == 'ParamMouthForm' else default, track)
              for pid, label, low, high, default, track in rigkit.STANDARD_PARAMETERS]
PARAMETERS += [(f'ParamBrow{s}{what}', f'Brow {s} {what}', -1, 1, 0, 'NONE')
               for what in ('X', 'Angle', 'Form') for s in 'LR']
PRO_MODEL3 = os.path.join(config.need('haru_dir'), 'runtime', 'haru_greeter_t05.model3.json')

EYE_PARTS = {side: [f'eyewhite-{side}', f'irides-{side}', f'eye light high-{side}', f'eye light low-{side}']
             for side in 'lr'}
# The eye white, iris and highlights only show inside the invisible eye-opening mask.
CLIPPED_BY = {name: f'eye mask-{side}' for side in 'lr' for name in EYE_PARTS[side]}
# Hairline pieces at the temples only show over the back hair (as in the artist's setup).
CLIPPED_BY.update({'hairline-l': 'back hair', 'hairline-r': 'back hair'})

MESH_SPACING = {**{f'lash {n}-{side}': 5 for n in range(1, 6) for side in 'lr'},
                **{f'eye mask-{side}': 5 for side in 'lr'},
                'mouth upper': 6, 'mouth lower': 6}

# Sway, in pixels for the reference face (scaled below), as for Qwen娘.
FRONT_HAIR_SWAY = 16
SIDE_HAIR_SWAY = 18
TAIL_SWAY = 40

HAIR_PHYSICS = [
    ('front hair', 'ParamHairFront', 5, 0.9, 0.95, 1.0),
    ('side hair', 'ParamHairSide', 7, 0.85, 0.95, 1.2),
    ('tail', 'ParamHairBack', 12, 0.8, 0.85, 1.5),
]
# As in Live2D's rig of her: the hair swings when the head turns or tilts (her body does not move yet).
HAIR_INPUTS = [('ParamAngleX', 'X', 60), ('ParamAngleZ', 'Angle', 60)]


def rig(project, meshes, parameters, size):
    group = lambda *names: [meshes[name] for name in names if name in meshes]
    face = drawn_box(meshes['face'])
    factor = (face[2] - face[0]) / rigkit.REFERENCE_FACE_WIDTH
    # Measured against Live2D's rig of this art (movement_table.txt): it turns the face further
    # up and down than Qwen娘's numbers, and moves the lower face most when looking up.
    tuning = Tuning(turn_face_middle=(52, 36), turn_face_edge=(12, 16), turn_y_centre=0.5,
                    turn_hair=(64, 37)).for_face(face[2] - face[0])

    for side in 'lr':
        meshes[f'eye mask-{side}'].opacity = 0.0
        # Shares from Live2D's rig of her (ref_haru/analysis): the inner corner piece (3) and the
        # wing (4) come down less than the line; the outer corner stroke (5) shrinks into the line.
        lashes = {meshes[f'lash {n}-{side}']: share for n, share in ((1, 1.0), (2, 1.0), (3, 0.8), (4, 0.9))}
        corner_stroke = {meshes[f'lash 5-{side}']: 1.0}
        crease = {mesh: 0.45 for mesh in group(f'crease-{side}', f'crease shadow-{side}')}
        lower = {meshes[f'lower lash 1-{side}']: 1.0, meshes[f'lower lash 2-{side}']: 0.8}
        rig_blink_masked(project, meshes, parameters, side, size, {**lashes, **crease}, meshes[f'eye mask-{side}'],
                         lower=lower, sides=corner_stroke, tuning=tuning)
    rig_mouth_patches(project, meshes, parameters, size)
    rig_mouth_form(project, meshes, parameters, size, tuning)
    rig_brows(project, meshes, parameters, tuning)
    rig_pupils(project, meshes, parameters, tuning,
               followers={side: [f'eye light high-{side}', f'eye light low-{side}'] for side in 'lr'})

    # Hair sway.
    _, top, _, bottom = drawn_box(meshes['front hair'])
    rig_sway(project, parameters['ParamHairFront'], 'sw_front', 'WarpHairFront', group('front hair'),
             FRONT_HAIR_SWAY * factor, top, bottom, cols=12, rows=12)
    sides = group('side hair-l', 'side hair-r')
    top = min(drawn_box(mesh)[1] for mesh in sides)
    bottom = max(drawn_box(mesh)[3] for mesh in sides)
    rig_sway(project, parameters['ParamHairSide'], 'sw_side', 'WarpHairSide', sides,
             SIDE_HAIR_SWAY * factor, top, bottom, cols=10, rows=8)
    _, top, _, bottom = drawn_box(meshes['tail'])
    rig_sway(project, parameters['ParamHairBack'], 'sw_tail', 'WarpHairTail', group('tail'),
             TAIL_SWAY * factor, top, bottom, cols=8, rows=12, power=1.5)

    # Head turn: one warp per depth.
    collar_top = drawn_box(meshes['collar'])[1]
    head = Head(meshes, tuning, neck_bottom=collar_top + 0.5 * (drawn_box(meshes['collar'])[3] - collar_top),
                hair_pinned=False)
    eyes = [name for side in 'lr' for name in
            [f'eye mask-{side}', f'crease-{side}', f'crease shadow-{side}', *EYE_PARTS[side],
             *[f'lash {n}-{side}' for n in range(1, 6)], *[f'lower lash {n}-{side}' for n in range(1, 3)]]]
    turns = [rig_turn(project, parameters, 'turn_face', 'WarpTurnFace',
             group('face', 'ears-l', 'ears-r', 'eyebrow-l', 'eyebrow-r', 'cheek-l', 'cheek-r',
                   'face light-l', 'face light-r', 'mouth upper', 'mouth lower', 'mouth inner', *eyes),
             head.skin_shift, cols=12, rows=14),
             rig_turn(project, parameters, 'turn_nose', 'WarpTurnNose', group('nose'), head.nose_shift, cols=2, rows=2),
             rig_turn(project, parameters, 'turn_hair', 'WarpTurnHair',
                      group('front hair', 'side hair-l', 'side hair-r', 'hairline-l', 'hairline-r'),
                      head.hair_shift, cols=12, rows=16),
             rig_turn(project, parameters, 'turn_back', 'WarpTurnBack', group('back hair'), head.back_shift, cols=14, rows=16),
             rig_turn(project, parameters, 'turn_tail', 'WarpTurnTail', group('tail'), head.behind_shift, cols=2, rows=2)]
    neck = rig_turn(project, parameters, 'turn_neck', 'WarpTurnNeck', group('neck'), head.neck_shift, cols=4, rows=8)

    # Head tilt: a rotation deformer around the five head warps; the neck bends after it.
    rig_tilt(project, parameters, turns, head, neck=neck)


# ---------------------------------------------------------------- comparison with Live2D's rig

COMPARE_POSES = [
    ('rest', {}),
    ('turn X+30', {'ParamAngleX': 30}),
    ('turn X-30', {'ParamAngleX': -30}),
    ('turn Y+30', {'ParamAngleY': 30}),
    ('turn Y-30', {'ParamAngleY': -30}),
    ('turn X+30 Y+30', {'ParamAngleX': 30, 'ParamAngleY': 30}),
    ('eyes half', {'ParamEyeLOpen': 0.5, 'ParamEyeROpen': 0.5}),
    ('eyes shut', {'ParamEyeLOpen': 0, 'ParamEyeROpen': 0}),
    ('mouth half', {'ParamMouthOpenY': 0.5}),
    ('mouth open', {'ParamMouthOpenY': 1}),
    ('look right-up', {'ParamEyeBallX': 1, 'ParamEyeBallY': 1}),
    ('look left-down', {'ParamEyeBallX': -1, 'ParamEyeBallY': -1}),
    ('hair +1', {'ParamHairFront': 1, 'ParamHairSide': 1, 'ParamHairBack': 1}),
    ('tilt Z+30', {'ParamAngleZ': 30}),
    ('tilt Z-30', {'ParamAngleZ': -30}),
    ('tilt Z+30, turn X-30 Y+30', {'ParamAngleZ': 30, 'ParamAngleX': -30, 'ParamAngleY': 30}),
    ('turned, shut, talking', {'ParamAngleX': -30, 'ParamAngleY': -20, 'ParamEyeLOpen': 0, 'ParamEyeROpen': 0,
                               'ParamMouthOpenY': 1}),
]


EXPRESSION_POSES = [
    ('rest (mouth form 1)', {}),
    ('mouth form 0', {'ParamMouthForm': 0}),
    ('mouth form -1', {'ParamMouthForm': -1}),
    ('form 0, half open', {'ParamMouthForm': 0, 'ParamMouthOpenY': 0.5}),
    ('form -1, open', {'ParamMouthForm': -1, 'ParamMouthOpenY': 1}),
    ('brows up', {'ParamBrowLY': 1, 'ParamBrowRY': 1}),
    ('brows down, out', {'ParamBrowLY': -1, 'ParamBrowRY': -1, 'ParamBrowLX': 1, 'ParamBrowRX': 1}),
    ('brows angle +1', {'ParamBrowLAngle': 1, 'ParamBrowRAngle': 1}),
    ('brows form +1', {'ParamBrowLForm': 1, 'ParamBrowRForm': 1}),
    ('brows form -1', {'ParamBrowLForm': -1, 'ParamBrowRForm': -1}),
    ('angry', {'ParamBrowLY': -1, 'ParamBrowRY': -1, 'ParamBrowLAngle': -1, 'ParamBrowRAngle': -1,
               'ParamBrowLForm': -1, 'ParamBrowRForm': -1, 'ParamMouthForm': -1}),
    ('eyes shut, smile', {'ParamEyeLOpen': 0, 'ParamEyeROpen': 0, 'ParamEyeLSmile': 1, 'ParamEyeRSmile': 1}),
    ('eyes half, smile', {'ParamEyeLOpen': 0.5, 'ParamEyeROpen': 0.5, 'ParamEyeLSmile': 1, 'ParamEyeRSmile': 1}),
    ('happy: shut smile, open', {'ParamEyeLOpen': 0, 'ParamEyeROpen': 0, 'ParamEyeLSmile': 1, 'ParamEyeRSmile': 1,
                                 'ParamMouthOpenY': 1, 'ParamBrowLForm': 1, 'ParamBrowRForm': 1}),
]


def report_expressions(built):
    """Nothing that must stay hidden shows in the new expression poses."""
    model, textures, project = built.model, built.textures, built.project
    for form in (-1, 0, 1):
        pose = {'ParamMouthForm': form, 'ParamMouthOpenY': 0}
        print(f'mouth form {form:+d}, closed: interior pixels showing',
              rigkit.showing(model, textures, project, {'mouth inner'}, pose, built.regions['mouth']))
    for label, pose in EXPRESSION_POSES:
        if 'smile' in label:
            print(f'{label}: eye pixels showing',
                  rigkit.showing(model, textures, project, rigkit.EYE_LAYERS, pose, built.regions['eyes']))
    model.reset()


def compare_with_pro(built, region, scale, file_name, poses=COMPARE_POSES):
    """Rows of (Live2D's rig | ours) at the same parameter values, cut to the same canvas region."""
    pro = l2dcore.Model(os.path.join(os.path.dirname(PRO_MODEL3),
                                     json.load(open(PRO_MODEL3, encoding='utf-8'))['FileReferences']['Moc']))
    pro_textures = l2dcore.load_textures(PRO_MODEL3)
    x0, y0, x1, y1 = region
    pro_region = (x0 + HEAD_BOX[0], y0 + HEAD_BOX[1], x1 + HEAD_BOX[0], y1 + HEAD_BOX[1])
    rows = []
    for label, pose in poses:
        pro.pose(pose)
        built.model.pose(pose)
        left = rigkit.labelled(l2dcore.render(pro, pro_textures, scale=scale, region=pro_region), 'Live2D: ' + label)
        right = rigkit.labelled(l2dcore.render(built.model, built.textures, scale=scale, region=region), 'ours: ' + label)
        rows.append(np.concatenate([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right], axis=1))
    built.model.reset()
    # Two columns of pairs, so the sheet stays readable.
    half = (len(rows) + 1) // 2
    blank = np.full_like(rows[0], 255)
    left_column = np.concatenate(rows[:half], axis=0)
    right_column = np.concatenate(rows[half:] + [blank] * (2 * half - len(rows)), axis=0)
    gap = np.full((left_column.shape[0], 24, 3), 200, np.uint8)
    l2dcore.save_rgb(built.path(file_name), np.concatenate([left_column, gap, right_column], axis=1))


def movement_table(built, poses, layers):
    """For each layer and pose: how far its points move on average, Live2D's rig against ours,
    as (dx, dy) in canvas pixels.  Layers are matched through prep.py's name table."""
    from prep import NAMES
    pro_path = os.path.join(os.path.dirname(PRO_MODEL3),
                            json.load(open(PRO_MODEL3, encoding='utf-8'))['FileReferences']['Moc'])
    pro = l2dcore.Model(pro_path)
    names = json.load(open(os.path.join(config.out_dir('haru_prep'), 'drawable_names.json'), encoding='utf-8'))
    pro_index = {NAMES.get(names.get(d, '').replace('\x7f', '').strip()): i for i, d in enumerate(pro.drawable_ids)}
    ours = built.model
    our_index = {mesh.name: ours.drawable_ids.index(mesh.id) for mesh in built.project.meshes}

    def shifts(model, index, pose):
        model.reset()
        rest = model.vertices_px()
        model.pose(pose)
        moved = model.vertices_px()
        return {name: (moved[i] - rest[i]).mean(axis=0) for name, i in index.items() if name}

    lines = []
    for label, pose in poses:
        a, b = shifts(pro, pro_index, pose), shifts(ours, our_index, pose)
        lines.append(f'{label}:  layer  Live2D (dx, dy)  ours (dx, dy)')
        for name in layers:
            if name in a and name in b:
                lines.append(f'    {name:18s} ({a[name][0]:+6.1f}, {a[name][1]:+6.1f})   ({b[name][0]:+6.1f}, {b[name][1]:+6.1f})')
    ours.reset()
    text = '\n'.join(lines)
    open(built.path('movement_table.txt'), 'w', encoding='utf-8').write(text)
    print(text)


def main():
    arguments = [a for a in sys.argv[1:] if a != '--install']
    out_dir = arguments[0] if arguments else config.out_dir(NAME)
    built = rigkit.build(NAME, OUT_PSD, out_dir, rig=rig, clipped_by=CLIPPED_BY, mesh_spacing=MESH_SPACING,
                         parameters=PARAMETERS)
    hair_physics = physics.physics_file([
        physics.pendulum(f'PhysicsSetting{number}', name, output, length, delay, mobility, scale=scale, inputs=HAIR_INPUTS)
        for number, (name, output, length, delay, mobility, scale) in enumerate(HAIR_PHYSICS, 1)])
    rigkit.convert(built, physics_data=hair_physics, install='--install' in sys.argv)
    built.regions = rigkit.measure_regions(built.meshes, built.size)
    built.regions['mouth'] = rigkit.padded(drawn_box(built.meshes['mouth lower']), 30, 25, built.size)

    rigkit.report_rest(built, clipped_in_psd=CLIPPED_BY, skip_in_psd=('eye mask-l', 'eye mask-r'))
    rigkit.report_positions(built)
    rigkit.report_mouth(built)
    rigkit.report_pupils(built)
    rigkit.report_blink(built)
    rigkit.report_turn(built)
    rigkit.report_tilt(built)
    rigkit.report_sway(built, [('check_sway.jpg', 'ParamHairFront', 'Hair Front', built.regions['head'], 0.8),
                               ('check_sway_side.jpg', 'ParamHairSide', 'Side hair', built.regions['head'], 0.8),
                               ('check_sway_back.jpg', 'ParamHairBack', 'Tail', built.regions['figure'], 0.6)],
                       lags={'ParamHairFront': 0.0, 'ParamHairSide': 0.5, 'ParamHairBack': 1.0})
    rigkit.report_physics(built, hair_physics)
    compare_with_pro(built, built.regions['head'], 0.6, 'compare_head.jpg')
    compare_with_pro(built, built.regions['eyes'], 1.5, 'compare_eyes.jpg',
                     [p for p in COMPARE_POSES if p[0].startswith(('rest', 'eyes', 'look', 'turn X+30'))])
    compare_with_pro(built, built.regions['mouth'], 3.0, 'compare_mouth.jpg',
                     [p for p in COMPARE_POSES if p[0].startswith(('rest', 'mouth'))])
    report_expressions(built)
    face = rigkit.union([built.regions['eyes'], built.regions['mouth'], drawn_box(built.meshes['eyebrow-l']),
                         drawn_box(built.meshes['eyebrow-r'])])
    compare_with_pro(built, rigkit.padded(face, 20, 20, built.size), 1.0, 'compare_expressions.jpg', EXPRESSION_POSES)
    movement_table(built, [('turn X+30', {'ParamAngleX': 30}), ('turn Y+30', {'ParamAngleY': 30}),
                           ('tilt Z+30', {'ParamAngleZ': 30}), ('tilt Z-30', {'ParamAngleZ': -30}),
                           ('eyes shut', {'ParamEyeLOpen': 0, 'ParamEyeROpen': 0}),
                           ('mouth open', {'ParamMouthOpenY': 1})] +
                   [p for p in EXPRESSION_POSES[1:] if p[0] != 'angry' and not p[0].startswith('happy')],
                   ['face', 'nose', 'eyewhite-l', 'eyewhite-r', 'lash 1-l', 'lash 2-l', 'lash 5-l', 'crease-l',
                    'eyebrow-l', 'eyebrow-r', 'lower lash 1-l', 'mouth upper', 'mouth lower', 'mouth inner', 'ears-l', 'ears-r',
                    'front hair', 'side hair-l', 'side hair-r', 'hairline-l', 'back hair', 'tail', 'neck'])


if __name__ == '__main__':
    main()
