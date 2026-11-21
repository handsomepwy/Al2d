"""Build the 'girl' model (first character) from its layered PSD, entirely in code.

    python characters/girl/build.py [output folder] [--install]

--install also copies the converted model into VTube Studio's model folder.

The recipes and checks are in rigkit.py; this file holds what belongs to this character: which
PSD, which layers go together, how far her hair swings.  (The script as it was before the recipes
moved out wrote the same .cmo3.)

Rig: head turn (ParamAngleX / ParamAngleY), blink, mouth open/close, pupils, and hair that sways
(ParamHairFront / ParamHairSide / ParamHairBack) driven by our own physics file.
"""
import os
import sys

from al2d import config, physics
from al2d import rigkit
from al2d.rigkit import Head, Tuning, drawn_box, rig_blink, rig_mouth, rig_pupils, rig_sway, rig_turn

PSD_PATH = os.path.join(config.ROOT, 'examples', 'girl', 'girl.psd')
NAME = 'girl_code'
TUNING = Tuning()       # the recipes' lengths were chosen on this character

# Layers that only show where another layer is visible (name -> masking layer).
# The shadow is drawn 40 px past the face on purpose and needs the mask even in the still picture.
# The irises fit inside the eye whites at rest; the mask matters once they move.
CLIPPED_BY = {'front hair shadow': 'face', 'irides-l': 'eyewhite-l', 'irides-r': 'eyewhite-r'}
CLIPPED_IN_PSD = {'front hair shadow': 'face'}      # what the flattened reference picture applies

# Pixels the lowest part of the hair moves sideways when its sway parameter is at +-1.  Physics
# reaches about +-0.3 with ordinary head movement and +-0.8 with a brisk turn (see HAIR_PHYSICS).
FRONT_HAIR_SWAY = 16
SIDE_HAIR_SWAY = 45
BACK_HAIR_SWAY = 70

# The pose the PSD shows.  The mouth is drawn open, but rests closed (ParamMouthOpenY default 0).
PSD_POSE = {'ParamMouthOpenY': 1}

# Grid cell size in pixels for layers that bend a lot (default: chosen from the layer's size).
MESH_SPACING = {'eyelash-l': 8, 'eyelash-r': 8, 'eyelid-l': 8, 'eyelid-r': 8, 'mouth inner': 6}

# Physics: one pendulum per sway parameter, as (name, output parameter, length, delay, mobility, scale).
# Longer = slower and calmer.  The scale is chosen so a brisk head turn brings the parameter to
# about 0.8 (tried with physics.Simulation; see the "physics" line the build prints).
HAIR_PHYSICS = [
    ('front hair', 'ParamHairFront', 5, 0.9, 0.95, 1.0),
    ('side hair', 'ParamHairSide', 8, 0.8, 0.95, 1.3),
    ('back hair', 'ParamHairBack', 15, 0.8, 0.85, 1.6),
]
# What swings the pendulums: (parameter, kind, weight).  Only what the rig shows: head tilt
# (ParamAngleZ) and body angles move nothing yet, and hair leaning with no visible cause looks wrong.
# Add ('ParamAngleZ', 'Angle', 60), ('ParamBodyAngleX', 'X', 40), ('ParamBodyAngleZ', 'Angle', 40)
# when those are rigged.
HAIR_INPUTS = [('ParamAngleX', 'X', 60)]


def rig(project, meshes, parameters, size):
    """The moving parts.  Everything here is in canvas pixels at the rest pose."""
    for side in 'lr':
        rig_blink(project, meshes, parameters, side, size, TUNING)
    rig_mouth(project, meshes, parameters, size)
    rig_pupils(project, meshes, parameters, TUNING)

    # Hair sway.  Front: the front hair was cut from the two side locks along a straight line and
    # overlaps their tops, so only the part between the locks may move, or the cut opens and their
    # top edges show.  Its shadow on the forehead goes with it.
    _, top, _, bottom = drawn_box(meshes['front hair'])
    between_locks = (float(meshes['front hair side-r'].vertices[:, 0].max()),
                     float(meshes['front hair side-l'].vertices[:, 0].min()))
    rig_sway(project, parameters['ParamHairFront'], 'sw_front', 'WarpHairFront',
             [meshes['front hair'], meshes['front hair shadow']], FRONT_HAIR_SWAY, top, bottom,
             cols=18, rows=14, hold=between_locks)
    # Side locks: each hangs from its top.
    for side in 'lr':
        lock = meshes[f'front hair side-{side}']
        _, top, _, bottom = drawn_box(lock)
        rig_sway(project, parameters['ParamHairSide'], f'sw_side_{side}', f'WarpHairSide{side.upper()}',
                 [lock], SIDE_HAIR_SWAY, top, bottom, cols=4, rows=16)
    # Back hair: three layers cut from one sheet with straight lines, so one warp for all three.
    # Still around the head (the outline there must not move), swinging freely below the chin.
    back = [meshes[name] for name in ('back hair side-l', 'back hair side-r', 'back hair')]
    rig_sway(project, parameters['ParamHairBack'], 'sw_back', 'WarpHairBack', back, BACK_HAIR_SWAY,
             drawn_box(meshes['face'])[3], max(drawn_box(mesh)[3] for mesh in back), cols=12, rows=20, power=1.5)

    # Head turn: one warp per depth, around everything above (the blink warps, the sway warps).
    head = Head(meshes, TUNING)
    group = lambda *names: [meshes[name] for name in names]
    rig_turn(project, parameters, 'turn_face', 'WarpTurnFace',
             group('face', 'ears-l', 'ears-r', 'eyebrow-l', 'eyebrow-r', 'eyelash-l', 'eyelash-r',
                   'eyelid-l', 'eyelid-r', 'irides-l', 'irides-r', 'eyewhite-l', 'eyewhite-r',
                   'mouth line', 'mouth upper', 'mouth inner'),
             head.skin_shift, cols=12, rows=14)
    rig_turn(project, parameters, 'turn_nose', 'WarpTurnNose', group('nose'), head.nose_shift, cols=2, rows=2)
    rig_turn(project, parameters, 'turn_hair', 'WarpTurnHair',
             group('front hair', 'front hair side-l', 'front hair side-r', 'front hair shadow'),
             head.hair_shift, cols=12, rows=24)
    rig_turn(project, parameters, 'turn_back', 'WarpTurnBack',
             group('back hair side-l', 'back hair side-r', 'back hair'), head.back_shift, cols=18, rows=20)
    rig_turn(project, parameters, 'turn_neck', 'WarpTurnNeck', group('neck'), head.neck_shift, cols=3, rows=6)


def main():
    arguments = [a for a in sys.argv[1:] if a != '--install']
    out_dir = arguments[0] if arguments else config.out_dir(NAME)
    built = rigkit.build(NAME, PSD_PATH, out_dir, rig=rig, clipped_by=CLIPPED_BY, mesh_spacing=MESH_SPACING)
    hair_physics = physics.physics_file([
        physics.pendulum(f'PhysicsSetting{number}', name, output, length, delay, mobility, scale=scale, inputs=HAIR_INPUTS)
        for number, (name, output, length, delay, mobility, scale) in enumerate(HAIR_PHYSICS, 1)])
    rigkit.convert(built, physics_data=hair_physics, install='--install' in sys.argv)
    built.regions = rigkit.measure_regions(built.meshes, built.size)

    rigkit.report_rest(built, PSD_POSE, CLIPPED_IN_PSD)
    rigkit.report_positions(built)
    rigkit.report_mouth(built)
    rigkit.report_pupils(built)
    rigkit.report_blink(built)
    rigkit.report_turn(built)
    left, top, right, bottom = (int(v) for v in next(w for w in built.project.warps if w.name == 'sw_front').box)
    hair = (max(left - 60, 0), max(top - 60, 0), right + 60, bottom + 120)
    figure = built.regions['figure']
    rigkit.report_sway(built, [('check_sway.jpg', 'ParamHairFront', 'Hair Front', hair, 1.0),
                               ('check_sway_side.jpg', 'ParamHairSide', 'Hair Side', figure, 0.6),
                               ('check_sway_back.jpg', 'ParamHairBack', 'Hair Back', figure, 0.6)],
                       lags={'ParamHairFront': 0.0, 'ParamHairSide': 0.5, 'ParamHairBack': 1.0})
    rigkit.report_physics(built, hair_physics)


if __name__ == '__main__':
    main()
