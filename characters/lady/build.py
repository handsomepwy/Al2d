"""Build the 'lady' model (second character) from its layered PSD, entirely in code.

    python characters/lady/prep.py     (once: See-through's layers -> out/lady_prep/lady.psd)
    python characters/lady/build.py [output folder] [--install]

--install also copies the converted model into VTube Studio's model folder.

The recipes are the same ones the first character uses (rigkit.py), with their lengths scaled to
this face.  This file holds what belongs to this character: which layers go together, how far
her hair swings.

Rig: head turn (ParamAngleX / ParamAngleY), blink, mouth open/close, pupils, and hair that sways
(ParamHairFront = fringe and the locks beside the face, ParamHairBack = the hair behind,
ParamHairSide = the braid) driven by our own physics file.
"""
import sys

import numpy as np
from PIL import Image

from al2d import config, physics
from al2d import rigkit
from al2d.rigkit import Head, Tuning, drawn_box, on_canvas, rig_blink, rig_mouth, rig_pupils, rig_sway, rig_turn
from prep import OUT_PSD, SOURCE_PICTURE

PSD_PATH = OUT_PSD                  # written by prep.py
NAME = 'lady_code'

# The irises fit inside the eye whites at rest; the mask matters once they move.
CLIPPED_BY = {'irides-l': 'eyewhite-l', 'irides-r': 'eyewhite-r'}

# The pose the PSD shows.  The mouth is drawn open, but rests closed (ParamMouthOpenY default 0).
PSD_POSE = {'ParamMouthOpenY': 1}

# Grid cell size in pixels for layers that bend a lot (default: chosen from the layer's size).
MESH_SPACING = {'eyelash-l': 8, 'eyelash-r': 8, 'eyelid-l': 8, 'eyelid-r': 8, 'mouth inner': 6}

# Pixels the lowest part of the hair moves sideways at sway +-1, for the first character's face
# size (scaled to this face below).  Same as the first character's front / side / back.
FRONT_HAIR_SWAY = 16
BRAID_SWAY = 20         # the middle of the braid (both ends are held)
BACK_HAIR_SWAY = 70

# Physics, as for the first character: (name, output parameter, length, delay, mobility, scale).
HAIR_PHYSICS = [
    ('front hair', 'ParamHairFront', 5, 0.9, 0.95, 1.0),
    ('braid', 'ParamHairSide', 8, 0.8, 0.95, 1.3),
    ('back hair', 'ParamHairBack', 15, 0.8, 0.85, 1.6),
]
HAIR_INPUTS = [('ParamAngleX', 'X', 60)]


def rig(project, meshes, parameters, size):
    """The moving parts.  Everything here is in canvas pixels at the rest pose."""
    face = drawn_box(meshes['face'])
    factor = (face[2] - face[0]) / rigkit.REFERENCE_FACE_WIDTH
    tuning = Tuning(lash_thickness=None).for_face(face[2] - face[0])
    for side in 'lr':
        rig_blink(project, meshes, parameters, side, size, tuning)
    rig_mouth(project, meshes, parameters, size)
    rig_pupils(project, meshes, parameters, tuning)

    group = lambda *names: [meshes[name] for name in names if name in meshes]
    # Pieces that artprep copied in front of what hid them move with the layer they came from,
    # except strands hanging in front of the hat: those go with the hat, so the hat never
    # slides from under them and shows the hidden hair a second time.
    hat = on_canvas(meshes['headwear'], size) > 128
    on_hat = {other for other in meshes if ' over' in other
              and hat[on_canvas(meshes[other], size) > 128].mean() > 0.5}
    copies = lambda name: [mesh for other, mesh in meshes.items()
                           if other.startswith(name + ' over') and other not in on_hat]

    # Hair sway.  Front: fringe and the locks beside the face are one piece of art, so one warp,
    # hanging from the top of the head.
    front = group('front hair') + copies('front hair')
    _, top, _, bottom = drawn_box(meshes['front hair'])
    rig_sway(project, parameters['ParamHairFront'], 'sw_front', 'WarpHairFront', front,
             FRONT_HAIR_SWAY * factor, top, bottom, cols=16, rows=16)
    # Back: still around the head, swinging freely below the chin.
    back = group('back hair') + copies('back hair')
    back_warp = rig_sway(project, parameters['ParamHairBack'], 'sw_back', 'WarpHairBack', back,
                         BACK_HAIR_SWAY * factor, face[3], max(drawn_box(mesh)[3] for mesh in back),
                         cols=14, rows=20, power=1.5)
    # The braid was cut from the back hair: it sits inside the back hair's warp, so where it comes
    # out of the hair at the shoulder the two always move together.  Its end is tied into the bow
    # at the waist, so both ends stay put and the middle swings.
    _, top, _, bottom = drawn_box(meshes['braid'])
    braid_warp = rig_sway(project, parameters['ParamHairSide'], 'sw_braid', 'WarpHairBraid', group('braid'),
                          BRAID_SWAY * factor, top, bottom, cols=4, rows=14, tied=True)
    braid_warp.parent = back_warp

    # Head turn: one warp per depth.
    head = Head(meshes, tuning)
    rig_turn(project, parameters, 'turn_face', 'WarpTurnFace',
             group('face', 'ears-l', 'ears-r', 'earwear', 'eyebrow-l', 'eyebrow-r', 'eyelash-l', 'eyelash-r',
                   'eyelid-l', 'eyelid-r', 'irides-l', 'irides-r', 'eyewhite-l', 'eyewhite-r',
                   'mouth line', 'mouth upper', 'mouth inner'),
             head.skin_shift, cols=12, rows=14)
    rig_turn(project, parameters, 'turn_nose', 'WarpTurnNose', group('nose'), head.nose_shift, cols=2, rows=2)
    rig_turn(project, parameters, 'turn_hair', 'WarpTurnHair', front, head.hair_shift, cols=12, rows=24)
    rig_turn(project, parameters, 'turn_hat', 'WarpTurnHat', group('headwear', *sorted(on_hat)),
             head.worn_shift, cols=8, rows=8)
    rig_turn(project, parameters, 'turn_back', 'WarpTurnBack', back + group('braid'), head.back_shift, cols=18, rows=20)
    rig_turn(project, parameters, 'turn_neck', 'WarpTurnNeck', group('neck') + copies('neck'), head.neck_shift,
             cols=3, rows=6)


def main():
    arguments = [a for a in sys.argv[1:] if a != '--install']
    out_dir = arguments[0] if arguments else config.out_dir(NAME)
    built = rigkit.build(NAME, PSD_PATH, out_dir, rig=rig, clipped_by=CLIPPED_BY, mesh_spacing=MESH_SPACING)
    hair_physics = physics.physics_file([
        physics.pendulum(f'PhysicsSetting{number}', name, output, length, delay, mobility, scale=scale, inputs=HAIR_INPUTS)
        for number, (name, output, length, delay, mobility, scale) in enumerate(HAIR_PHYSICS, 1)])
    rigkit.convert(built, physics_data=hair_physics, install='--install' in sys.argv)
    built.regions = rigkit.measure_regions(built.meshes, built.size)
    source = np.asarray(Image.open(SOURCE_PICTURE).convert('RGB').resize(built.size, Image.LANCZOS))

    rigkit.report_rest(built, PSD_POSE, source_picture=source)
    rigkit.report_positions(built)
    rigkit.report_mouth(built)
    rigkit.report_pupils(built)
    rigkit.report_blink(built)
    rigkit.report_turn(built)
    left, top, right, bottom = (int(v) for v in next(w for w in built.project.warps if w.name == 'sw_front').box)
    hair = rigkit.padded((left, top, right, bottom), 60, 40, built.size)
    figure = built.regions['figure']
    rigkit.report_sway(built, [('check_sway.jpg', 'ParamHairFront', 'Hair Front', hair, 1.0),
                               ('check_sway_side.jpg', 'ParamHairSide', 'Braid', figure, 0.6),
                               ('check_sway_back.jpg', 'ParamHairBack', 'Hair Back', figure, 0.6)],
                       lags={'ParamHairFront': 0.0, 'ParamHairSide': 0.5, 'ParamHairBack': 1.0})
    rigkit.report_physics(built, hair_physics)


if __name__ == '__main__':
    main()
