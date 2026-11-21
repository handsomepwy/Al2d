"""Rig recipes and checks shared by the character build scripts (characters/*/build.py).

A character script names its PSD, says which layers belong together, and calls the recipes:

    built = rigkit.build(name, psd_path, out_dir, rig=my_rig, ...)     # PSD -> .cmo3
    rigkit.convert(built, physics=..., install=...)                    # .cmo3 -> runtime files, loaded in Core
    rigkit.report_rest(built, ...), rigkit.report_blink(built), ...    # numbers and pictures to look at

Recipes (each measures the art it is given; lengths that cannot be measured come from a Tuning):
    rig_blink   rig_mouth   rig_pupils   rig_sway   Head + rig_turn

Layer names are See-through's, plus the ones artprep adds (eyelid-*, mouth inner / upper / line).
Layers ending in -l are the character's left (right on screen).
"""
import itertools
import os
import re
from dataclasses import dataclass, field, fields, replace

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from . import l2dcore, physics, pipeline
from .automesh import grid_mesh
from .cmo3writer import Project, Rotation, Warp, turn as turn_points

# (id, name, minimum, maximum, default, cursor tracking in the Editor's viewer)
STANDARD_PARAMETERS = [
    ('ParamAngleX', 'Angle X', -30, 30, 0, 'X'),
    ('ParamAngleY', 'Angle Y', -30, 30, 0, 'Y'),
    ('ParamAngleZ', 'Angle Z', -30, 30, 0, 'NONE'),
    ('ParamEyeLOpen', 'Eye L Open', 0, 1, 1, 'NONE'),
    ('ParamEyeLSmile', 'Eye L Smile', 0, 1, 0, 'NONE'),
    ('ParamEyeROpen', 'Eye R Open', 0, 1, 1, 'NONE'),
    ('ParamEyeRSmile', 'Eye R Smile', 0, 1, 0, 'NONE'),
    ('ParamEyeBallX', 'Eyeball X', -1, 1, 0, 'X'),
    ('ParamEyeBallY', 'Eyeball Y', -1, 1, 0, 'Y'),
    ('ParamBrowLY', 'Brow L Y', -1, 1, 0, 'NONE'),
    ('ParamBrowRY', 'Brow R Y', -1, 1, 0, 'NONE'),
    ('ParamMouthForm', 'Mouth Form', -1, 1, 0, 'NONE'),
    ('ParamMouthOpenY', 'Mouth Open', 0, 1, 0, 'NONE'),
    ('ParamBodyAngleX', 'Body Angle X', -10, 10, 0, 'X'),
    ('ParamBodyAngleY', 'Body Angle Y', -10, 10, 0, 'NONE'),
    ('ParamBodyAngleZ', 'Body Angle Z', -10, 10, 0, 'NONE'),
    ('ParamBreath', 'Breath', 0, 1, 0, 'NONE'),
    ('ParamHairFront', 'Hair Front', -1, 1, 0, 'NONE'),
    ('ParamHairSide', 'Hair Side', -1, 1, 0, 'NONE'),
    ('ParamHairBack', 'Hair Back', -1, 1, 0, 'NONE'),
]


@dataclass(frozen=True)
class Tuning:
    """The numbers the recipes cannot measure from the art.  Lengths are pixels for a face
    REFERENCE_FACE_WIDTH wide (the first character); for_face() scales them to another face."""
    lower_lid_share: float = 0.3        # share of the eye opening the lower lid closes; the lashes do the rest
    blink_overlap: float = 2            # pixels the lashes go past the lower lid, so no eye shows between them
    # pixels above the visible eye that move down undeformed (lashes + crease); None = measure from the art
    lash_thickness: float = 14
    tail_keep: float = 0.5              # closed eye: how much of the eyeliner tail's length below the lash line stays
    pupil_reach: tuple = (8, 5)         # pixels the irises move sideways / up and down at ParamEyeBallX / Y = +-1
    # Head turn.  Each pair is the pixels a point moves at ParamAngleX = 30 (towards screen right)
    # and at ParamAngleY = 30 (up).  The face is treated as a dome: its middle moves most, its outline least.
    turn_face_middle: tuple = (46, 26)
    turn_face_edge: tuple = (10, 6)
    turn_squeeze: float = 0.045         # at a full sideways turn the face is this much narrower (turned away from us)
    turn_nose_extra: tuple = (12, 7)    # the nose sticks out: this much more than the skin under it
    turn_hair: tuple = (48, 27)         # front hair over the middle of the forehead; nothing at the hair's outline
    turn_back: tuple = (-10, -5)        # the hair behind the head, at chin height: the other way, for depth
    turn_neck: float = 8                # the top of the neck leans after the chin (sideways only)
    hair_hang: float = 380              # long locks follow the head down to the chin, and not at all this far below it
    # Where the up/down turn moves most: 0 = at eye level, 1 = at the chin.  Live2D's Haru moves
    # the mouth more than the eyes when looking up (the head pivots on the neck), about 0.5.
    turn_y_centre: float = 0.0
    turn_behind: tuple = (-14, -11)     # what hangs behind the head (a ponytail) moves the other way, as one piece
    # Expressions.  Shares of the brow's width, the lip line's width or the eye opening's height,
    # fitted to Live2D's Haru (ref_haru/analysis/expr_fit.py), so they need no scaling.
    brow_move: tuple = (0.14, 0.14)     # ParamBrow*Y = +1 moves the brow up, ParamBrow*X = +1 outwards
    brow_angle: float = 27.0            # degrees at ParamBrow*Angle = +-1; +1 lifts the inner end
    # ParamBrow*Form: (key, lift, bend): every point moves down by lift (negative = up) and the middle
    # by bend more than the ends.
    brow_form: tuple = ((-1, -0.195, 0.22), (1, -0.13, -0.08))
    # ParamMouthForm: (key, width, lift, corners), each as (closed, open): the mouth is narrowed
    # to `width` of itself, moved down by lift and its corners down by `corners` more than its
    # middle.  The key with (1, 0, 0) is the mouth as drawn (Haru's PSD has the smile, form 1).
    mouth_forms: tuple = ((-1, (0.73, 0.59), (0.078, 0.049), (0.153, 0.160)),
                          (0, (0.763, 0.794), (0.027, 0.035), (0.105, 0.056)),
                          (1, (1.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    eye_smile_close: float = 0.42       # closed and smiling: the lash line comes down this share, keeping its arch
    eye_smile_lift: float = 0.32        # and the lower lashes come up this share of the opening
    # Head tilt (ParamAngleZ), from Live2D's Haru (her rig turns the head only this little; the body
    # does not lean with it): degrees at ParamAngleZ = +-30 (+ = clockwise on screen), and the pivot
    # this share of the face's height above the chin.
    tilt_angle: float = 5.6
    tilt_pivot: float = 0.09
    # The neck bends to follow: (below, span) as shares of the face's height.  None of it moves from
    # `below` under the chin down; going up it follows more and more, fully `span` higher.
    tilt_neck: tuple = (0.1, 0.47)

    LENGTHS = ('blink_overlap', 'lash_thickness', 'pupil_reach', 'turn_face_middle', 'turn_face_edge',
               'turn_nose_extra', 'turn_hair', 'turn_back', 'turn_neck', 'hair_hang', 'turn_behind')

    def for_face(self, face_width):
        factor = face_width / REFERENCE_FACE_WIDTH
        scale = lambda value: None if value is None else \
            tuple(v * factor for v in value) if isinstance(value, tuple) else value * factor
        return replace(self, **{name: scale(getattr(self, name)) for name in self.LENGTHS})


REFERENCE_FACE_WIDTH = 312.0


def drawable_id(layer_name):
    """'front hair side-l' -> 'FrontHairSideL' (ids may only hold letters, digits, underscore)."""
    return ''.join(word.capitalize() for word in re.split(r'[^A-Za-z0-9]+', layer_name) if word)


def read_layers(psd_path):
    """Visible pixel layers of the PSD, front to back, as (name, RGBA crop, (x, y))."""
    psd = PSDImage.open(psd_path)
    layers = []
    for layer in reversed(list(psd)):            # psd_tools lists back to front
        if not layer.visible or layer.kind != 'pixel':
            continue
        image = np.asarray(layer.topil().convert('RGBA'))
        layers.append((layer.name, image, (layer.left, layer.top)))
    return psd.size, layers


def reference_picture(size, layers, clipped_in_psd=None, skip=()):
    """The layers flattened back to front (clipped layers masked), over white: the target look."""
    clipped_in_psd = clipped_in_psd or {}
    width, height = size
    colour = np.zeros((height, width, 3), np.float32)
    alpha = np.zeros((height, width), np.float32)
    full_alpha = {}
    for name, image, (x, y) in layers:
        plane = np.zeros((height, width), np.float32)
        plane[y:y + image.shape[0], x:x + image.shape[1]] = image[..., 3] / 255.0
        full_alpha[name] = plane
    for name, image, (x, y) in reversed(layers):
        if name in skip:
            continue
        h, w = image.shape[:2]
        source_alpha = image[..., 3].astype(np.float32) / 255.0
        if name in clipped_in_psd:
            source_alpha = source_alpha * full_alpha[clipped_in_psd[name]][y:y + h, x:x + w]
        source_alpha = source_alpha[..., None]
        target = colour[y:y + h, x:x + w]
        target *= 1 - source_alpha
        target += image[..., :3].astype(np.float32) * source_alpha
        alpha[y:y + h, x:x + w] = source_alpha[..., 0] + alpha[y:y + h, x:x + w] * (1 - source_alpha[..., 0])
    return np.clip(colour + 255.0 * (1 - alpha[..., None]) + 0.5, 0, 255).astype(np.uint8)


def box_around(vertices, margin=0.05):
    """A box holding the points with a margin on every side, as (left, top, right, bottom).

    The Editor leaves about 5 % too; a point on or outside the edge of its warp deforms badly.
    """
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    pad = (high - low) * margin / (1 - 2 * margin)
    return (*(low - pad), *(high + pad))


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def drawn_box(mesh, threshold=40):
    """(left, top, right, bottom) of the layer's visible pixels, in canvas pixels."""
    rows, columns = np.nonzero(mesh.image[..., 3] > threshold)
    return (mesh.origin[0] + columns.min(), mesh.origin[1] + rows.min(),
            mesh.origin[0] + columns.max(), mesh.origin[1] + rows.max())


def full_canvas(mesh, size):
    """The layer as a full-canvas RGBA array."""
    canvas = np.zeros((size[1], size[0], 4), np.uint8)
    height, width = mesh.image.shape[:2]
    canvas[mesh.origin[1]:mesh.origin[1] + height, mesh.origin[0]:mesh.origin[0] + width] = mesh.image
    return canvas


def on_canvas(mesh, size):
    """The layer's alpha as a full-canvas (height, width) array."""
    plane = np.zeros((size[1], size[0]), np.uint8)
    height, width = mesh.image.shape[:2]
    plane[mesh.origin[1]:mesh.origin[1] + height, mesh.origin[0]:mesh.origin[0] + width] = mesh.image[..., 3]
    return plane


# ---------------------------------------------------------------- recipes

def hang_offsets(warp, amount, top, bottom, power=2.0, hold=None, ramp=50.0, tied=False):
    """Grid offsets for hanging hair pushed sideways: nothing moves at the height `top` or above,
    the height `bottom` and everything lower moves `amount` pixels, with a curve in between.

    hold = (left, right): columns outside stay still; the movement fades in over `ramp` pixels.
    tied: the lower end is held too (a braid tied at the waist); the middle moves `amount`.
    """
    x, y = warp.rest[..., 0], warp.rest[..., 1]
    depth = np.clip((y - top) / (bottom - top), 0.0, 1.0)
    depth = np.sin(np.pi * depth) if tied else depth ** power
    fade = 1.0 if hold is None else smoothstep(np.minimum(x - hold[0], hold[1] - x) / ramp)
    offsets = np.zeros_like(warp.rest)
    offsets[..., 0] = amount * depth * fade
    return offsets


def rig_sway(project, parameter, name, deformer_id, members, amount, top, bottom, cols, rows, **shape):
    """One warp around the meshes `members` that swings them with a sway parameter (-1, 0, 1)."""
    warp = project.add_warp(name, deformer_id, box_around(np.concatenate([mesh.vertices for mesh in members])),
                            cols=cols, rows=rows)
    warp.bind(parameter, [-1, 0, 1])
    warp.set_form(0, warp.rest + hang_offsets(warp, -amount, top, bottom, **shape))
    warp.set_form(2, warp.rest + hang_offsets(warp, amount, top, bottom, **shape))
    for mesh in members:
        mesh.parent = warp
    return warp


def rig_blink(project, meshes, parameters, side, size, tuning=Tuning()):
    """One eye's blink: the lash layer comes down, the lower lid comes up, they meet in between.

    The lash layer carries a block of lid skin above the lashes and the lower-lid layer is a strip
    of skin under the eye, so the eye is covered without squashing the iris.  How far each grid
    column travels is measured from the art: the height of the visible eye in that column.
    Layers ending in -l are the character's left eye (right on screen) and follow ParamEyeLOpen.
    """
    lash, lid = meshes[f'eyelash-{side}'], meshes[f'eyelid-{side}']
    eye = (on_canvas(meshes[f'eyewhite-{side}'], size) > 40) | (on_canvas(meshes[f'irides-{side}'], size) > 40)
    visible = eye & (on_canvas(lash, size) < 128) & (on_canvas(lid, size) < 128)
    columns = np.nonzero(visible.any(axis=0))[0]
    top = np.array([np.nonzero(visible[:, x])[0].min() for x in columns])
    bottom = np.array([np.nonzero(visible[:, x])[0].max() for x in columns])
    opening = np.zeros(size[0])
    opening[columns] = bottom - top + 1
    # Widen before smoothing, so the smoothed curve never falls short of the real opening
    # (falling short leaves a sliver of eye showing at the corners).
    opening = gaussian_filter1d(maximum_filter1d(opening, 9), 3)
    lash_drop = opening * (1 - tuning.lower_lid_share) + tuning.blink_overlap * np.clip(opening / 6, 0, 1)
    lid_rise = opening * tuning.lower_lid_share
    parameter = parameters['ParamEyeLOpen' if side == 'l' else 'ParamEyeROpen']
    canvas_x = np.arange(size[0])

    # Upper: the top row is pinned so the skin block stretches instead of sliding over the face;
    # everything from a little above the lashes downwards moves the whole way, unsquashed.
    upper = project.add_warp(f'bl_lash_{side}', f'WarpBlinkLash{side.upper()}', box_around(lash.vertices),
                             cols=10, rows=12)
    x, y = upper.rest[..., 0], upper.rest[..., 1]
    if tuning.lash_thickness is None:
        # Measured: the highest drawn stroke of the lash layer (lashes, crease) near the eye.  What
        # the layer has above that is the added lid skin, which looks the same as the face under it.
        ours, under = full_canvas(lash, size), full_canvas(meshes['face'], size)
        stroke = (ours[..., 3] > 128) & (np.abs(ours[..., :3].astype(int) - under[..., :3].astype(int)).max(axis=2) > 24)
        stroke[:, :columns.min()] = False
        stroke[:, columns.max() + 1:] = False
        whole_from = np.nonzero(stroke.any(axis=1))[0].min() - 2
    else:
        whole_from = top.min() - tuning.lash_thickness
    share = np.clip((y - upper.box[1]) / (whole_from - upper.box[1]), 0, 1)
    closed = upper.rest.copy()
    closed[..., 1] += np.interp(x, canvas_x, lash_drop) * share
    # The eyeliner's outer tail hangs below the closed line like a spike; pull it up towards the line.
    line = np.interp(x, columns, top) + np.interp(x, canvas_x, lash_drop)
    below = np.maximum(closed[..., 1] - (line + 3), 0)
    closed[..., 1] -= below * (1 - tuning.tail_keep)
    upper.bind(parameter, [0, 1])
    upper.set_form(0, closed)
    lash.parent = upper

    # Lower: the bottom row is pinned; the strip's top edge rises the whole way.
    lower = project.add_warp(f'bl_lid_{side}', f'WarpBlinkLid{side.upper()}', box_around(lid.vertices),
                             cols=10, rows=5)
    x, y = lower.rest[..., 0], lower.rest[..., 1]
    whole_to = bottom.max() + 6
    share = np.clip((lower.box[3] - y) / (lower.box[3] - whole_to), 0, 1)
    closed = lower.rest.copy()
    closed[..., 1] -= np.interp(x, canvas_x, lid_rise) * share
    lower.bind(parameter, [0, 1])
    lower.set_form(0, closed)
    lid.parent = lower


def columns_of(plane):
    """For an (height, width) bool plane: the columns that have pixels, and each one's top and bottom row."""
    columns = np.nonzero(plane.any(axis=0))[0]
    top = np.array([np.nonzero(plane[:, x])[0].min() for x in columns], np.float64)
    bottom = np.array([np.nonzero(plane[:, x])[0].max() for x in columns], np.float64)
    return columns, top, bottom


def add_axis(keyed, parameter, keys, shape_at):
    """Key an object on one more parameter.  For every key combination it already has and every
    new key, shape_at(shape, key, values) gives the new form from the old one; values are the
    old combination's key values, one per earlier binding."""
    old = [(index, keyed.form_at(index), tuple(keys_[i] for (_, keys_), i in zip(keyed.bindings, index)))
           for index in keyed.key_indices()]
    keyed.bind(parameter, keys)
    keyed.forms = {}
    for index, shape, values in old:
        for k, key in enumerate(keys):
            keyed.forms[index + (k,)] = np.asarray(shape_at(np.asarray(shape, np.float64), key, values), np.float32)


def rig_blink_masked(project, meshes, parameters, side, size, upper, mask, close_at=0.70, spread=3, corner_share=0.56,
                     lower=None, sides=None, thickness=(0.65, 0.85), side_thickness=0.2, tuning=Tuning()):
    """One eye's blink, the way an artist's PSD is drawn for Live2D: an invisible `mask` layer is
    the shape of the eye opening, and the eye white, iris and highlights are clipped by it.

    Closed, the mask is squashed down onto the opening's lower edge, so everything it clips is
    gone, and the upper lash line comes down `close_at` of the opening's height.  Between the
    closed lash line and the lower lashes the face shows: no lid skin is needed.

    upper  {mesh: share}: what comes down with the lid and how much of the way (1 = all of it).
           Each piece moves straight down by an amount that depends only on the column (as in
           Live2D's rig of Haru); since the drop is largest in the middle, the open eye's arch
           bends into the closed eye's downward curve.  In each column the lash line also thins,
           to `thickness` (closed, closed and smiling) of its height: Haru's closed lash line is
           about 0.65 as thick as the open one.  The pieces of the line (share 0.8 or more) thin
           towards the middle row of all of them together, so pieces lying on each other stay
           together.  Other pieces (the crease) move whole, by their share of the middle's drop.
    sides  {mesh: share}: strokes that run down the side of the opening (Haru's outer corner
           stroke).  They come down with the line and thin towards its middle to side_thickness,
           so a stroke that runs down becomes a short dash inside the closed line.
    corner_share  the drop never falls below this share of the drop in the middle, even past
           the ends of the opening, so the wing at the outer corner comes down with the line
           instead of being left behind and stretched.  close_at 0.70 and corner_share 0.56
           match Haru's rig: the middle of her lash line comes down 63 px, its ends 40-47.
    lower  {mesh: share}: the lower lashes.  Closed, they tuck up under the lash line.  With ParamEye*Smile among the parameters, the eye
           closed and smiling is the happy arch: the lash line comes down only
           tuning.eye_smile_close of the opening, the same everywhere so it keeps its arch, and
           the lower lashes come up tuning.eye_smile_lift.  Open, the smile changes nothing (as
           in Haru); in between it blends.
    """
    columns, top, bottom = columns_of(on_canvas(mask, size) > 128)
    canvas_x = np.arange(size[0])
    opening = np.interp(canvas_x, columns, gaussian_filter1d(maximum_filter1d(bottom - top, spread), spread / 3))
    height = opening.max()
    drop = close_at * np.maximum(opening, corner_share * height)
    upper = {**upper, **(sides or {})}
    thin = {mesh: (side_thickness, side_thickness) if mesh in (sides or {}) else thickness for mesh in upper}
    parameter = parameters['ParamEyeLOpen' if side == 'l' else 'ParamEyeROpen']
    upper_edge = lambda x: np.interp(x, columns, top)
    lower_edge = lambda x: np.interp(x, columns, bottom)

    def middle_row(plane):
        """The middle row of the drawn pixels in each canvas column, as a function of x."""
        piece_columns, piece_top, piece_bottom = columns_of(plane)
        middle = gaussian_filter1d((piece_top + piece_bottom) / 2, 2)
        return lambda x: np.interp(x, piece_columns, middle)

    line_pieces = [mesh for mesh, share in upper.items() if share >= 0.8 and mesh not in (sides or {})]
    line_middle = middle_row(np.any([on_canvas(mesh, size) > 40 for mesh in line_pieces], axis=0))
    thin_sides = set(sides or {})
    centre_of = {mesh: line_middle for mesh in line_pieces + list(thin_sides)}

    def lash_closed(mesh, share, drop=drop, smiling=False):
        points = mesh.vertices.astype(np.float64)
        x, y = points[:, 0], points[:, 1]
        if mesh not in line_pieces and mesh not in thin_sides:
            # Not part of the lash line (the crease): it moves as one piece, its share of the
            # middle's drop (Haru's crease comes down 25-28 px, a little under half the line's 63).
            return np.stack([x, y + np.max(drop) * share], axis=1)
        centre = centre_of[mesh](x)
        k = thin[mesh][1 if smiling else 0]
        return np.stack([x, centre + (y - centre) * k + np.interp(x, canvas_x, drop) * share], axis=1)

    def mask_closed(points):
        x, y = points[:, 0], points[:, 1]
        down = (lower_edge(x) - upper_edge(x)) * np.clip((lower_edge(x) - y) / np.maximum(lower_edge(x) - upper_edge(x), 1), 0, 1)
        return points + np.stack([np.zeros_like(y), down], axis=1)

    for mesh, share in upper.items():
        mesh.bind(parameter, [0, 1])
        mesh.set_form(0, lash_closed(mesh, share).astype(np.float32))
    mask.bind(parameter, [0, 1])
    mask.set_form(0, mask_closed(mask.vertices.astype(np.float64)).astype(np.float32))
    # The lower lashes tuck up under the closed line where they would show below it, only in
    # the last quarter of the blink (earlier they would cross the iris).
    closed_line = lambda x: line_middle(x) + np.interp(x, canvas_x, drop)
    for mesh, share in (lower or {}).items():
        points = mesh.vertices.astype(np.float64)
        below = np.maximum(middle_row(on_canvas(mesh, size) > 40)(points[:, 0]) - closed_line(points[:, 0]), 0)
        mesh.bind(parameter, [0, 0.25, 1])
        mesh.set_form(0, (points - np.stack([np.zeros(len(points)), below], axis=1)).astype(np.float32))

    smile = parameters.get(f'ParamEye{side.upper()}Smile')
    if smile is not None:
        smile_drop = np.full(size[0], tuning.eye_smile_close * height)
        for mesh, share in upper.items():
            add_axis(mesh, smile, [0, 1], lambda shape, key, values, mesh=mesh, share=share:
                     lash_closed(mesh, share, smile_drop, smiling=True) if key and values[0] == 0 else shape)
        # Closed and smiling, the mask collapses onto the line where the risen lower lashes are,
        # so in between the visible eye ends at the lower lashes (at rest the opening reaches a
        # little below them).  Where there are no lower lashes, its lower edge rises by the same
        # share of its height in every column, so the corners stay.
        between = canvas_x[(canvas_x >= columns[0]) & (canvas_x <= columns[-1])]
        target = lower_edge(between) - tuning.eye_smile_lift * (lower_edge(between) - upper_edge(between))
        lashes_at = np.full(len(between), np.inf)
        for mesh, share in (lower or {}).items():
            lash_columns, lash_top, lash_bottom = columns_of(on_canvas(mesh, size) > 128)
            risen = (lash_top + lash_bottom) / 2 - tuning.eye_smile_lift * height * share
            inside = np.isin(between, lash_columns)
            lashes_at[inside] = np.minimum(lashes_at[inside], np.interp(between[inside], lash_columns, risen))
        target = gaussian_filter1d(-maximum_filter1d(-np.minimum(target, lashes_at), 5), 2)

        def mask_smiled(points):
            return np.stack([points[:, 0], np.interp(points[:, 0], between, target)], axis=1)

        add_axis(mask, smile, [0, 1], lambda shape, key, values:
                 mask_smiled(mask.vertices.astype(np.float64)) if key and values[0] == 0 else shape)
        for mesh, share in (lower or {}).items():
            rise = np.float64([0, -tuning.eye_smile_lift * height * share])
            # As if keyed on open 0 and 1 only, so they keep pace with the mask's lower edge.
            add_axis(mesh, smile, [0, 1], lambda shape, key, values, mesh=mesh, rise=rise:
                     mesh.vertices.astype(np.float64) + rise * (1 - values[0]) if key else shape)


def dark_line(mesh, size):
    """The dark drawn line in a layer (a lip line): its columns and the middle row in each."""
    image = full_canvas(mesh, size)
    dark = (image[..., 3] > 128) & (image[..., :3].astype(int).sum(axis=2) < 300)
    line_columns, line_top, line_bottom = columns_of(dark)
    return line_columns, (line_top + line_bottom) / 2


def rig_mouth_form(project, meshes, parameters, size, tuning=Tuning(), names=('mouth upper', 'mouth lower', 'mouth inner'),
                   line='mouth upper'):
    """ParamMouthForm: the smile (+1) and the frown (-1).  Call after the mouth's open/close recipe.

    Every point of the mouth layers moves by one map, so whatever covered the interior when
    closed still covers it: the mouth narrows towards its middle, moves down, and its corners
    drop more than its middle (tuning.mouth_forms, measured on Haru, as shares of the lip line's
    width).  How much depends on how open the mouth is, so each layer is keyed on
    ParamMouthOpenY x ParamMouthForm.
    """
    columns, _ = dark_line(meshes[line], size)
    cx, half = (columns.min() + columns.max()) / 2, (columns.max() - columns.min()) / 2
    width = 2 * half
    opening, form = parameters['ParamMouthOpenY'], parameters['ParamMouthForm']
    table = {key: (w, lift, corners) for key, w, lift, corners in tuning.mouth_forms}

    def shaped(shape, key, values):
        opened = values[0] if values else 0.0
        mix = lambda pair: pair[0] + (pair[1] - pair[0]) * opened
        w, lift, corners = (mix(pair) for pair in table[key])
        x, y = shape[:, 0], shape[:, 1]
        u = np.clip((x - cx) / half, -1, 1)
        return np.stack([cx + (x - cx) * w, y + (lift + corners * u * u) * width], axis=1)

    for name in names:
        mesh = meshes[name]
        if not any(parameter is opening for parameter, _ in mesh.bindings):
            mesh.bind(opening, [0, 1])          # same shape open and closed, but formed per opening
        assert mesh.bindings[0][0] is opening and len(mesh.bindings) == 1
        add_axis(mesh, form, sorted(table), shaped)


def rig_brows(project, meshes, parameters, tuning=Tuning()):
    """The brows: ParamBrow{L,R}Y (up / down), and when the character has them X (outwards /
    inwards), Angle (+1 lifts the inner end) and Form (tuning.brow_form).  Each brow keeps its
    shape apart from the form's bend: it is bent, then turned about its middle, then moved.
    Lengths are shares of the brow's width (Haru's, tuning.brow_*)."""
    for side in 'lr':
        brow = meshes[f'eyebrow-{side}']
        left, top, right, bottom = drawn_box(brow)
        width, middle = right - left, np.float64([(left + right) / 2, (top + bottom) / 2])
        outwards = 1.0 if side == 'l' else -1.0          # her left brow is on the right of the screen
        turn_sign = 1.0 if side == 'l' else -1.0         # + angle on screen (clockwise) lifts the left brow's inner end
        name = lambda what: f'ParamBrow{side.upper()}{what}'
        forms = {key: (lift, bend) for key, lift, bend in tuning.brow_form}

        def bent(shape, key, values):
            if key == 0:
                return shape
            lift, bend = forms[key]
            u = np.clip((shape[:, 0] - middle[0]) / (width / 2), -1, 1)
            return shape + np.stack([np.zeros(len(shape)), (lift + bend * (1 - u * u)) * width], axis=1)

        def turned(shape, key, values):
            angle = np.radians(tuning.brow_angle * key * turn_sign)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            return (shape - middle) @ rotation.T + middle

        steps = [('Form', [-1, 0, 1], bent), ('Angle', [-1, 0, 1], turned),
                 ('X', [-1, 0, 1], lambda shape, key, values: shape + [key * tuning.brow_move[1] * width * outwards, 0]),
                 ('Y', [-1, 0, 1], lambda shape, key, values: shape + [0, -key * tuning.brow_move[0] * width])]
        for what, keys, shape_at in steps:
            if name(what) in parameters:
                add_axis(brow, parameters[name(what)], keys, shape_at)


def rig_mouth_patches(project, meshes, parameters, size, inner='mouth inner', upper='mouth upper',
                      lower='mouth lower', upper_share=1.0, lower_share=1.0):
    """Mouth open/close, the way an artist's PSD is drawn: the open mouth (`inner`) is drawn whole,
    and at rest two skin patches carrying the lip line hide it, `upper` covering its top half and
    `lower` its bottom half; their lines meet in the closed mouth.  Opening moves the upper patch's
    line up to the top edge of the inner drawing and the lower patch's line down to its bottom edge,
    column by column, so the lips frame the open mouth exactly.
    """
    columns, inner_top, inner_bottom = columns_of(on_canvas(meshes[inner], size) > 128)

    for name, edge, share in ((upper, inner_top, upper_share), (lower, inner_bottom, lower_share)):
        line_columns, line = dark_line(meshes[name], size)
        # How far the line moves in each column: to the inner drawing's edge, nothing past its ends.
        move = edge - np.interp(columns, line_columns, line)
        move[0] = move[-1] = 0.0
        mesh = meshes[name]
        opened = mesh.vertices.astype(np.float64).copy()
        opened[:, 1] += np.interp(opened[:, 0], columns, move, left=0.0, right=0.0) * share
        mesh.bind(parameters['ParamMouthOpenY'], [0, 1])
        mesh.set_form(1, opened.astype(np.float32))


def rig_mouth(project, meshes, parameters, size):
    """Mouth open/close.  The PSD has the mouth open: `mouth inner` is the drawn interior, a
    crescent hanging from the lip line.  Closed, it is flattened up onto the lip line, where the
    skin cover (`mouth upper`) and the lip line layer hide it.
    """
    inner = meshes['mouth inner']
    drawn = on_canvas(inner, size) > 128
    columns = np.nonzero(drawn.any(axis=0))[0]
    lip = np.array([np.nonzero(drawn[:, x])[0].min() for x in columns])     # top edge = the lip line
    closed = inner.vertices.copy()
    closed[:, 1] = np.interp(inner.vertices[:, 0], columns, lip) - 2
    inner.bind(parameters['ParamMouthOpenY'], [0, 1])
    inner.set_form(0, closed)


def rig_pupils(project, meshes, parameters, tuning=Tuning(), followers=None):
    """The irises follow ParamEyeBallX (+1 = towards screen right) and ParamEyeBallY (+1 = up).

    They sit behind the lashes and both lids, which hide whatever leaves the eye opening.
    followers  {'l': [layer names], 'r': [...]}: more layers that move with each iris (highlights).
    """
    reach_x, reach_y = tuning.pupil_reach
    for side in 'lr':
        for name in [f'irides-{side}'] + list((followers or {}).get(side, [])):
            iris = meshes[name]
            iris.bind(parameters['ParamEyeBallX'], [-1, 0, 1])
            iris.bind(parameters['ParamEyeBallY'], [-1, 0, 1])
            for index in iris.key_indices():
                offset = np.float32([(index[0] - 1) * reach_x, -(index[1] - 1) * reach_y])
                iris.set_form(index, iris.vertices + offset)


class Head:
    """Where the head is in the picture, and how far each spot moves when it turns.

    Each *_shift(points, x, y) gives the offsets of canvas points for a turn of x and y, both
    -1..1 (ParamAngleX / 30 and ParamAngleY / 30).  Layers that use the same shift move as one
    sheet, so nothing between them opens up; different shifts slide over each other, which is
    what makes the turn look deep.
    """

    def __init__(self, meshes, tuning=Tuning(), neck_bottom=None, hair_pinned=True):
        """hair_pinned: the front hair's outline stays put (See-through cut the front hair out of
        one mass of hair, so its outer edge must not slide over the hair behind it).  False for an
        artist's PSD, where the front hair is a whole piece of its own: it moves like the face."""
        self.tuning = tuning
        self.hair_pinned = hair_pinned
        left, self.face_top, right, self.chin = drawn_box(meshes['face'])
        eyes = [drawn_box(meshes[name]) for name in ('eyewhite-l', 'eyewhite-r')]
        eye_level = np.mean([(box[1] + box[3]) / 2 for box in eyes])
        self.centre = np.float64([(left + right) / 2, eye_level])
        self.radius = np.float64([(right - left) / 2 * 1.06, (self.chin - eye_level) * 1.1])
        self.y_centre = eye_level + tuning.turn_y_centre * (self.chin - eye_level)
        left, self.hair_top, right, _ = drawn_box(meshes['front hair'])
        # Exactly the drawn width and top: the front layers were cut out of one mass of hair, and
        # along the cut (their outer edge) they must not move against the hair behind them.
        self.hair_centre = np.float64([(left + right) / 2, (self.hair_top + self.chin) / 2])
        self.hair_radius = np.float64([(right - left) / 2, (self.chin - self.hair_top) / 2])
        # Where the neck goes into the collar (by default the bottom of the neck layer).
        self.neck_bottom = drawn_box(meshes['neck'])[3] if neck_bottom is None else neck_bottom

    def skin_shift(self, points, x, y, extra=(0, 0)):
        """The face and everything painted on it.

        Sideways the dome is flat across the eyes and falls off in the cheeks (power 4), and the
        whole face gets a little narrower.  Together: at a full turn the far eye is about 85 %
        as wide and the near eye 105 %, which is what Cubism Editor's own face-turn tool gave the
        hand-made rig (85 % / 106 %).  A plain round dome made it 80 % / 122 %, which looked squashed.
        """
        points = np.asarray(points, np.float64)
        r = (points - self.centre) / self.radius
        sideways = np.clip(1 - r[:, 0] ** 4 - 0.5 * r[:, 1] ** 2, 0, 1)
        r_up = (points[:, 1] - self.y_centre) / self.radius[1]
        if self.tuning.turn_y_centre:       # above a lowered centre the face still moves nearly as much
            r_up = np.where(r_up < 0, 0.5 * r_up, r_up)
        upwards = np.clip(1 - 0.5 * r[:, 0] ** 2 - r_up ** 2, 0, 1)
        edge, middle = self.tuning.turn_face_edge, self.tuning.turn_face_middle
        squeeze = -abs(x) * self.tuning.turn_squeeze * (points[:, 0] - self.centre[0])
        return np.stack([x * (edge[0] + (middle[0] - edge[0]) * sideways + extra[0]) + squeeze,
                         -y * (edge[1] + (middle[1] - edge[1]) * upwards + extra[1])], axis=1)

    def nose_shift(self, points, x, y):
        return self.skin_shift(points, x, y, extra=self.tuning.turn_nose_extra)

    def hair_shift(self, points, x, y):
        """Hair lying on the front of the head; what hangs below the chin follows less and less."""
        points = np.asarray(points, np.float64)
        r = (points - self.hair_centre) / self.hair_radius
        above = np.minimum(r[:, 1], 0)          # below the middle the head is as wide as at the middle
        if self.hair_pinned:
            sideways = np.clip(1 - np.abs(r[:, 0]) ** 3 - above ** 2, 0, 1)    # between round and the skin's
            upwards = np.clip(1 - r[:, 0] ** 4 - above ** 2, 0, 1)
        else:                                   # a whole piece: it slides with the face, outline and all
            sideways = np.clip(1 - 0.5 * r[:, 0] ** 2 - 0.3 * above ** 2, 0, 1)
            upwards = np.clip(1 - 0.5 * r[:, 0] ** 2 - 0.3 * above ** 2, 0, 1)
        hang = 1 - smoothstep((points[:, 1] - self.chin) / self.tuning.hair_hang)
        turn = self.tuning.turn_hair
        return np.stack([x * turn[0] * sideways * hang, -y * turn[1] * upwards * hang], axis=1)

    def back_shift(self, points, x, y):
        """Hair behind the head: swings the other way around chin height.  Still at the top and
        from the outer edge of the front hair outwards, so the outline of the hair never moves."""
        points = np.asarray(points, np.float64)
        height = points[:, 1]
        inside = np.clip(1 - ((points[:, 0] - self.hair_centre[0]) / self.hair_radius[0]) ** 2, 0, 1)
        swing = smoothstep((height - self.hair_top) / (self.chin - self.hair_top)) \
            * (1 - smoothstep((height - self.chin) / self.tuning.hair_hang)) * inside
        turn = self.tuning.turn_back
        return np.stack([x * turn[0] * swing, -y * turn[1] * swing], axis=1)

    def behind_shift(self, points, x, y):
        """What hangs behind the head (a ponytail): it moves the other way, all of it alike."""
        behind = self.tuning.turn_behind
        return np.tile([x * behind[0], -y * behind[1]], (len(points), 1))

    def worn_shift(self, points, x, y):
        """Something worn on the head (a hat): it wraps around the head like the hair does, so
        the middle moves most and whatever hangs at the outline (a ribbon) hardly moves.  Unlike
        the hair it does not stand still at the top: it sits on the head, not inside its outline."""
        points = np.asarray(points, np.float64)
        r = (points - self.hair_centre) / self.hair_radius
        sideways = np.clip(1 - np.abs(r[:, 0]) ** 3, 0, 1)
        upwards = np.clip(1 - r[:, 0] ** 4, 0, 1)
        turn = self.tuning.turn_hair
        return np.stack([x * turn[0] * sideways, -y * turn[1] * upwards], axis=1)

    def neck_shift(self, points, x, y):
        """The neck leans after the chin; its bottom stays in the collar."""
        height = np.asarray(points, np.float64)[:, 1]
        lean = np.clip((self.neck_bottom - height) / (self.neck_bottom - self.chin), 0, 1)
        return np.stack([x * self.tuning.turn_neck * lean, np.zeros_like(lean)], axis=1)


def rig_turn(project, parameters, name, deformer_id, members, shift, cols, rows):
    """One warp around `members` (meshes, or the warps they already sit in) that follows the head turn."""
    tops = []
    for member in members:
        while member.parent is not None:
            member = member.parent
        if member not in tops:
            tops.append(member)
    points = np.concatenate([np.reshape(top.rest_shape(), (-1, 2)) for top in tops])
    warp = project.add_warp(name, deformer_id, box_around(points), cols=cols, rows=rows)
    warp.bind(parameters['ParamAngleX'], [-30, 0, 30])
    warp.bind(parameters['ParamAngleY'], [-30, 0, 30])
    for index in warp.key_indices():
        offsets = shift(warp.rest.reshape(-1, 2), index[0] - 1, index[1] - 1)
        warp.set_form(index, warp.rest + offsets.reshape(warp.rest.shape))
    for top in tops:
        top.parent = warp
    return warp


def rig_tilt(project, parameters, members, head, neck=None):
    """Head tilt (ParamAngleZ): one rotation deformer around everything that turns with the head
    (`members`: the turn warps, or anything inside them), turning about a point just above the chin.

    neck: the neck's warp (outside the rotation); it bends after the head, its bottom stays put.
    Returns the rotation deformer.
    """
    tuning = head.tuning
    height = head.chin - head.face_top
    pivot = (float(head.centre[0]), float(head.chin - tuning.tilt_pivot * height))
    rotation = project.add_rotation('tilt_head', 'RotationHeadTilt', pivot, handle=0.4 * height)
    rotation.bind(parameters['ParamAngleZ'], [-30, 0, 30])
    rotation.set_form(0, Rotation.form(pivot, -tuning.tilt_angle))
    rotation.set_form(2, Rotation.form(pivot, tuning.tilt_angle))
    for member in members:
        while member.parent is not None:
            member = member.parent
        if member is not rotation:
            member.parent = rotation
    if neck is not None:
        below, span = tuning.tilt_neck
        lowest = head.chin + below * height
        follow = np.clip((lowest - neck.rest[..., 1:2]) / (span * height), 0, 1)

        def bent(shape, key, values):
            turned = np.float64(pivot) + turn_points(shape - np.float64(pivot), tuning.tilt_angle * key / 30)
            return shape + follow * (turned - shape)

        add_axis(neck, parameters['ParamAngleZ'], [-30, 0, 30], bent)
    return rotation


# ---------------------------------------------------------------- build and convert

@dataclass
class Built:
    name: str
    out_dir: str
    project: object
    meshes: dict
    parameters: dict
    layers: list
    size: tuple
    cmo3_path: str
    model3_path: str = None
    model: object = None
    textures: list = None
    regions: dict = field(default_factory=dict)

    def path(self, file_name):
        return os.path.join(self.out_dir, file_name)


def build(name, psd_path, out_dir, rig=None, clipped_by=None, mesh_spacing=None, parameters=STANDARD_PARAMETERS):
    """PSD -> <out_dir>/<name>.cmo3.  Every layer becomes an art mesh, in PSD order, on one texture;
    rig(project, meshes, parameters, size) then adds the moving parts.

    clipped_by    {layer: masking layer}: the layer only shows where the masking layer is visible
    mesh_spacing  {layer: grid cell size in pixels} for layers that bend a lot
    """
    os.makedirs(out_dir, exist_ok=True)
    size, layers = read_layers(psd_path)
    project = Project(name, size[0], size[1], psd_path=psd_path)
    by_id = {}
    for param_id, label, minimum, maximum, default, track in parameters:
        by_id[param_id] = project.add_parameter(param_id, label, minimum, maximum, default, track=track)

    meshes = {}
    for layer_name, image, origin in layers:
        vertices, triangles = grid_mesh(image[..., 3], origin, spacing=(mesh_spacing or {}).get(layer_name))
        meshes[layer_name] = project.add_mesh(layer_name, drawable_id(layer_name), image, origin, vertices, triangles)
    for layer_name, mask_name in (clipped_by or {}).items():
        meshes[layer_name].clip = [meshes[mask_name]]
    if rig is not None:
        rig(project, meshes, by_id, size)

    cmo3_path = os.path.join(out_dir, f'{name}.cmo3')
    xml_text = project.save(cmo3_path)
    open(os.path.join(out_dir, 'main.xml'), 'w', encoding='utf-8', newline='\n').write(xml_text)
    print(f'wrote {cmo3_path}: {len(project.meshes)} meshes, {len(project.warps)} warps, {len(project.rotations)} rotations, '
          f'{sum(len(m.vertices) for m in project.meshes)} vertices, '
          f'{sum(len(m.triangles) for m in project.meshes)} triangles, texture {project.atlas_page} px')
    return Built(name, out_dir, project, meshes, by_id, layers, size, cmo3_path)


def convert(built, physics_data=None, install=False):
    """.cmo3 -> runtime files (Umamo), physics file and model3.json groups added, loaded in Core."""
    runtime = built.path('runtime')
    built.model3_path, _ = pipeline.convert_with_umamo(built.cmo3_path, runtime, built.name)
    physics_path = None
    if physics_data is not None:
        physics_path = built.path('hair.physics3.json')
        physics.save(physics_path, physics_data)
    pipeline.fill_model3(built.model3_path, physics_from=physics_path)
    if install:
        print('installed in', pipeline.install_in_vtube_studio(runtime, built.name))
    built.model, built.textures = pipeline.load(built.model3_path)
    model = built.model
    print(f'moc version {model.file_version}, Core accepts it: {model.consistent}; '
          f'{model.param_count} parameters, {model.part_count} parts, {model.drawable_count} drawables')
    return built


# ---------------------------------------------------------------- our own warp maths, to check Core

def warp_points(warp, grid, points):
    """Where a warp with the given grid (canvas pixels) puts points that rest at `points`.

    Our own version of the warp maths (bilinear inside each grid cell), used to check Core.
    """
    local = warp.to_local(points).astype(np.float64)
    u, v = local[:, 0] * warp.cols, local[:, 1] * warp.rows
    column = np.clip(np.floor(u).astype(int), 0, warp.cols - 1)
    row = np.clip(np.floor(v).astype(int), 0, warp.rows - 1)
    fx, fy = (u - column)[:, None], (v - row)[:, None]
    grid = np.asarray(grid, np.float64)
    top = grid[row, column] * (1 - fx) + grid[row, column + 1] * fx
    bottom = grid[row + 1, column] * (1 - fx) + grid[row + 1, column + 1] * fx
    return top * (1 - fy) + bottom * fy


def blended(keyed, pose):
    """A keyed object's shape in a pose: linear between its keys, one parameter after another."""
    choices = []
    for parameter, keys in keyed.bindings:
        value = float(np.clip(pose.get(parameter.id, parameter.default), keys[0], keys[-1]))
        upper = int(np.clip(np.searchsorted(keys, value, side='right'), 1, len(keys) - 1))
        share = (value - keys[upper - 1]) / (keys[upper] - keys[upper - 1])
        choices.append([(upper - 1, 1 - share), (upper, share)])
    shape = np.zeros(keyed.rest_shape().shape, np.float64)
    for combination in itertools.product(*choices):
        weight = float(np.prod([share for _, share in combination]))
        if weight:
            shape += weight * keyed.form_at([index for index, _ in combination])
    return shape


def place(parent, pose, points):
    """Where points stored for `parent` (canvas pixels with the parent at rest) land in a pose.

    A parent warp moves the points by its grid; a rotation turns them about its pivot (a rotation
    form's angle is linear between keys, like every other value).  A rotation inside a warp is
    not followed: Cubism takes the warp's local turn there, which this does not model."""
    if parent is None:
        return points
    if isinstance(parent, Rotation):
        assert not isinstance(parent.parent, Warp), 'a rotation inside a warp is not modelled'
        moved = Rotation.apply(blended(parent, pose), parent.to_local(points).astype(np.float64))
        return place(parent.parent, pose, moved)
    return warp_points(parent, grid_in_pose(parent, pose), points)


def grid_in_pose(warp, pose):
    """A warp's grid in canvas pixels in a pose.  A parent moves the grid points and nothing
    else: what is inside the child follows the child's grid, however coarse that is."""
    grid = blended(warp, pose)
    return place(warp.parent, pose, grid.reshape(-1, 2)).reshape(grid.shape)


def expected_positions(mesh, pose):
    """Where our own maths puts a mesh's points in a pose."""
    return place(mesh.parent, pose, blended(mesh, pose))


def check_positions(model, project, poses):
    """Largest distance, over all meshes and the given poses, between Core and our own maths."""
    worst, where = 0.0, ''
    for pose in poses:
        model.pose(pose)
        positions = dict(zip(model.drawable_ids, model.vertices_px()))
        for mesh in project.meshes:
            error = float(np.abs(positions[mesh.id] - expected_positions(mesh, pose)).max())
            if error > worst:
                worst, where = error, f'{mesh.name} at {pose}'
    model.reset()
    return worst, where


def showing(model, textures, project, names, pose, region=None):
    """How many canvas pixels of the named layers can be seen in a pose.

    Renders with those layers painted pure magenta and every other layer black.
    """
    page = textures[0].copy()
    page[..., :3] = 0
    for mesh in project.meshes:
        if mesh.name in names:
            x, y = mesh.material_in_atlas
            height, width = mesh.image.shape[:2]
            area = page[y:y + height, x:x + width]
            area[..., 0] = area[..., 3]
            area[..., 2] = area[..., 3]
    model.pose(pose)
    picture = l2dcore.render(model, [page], background=(0, 0, 0), region=region)
    return int(((picture[..., 0] > 128) & (picture[..., 1] < 60)).sum())


# ---------------------------------------------------------------- pictures

def labelled(image, text):
    image = image.copy()
    cv2.putText(image, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5)
    cv2.putText(image, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 0, 0), 2)
    return image


def pose_sheet(path, model, textures, region, scale, poses):
    """Saves close-ups of one region, one panel per (label, {parameter: value})."""
    panels = []
    for label, pose in poses:
        model.pose(pose)
        panels.append(labelled(l2dcore.render(model, textures, scale=scale, region=region), label))
    l2dcore.save_rgb(path, np.concatenate(panels, axis=1))
    model.reset()


def pose_gif(path, model, textures, region, scale, poses, frame_ms=60):
    frames = []
    for pose in poses:
        model.pose(pose)
        frames.append(Image.fromarray(l2dcore.render(model, textures, scale=scale, region=region)))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
    model.reset()


def padded(box, pad_x, pad_y, size):
    """A box grown by a margin and kept on the canvas, as whole pixels."""
    return (int(max(box[0] - pad_x, 0)), int(max(box[1] - pad_y, 0)),
            int(min(box[2] + pad_x, size[0])), int(min(box[3] + pad_y, size[1])))


def union(boxes):
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def measure_regions(meshes, size):
    """Where to point the close-ups, measured from the layers: mouth, eyes, face, head, figure."""
    face = drawn_box(meshes['face'])
    width = face[2] - face[0]
    regions = {'figure': (0, 0, size[0], size[1]),
               'face': padded(face, 0.2 * width, 0.1 * width, size)}
    head_layers = [name for name in ('face', 'front hair', 'headwear', 'ears-l', 'ears-r') if name in meshes]
    regions['head'] = padded(union(drawn_box(meshes[name]) for name in head_layers), 0.3 * width, 0.25 * width, size)
    eye_layers = [name for name in ('eyelash-l', 'eyelash-r', 'eyewhite-l', 'eyewhite-r') if name in meshes]
    if eye_layers:
        regions['eyes'] = padded(union(drawn_box(meshes[name]) for name in eye_layers), 0.12 * width, 0.12 * width, size)
    if 'mouth inner' in meshes:
        regions['mouth'] = padded(drawn_box(meshes['mouth inner']), 0.12 * width, 0.1 * width, size)
    return regions


# ---------------------------------------------------------------- reports

def report_rest(built, psd_pose=None, clipped_in_psd=None, source_picture=None, skip_in_psd=()):
    """The pose the PSD was drawn in, against the flattened PSD (and the artist's picture, if given).

    skip_in_psd: layers left out of the flattened PSD (invisible masks)."""
    model, textures = built.model, built.textures
    model.pose(psd_pose or {})
    rendered = l2dcore.render(model, textures)
    reference = reference_picture(built.size, built.layers, clipped_in_psd, skip=skip_in_psd)
    mean, over24, over64, difference = pipeline.difference_report(reference, rendered)
    print(f'PSD pose vs flattened PSD: mean difference {mean:.3f} of 255, '
          f'{over24} pixels differ by more than 24, {over64} by more than 64')
    heat = np.repeat(np.clip(difference * 4, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    model.reset()
    at_rest = l2dcore.render(model, textures)
    panels = [labelled(reference, 'flattened PSD'), labelled(rendered, 'model, same pose'),
              labelled(heat, 'difference x4'), labelled(at_rest, 'model at rest')]
    if source_picture is not None:
        mean, over24, over64, _ = pipeline.difference_report(source_picture, at_rest)
        print(f'model at rest vs the source picture: mean difference {mean:.3f} of 255, '
              f'{over24} pixels differ by more than 24, {over64} by more than 64')
        panels.append(labelled(source_picture, 'source picture'))
    l2dcore.save_rgb(built.path('check_rest.jpg'), np.concatenate(panels, axis=1))
    return at_rest


def bound_parameters(project):
    return list({binding[0].id: binding[0] for keyed in project.warps + project.rotations + project.meshes
                 for binding in keyed.bindings}.values())


def report_positions(built):
    """Every mesh, in many poses, against our own warp maths."""
    project, model = built.project, built.model
    keyed_objects = project.warps + project.rotations + project.meshes
    bound = bound_parameters(project)
    key_poses = [{}]
    for parameter in bound:
        keys = sorted({key for keyed in keyed_objects for other, values in keyed.bindings
                       if other is parameter for key in values})
        key_poses += [{parameter.id: key} for key in keys]
    turning = [p for p in bound if p.id in ('ParamAngleX', 'ParamAngleY')]
    others = [p for p in bound if p not in turning]
    corners = [dict(zip((p.id for p in turning), values))
               for values in itertools.product(*[(p.minimum, p.maximum) for p in turning])] if turning else [{}]
    key_poses += corners
    extreme = {p.id: (p.maximum if p.default != p.maximum and number % 3 else p.minimum if p.default != p.minimum else p.maximum)
               for number, p in enumerate(others)}
    key_poses += [dict(corner, **extreme) for corner in corners]
    random = np.random.default_rng(7)
    between = [{p.id: float(random.uniform(p.minimum, p.maximum)) for p in bound} for _ in range(3)]
    print('positions, Core against our own maths: worst %.4f px at keys (%s); worst %.4f px between keys (%s)'
          % (*check_positions(model, project, key_poses), *check_positions(model, project, between)))


def report_mouth(built):
    """Mouth: the interior must be hidden when closed."""
    model, textures, project, region = built.model, built.textures, built.project, built.regions['mouth']
    count = lambda value: showing(model, textures, project, {'mouth inner'}, {'ParamMouthOpenY': value}, region)
    print(f'mouth: interior pixels showing when open {count(1)}, half open {count(0.5)}, closed {count(0)}')
    pose_sheet(built.path('check_mouth.jpg'), model, textures, region, 4.0,
               [(f'{value}', {'ParamMouthOpenY': value}) for value in (0, 0.25, 0.5, 0.75, 1)])
    talk = [0, 0, 0.3, 0.7, 1, 0.8, 0.4, 0.6, 1, 0.5, 0.1, 0, 0]
    pose_gif(built.path('mouth.gif'), model, textures, built.regions['face'], 2.0,
             [{'ParamMouthOpenY': v} for v in talk], frame_ms=90)


def report_pupils(built):
    """Pupils: how far they go, and whether anything of them shows outside the eye opening."""
    model, textures, region = built.model, built.textures, built.regions['eyes']
    model.reset()
    default = dict(zip(model.drawable_ids, model.vertices_px()))
    looks = [('left', {'ParamEyeBallX': -1}), ('right', {'ParamEyeBallX': 1}), ('up', {'ParamEyeBallY': 1}),
             ('down', {'ParamEyeBallY': -1}), ('up-right', {'ParamEyeBallX': 1, 'ParamEyeBallY': 1}),
             ('down-left', {'ParamEyeBallX': -1, 'ParamEyeBallY': -1})]
    for label, pose in looks[:3]:
        model.pose(pose)
        shift = dict(zip(model.drawable_ids, model.vertices_px()))['IridesL'] - default['IridesL']
        print(f'pupils look {label}: irises move ({shift[:, 0].mean():+.2f}, {shift[:, 1].mean():+.2f}) px')
    pose_sheet(built.path('check_pupils.jpg'), model, textures, region, 2.0, [('centre', {})] + looks[:4])
    pose_sheet(built.path('check_pupils_corners.jpg'), model, textures, region, 2.0,
               looks[4:] + [('up-right, half closed', dict(looks[4][1], ParamEyeLOpen=0.5, ParamEyeROpen=0.5))])
    circle = [{'ParamEyeBallX': float(np.cos(a)), 'ParamEyeBallY': float(np.sin(a))}
              for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    pose_gif(built.path('pupils.gif'), model, textures, region, 2.0, circle)


EYE_LAYERS = {'eyewhite-l', 'eyewhite-r', 'irides-l', 'irides-r'}
both_eyes = lambda value: {'ParamEyeLOpen': value, 'ParamEyeROpen': value}


def report_blink(built):
    """Blink: the eyes must be fully covered when closed."""
    model, textures, project, region = built.model, built.textures, built.project, built.regions['eyes']
    count = lambda value: showing(model, textures, project, EYE_LAYERS, both_eyes(value), region)
    print(f'blink: eye pixels showing when open {count(1)}, half closed {count(0.5)}, closed {count(0)}')
    pose_sheet(built.path('check_blink.jpg'), model, textures, region, 2.0,
               [(f'Eye Open = {value}', both_eyes(value)) for value in (1, 0.66, 0.33, 0)])
    cycle = [1, 1, 1, 1, 0.8, 0.5, 0.2, 0, 0, 0.2, 0.5, 0.8, 1, 1, 1, 1]
    pose_gif(built.path('blink.gif'), model, textures, region, 2.0, [both_eyes(v) for v in cycle])
    pose_sheet(built.path('check_wink.jpg'), model, textures, region, 2.0,
               [('her left eye closed', {'ParamEyeLOpen': 0}), ('her right eye closed', {'ParamEyeROpen': 0})])


def turned(x, y, **more):
    return dict(ParamAngleX=x, ParamAngleY=y, **more)


def report_turn(built, everyday=15):
    """Head turn: the full range, the range ordinary tracking reaches, and that nothing hidden comes out."""
    model, textures, project = built.model, built.textures, built.project
    head, face = built.regions['head'], built.regions['face']
    scale = min(1.0, 560 / (head[2] - head[0]))
    for file_name, reach in (('check_turn.jpg', 30), ('check_turn_everyday.jpg', everyday)):
        rows = []
        for y in (reach, 0, -reach):
            panels = []
            for x in (-reach, 0, reach):
                model.pose(turned(x, y))
                panels.append(labelled(l2dcore.render(model, textures, scale=scale * 0.75, region=head), f'X {x:+d}  Y {y:+d}'))
            rows.append(np.concatenate(panels, axis=1))
        l2dcore.save_rgb(built.path(file_name), np.concatenate(rows, axis=0))
    pose_sheet(built.path('check_turn_x.jpg'), model, textures, head, scale,
               [(f'Angle X = {x:+d}', turned(x, 0)) for x in (-30, -everyday, 0, everyday, 30)])
    pose_sheet(built.path('check_turn_y.jpg'), model, textures, head, scale,
               [(f'Angle Y = {y:+d}', turned(0, y)) for y in (-30, -everyday, 0, everyday, 30)])
    circle = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    pose_gif(built.path('turn.gif'), model, textures, head, scale * 0.75,
             [turned(30 * float(np.cos(a)), 30 * float(np.sin(a))) for a in circle])
    pose_gif(built.path('turn_everyday.gif'), model, textures, head, scale * 0.75,
             [turned(everyday * float(np.cos(a)), everyday * float(np.sin(a))) for a in circle])
    pose_gif(built.path('turn_x.gif'), model, textures, head, scale * 0.75,
             [turned(30 * float(np.sin(a)), 0) for a in np.linspace(0, 2 * np.pi, 30, endpoint=False)])
    hidden = {}
    corners = [turned(x, y) for x in (-30, 30) for y in (-30, 30)]
    for corner in corners + [turned(30, 0), turned(-30, 0), turned(0, 30), turned(0, -30)]:
        key = f"({corner['ParamAngleX']:+d}, {corner['ParamAngleY']:+d})"
        hidden[key] = (showing(model, textures, project, EYE_LAYERS, dict(corner, **both_eyes(0))),
                       showing(model, textures, project, {'mouth inner'}, dict(corner, ParamMouthOpenY=0)))
    print('turned head, pixels showing (closed eyes, closed mouth interior):',
          ', '.join(f'{key} {eye}/{mouth_pixels}' for key, (eye, mouth_pixels) in hidden.items()))
    pose_sheet(built.path('check_turn_faces.jpg'), model, textures, face, 1.5,
               [('turned, eyes shut, talking', turned(30, -30, ParamMouthOpenY=1, **both_eyes(0))),
                ('turned, looking back', turned(-30, 20, ParamEyeBallX=1, ParamEyeBallY=-1, ParamMouthOpenY=0.5)),
                ('turned, wink', turned(30, 30, ParamEyeLOpen=0))])


def report_tilt(built):
    """Head tilt: the range, and that nothing hidden comes out when tilted, turned or not."""
    model, textures, project = built.model, built.textures, built.project
    head = built.regions['head']
    scale = min(1.0, 560 / (head[2] - head[0]))
    pose_sheet(built.path('check_tilt.jpg'), model, textures, head, scale * 0.75,
               [(f'Angle Z = {z:+d}', {'ParamAngleZ': z}) for z in (-30, -15, 0, 15, 30)])
    hidden = []
    for z in (-30, 30):
        for x, y in ((0, 0), (-30, -30), (-30, 30), (30, -30), (30, 30)):
            pose = turned(x, y, ParamAngleZ=z)
            hidden.append(f'({x:+d}, {y:+d}, {z:+d}) '
                          f'{showing(model, textures, project, EYE_LAYERS, dict(pose, **both_eyes(0)))}/'
                          f'{showing(model, textures, project, {"mouth inner"}, dict(pose, ParamMouthOpenY=0))}')
    print('tilted head, pixels showing (closed eyes / closed mouth interior) at (X, Y, Z):', ', '.join(hidden))
    pose_gif(built.path('tilt.gif'), model, textures, head, scale * 0.75,
             [{'ParamAngleZ': 30 * float(np.sin(a))} for a in np.linspace(0, 2 * np.pi, 30, endpoint=False)])
    model.reset()


def report_sway(built, sheets, lags, figure_scale=0.5):
    """Hair sway.  sheets: [(file name, parameter id, label, region, scale)], one parameter at a
    time; lags: {parameter id: phase lag} for the GIF with all of them swinging together."""
    model, textures = built.model, built.textures
    for file_name, parameter, label, region, scale in sheets:
        pose_sheet(built.path(file_name), model, textures, region, scale,
                   [(f'{label} = {value:+d}', {parameter: value}) for value in (-1, 0, 1)])
    swing = [{name: float(np.sin(phase - lag)) for name, lag in lags.items()}
             for phase in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    pose_gif(built.path('sway.gif'), model, textures, built.regions['figure'], figure_scale, swing)


def demo_head_angle(t):
    """A scripted head movement in degrees over 7.5 seconds: two quick turns, back, gentle shaking."""
    if t < 2.0:
        return 30 * float(smoothstep((t - 0.5) / 0.2))             # quick turn to her left, hold
    if t < 3.6:
        return 30 - 60 * float(smoothstep((t - 2.0) / 0.35))       # quick turn to the other side, hold
    if t < 4.2:
        return -30 + 30 * float(smoothstep((t - 3.6) / 0.6))       # back to the middle
    return 18 * float(np.sin(2 * np.pi * (t - 4.2))) * float(1 - smoothstep((t - 5.7) / 1.0))   # gentle shaking


def report_physics(built, physics_data, figure_scale=0.5):
    """Physics preview: turn the head and let our own pendulum simulation move the hair."""
    ranges = {parameter.id: (parameter.minimum, parameter.maximum) for parameter in built.project.parameters}
    simulation = physics.Simulation(physics_data, ranges)
    frames, peaks = [], {}
    for step in range(int(7.5 * 60)):
        angle = demo_head_angle(step / 60)
        values = simulation.step({'ParamAngleX': angle}, 1 / 60)
        for name, value in values.items():
            peaks[name] = max(peaks.get(name, 0.0), abs(value))
        if step % 3 == 0:
            frames.append(dict(values, ParamAngleX=angle))
    print('physics preview (our simulation), largest value of each sway parameter:',
          ', '.join(f'{name} {value:.2f}' for name, value in peaks.items()))
    pose_gif(built.path('physics_preview.gif'), built.model, built.textures, built.regions['figure'],
             figure_scale, frames, frame_ms=50)
