"""Write a Cubism Editor project file (.cmo3) from Python.

A .cmo3 is a CAFF archive (see caff.py) holding main.xml - a serialized object graph - and PNGs.
The XML written here follows, field by field, the projects Cubism Editor 5.3 saved for this
project (work/v6/girl_v6.cmo3 and the v3 series).  Which objects a minimal project needs was
learned from StretchyStudio's MIT-licensed cmo3writer.js (Copyright (c) 2026 Nguyen Phan).

Scope so far: parameters, one root part, art meshes, warp deformers and rotation deformers (nested
in each other or at the root; a rotation inside a warp is written but our own maths in rigkit does
not follow it), keys on parameters for all of them, one texture atlas.  The rotation deformer's
fields follow the one the Editor 5.3 saved in work/code1/in/girl_v6.cmo3 (面部旋转).

Everything is described in canvas pixels, as it looks with every parameter at its default.  The
writer converts to the space the file wants (see Warp.to_local).

Typical use:

    project = Project('girl', 1280, 1280, psd_path=...)
    sway_parameter = project.add_parameter('ParamHairFront', 'Hair Front', -1, 1, 0)
    mesh = project.add_mesh('face', 'face', rgba_crop, (x, y), vertices, triangles)
    warp = project.add_warp('sw_front', 'WarpHairFront', (left, top, right, bottom))
    warp.bind(sway_parameter, [-1, 0, 1])
    warp.set_form(0, warp.rest + offsets_left)      # the shape at key -1; unset keys keep the rest shape
    warp.set_form(2, warp.rest + offsets_right)
    mesh.parent = warp
    project.save('girl.cmo3')
"""
import datetime
import io
import itertools
import math
import os
import time
import uuid

import numpy as np
from PIL import Image
from scipy import ndimage

from . import caff
from .automesh import edges_of

FILE_FORMAT_VERSION = '503040001'
EDITOR_VERSION_NO = '5030000'

VERSION_PIS = [
    ('CArtMeshSource', '5'),
    ('RotationDeformerOriginalShape', '1'),
    ('KeyformGridSource', '1'),
    ('CModelSource', '15'),
    ('CFloatColor', '1'),
    ('CPartForm', '2'),
    ('ACDeformerOriginalShape', '1'),
    ('CParameterGroup', '4'),
    ('com.live2d.cubism.doc.model.texture.textureAtlas.ModelImageEntry', '2'),
    ('WarpDeformerOriginalShape', '1'),
    ('SerializeFormatVersion', '2'),
    ('CRotationDeformerForm', '1'),
    ('CPartSource', '2'),
    ('CLabelColor', '0'),
    ('CModelImage', '3'),
]

# Fixed ids the Editor recognises by value; all of them appear in our own saved projects.
ROOT_DEFORMER_UUID = '71fae776-e218-4aee-873e-78e8ac0cb48a'
ROOT_PARAMETER_GROUP_UUID = 'e9fe6eff-953b-4ce2-be7c-4a7c3913686b'
FILTER_DEF_LAYER_SELECTOR = '5e9fe1ea-0ec3-4d68-a5fa-018fc7abe301'
FILTER_DEF_LAYER_FILTER = '4083cd1f-40ba-4eda-8400-379019d55ed8'

IDENTITY = dict(m00='1.0', m01='0.0', m02='0.0', m10='0.0', m11='1.0', m12='0.0')


def number(value):
    """A float the way the Editor (Java) writes it: shortest text that round-trips as 32-bit,
    as 9.765625E-4 below a thousandth and from ten million up."""
    value = np.float32(value)
    size = abs(value)
    if size != 0 and (size < 1e-3 or size >= 1e7) and np.isfinite(value):
        mantissa, exponent = np.format_float_scientific(value, unique=True, trim='0').split('e')
        return f'{mantissa}E{int(exponent)}'
    return str(value)


def numbers(values):
    return ' '.join(number(v) for v in np.asarray(values, np.float32).ravel())


def new_uuid():
    return str(uuid.uuid4())


def png_bytes(rgba):
    buffer = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(buffer, 'PNG')
    return buffer.getvalue()


def thumbnail(rgba, size):
    """The picture shrunk to fit a size x size transparent square."""
    image = Image.fromarray(rgba, 'RGBA')
    image.thumbnail((size, size), Image.LANCZOS)
    square = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    square.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return np.asarray(square)


def extend_edge_colour(rgba, reach=8):
    """Copies the nearest visible colour into fully transparent pixels close to the art.

    Alpha is untouched.  Without this, apps that blend plain (not premultiplied) textures pull
    black into soft edges.
    """
    known = rgba[..., 3] > 0
    if known.all() or not known.any():
        return rgba
    distance, nearest = ndimage.distance_transform_edt(~known, return_indices=True)
    fill = ~known & (distance <= reach)
    result = rgba.copy()
    result[fill, :3] = rgba[nearest[0][fill], nearest[1][fill], :3]
    return result


def pack_shelves(sizes, page, gap=2):
    """Places rectangles on a square page in rows, tallest first.  Returns positions or None."""
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0]))
    positions = [None] * len(sizes)
    x = y = shelf = 0
    for index in order:
        width, height = sizes[index]
        if width > page:
            return None
        if x + width > page:
            x, y, shelf = 0, y + shelf + gap, 0
        if y + height > page:
            return None
        positions[index] = (x, y)
        x += width + gap
        shelf = max(shelf, height)
    return positions


class Node:
    __slots__ = ('tag', 'attrs', 'children', 'text')

    def __init__(self, tag, attrs=None):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.children = []
        self.text = None


def _escape(text, quotes):
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return text.replace('"', '&quot;') if quotes else text


class XmlBuilder:
    """Builds the object-graph XML: shared objects get an xs.id and are referred to by xs.ref."""

    def __init__(self):
        self.shared_nodes = []

    def shared(self, tag, **attrs):
        node = Node(tag, attrs)
        node.attrs['xs.id'] = f'#{len(self.shared_nodes)}'
        node.attrs['xs.idx'] = str(len(self.shared_nodes))
        self.shared_nodes.append(node)
        return node

    @staticmethod
    def sub(parent, tag, name=None, text=None, **attrs):
        node = Node(tag)
        if name is not None:
            node.attrs['xs.n'] = name
        node.attrs.update(attrs)
        if text is not None:
            node.text = str(text)
        parent.children.append(node)
        return node

    @staticmethod
    def ref(parent, target, name=None):
        node = Node(target.tag)
        if name is not None:
            node.attrs['xs.n'] = name
        node.attrs['xs.ref'] = target.attrs['xs.id']
        parent.children.append(node)
        return node

    @staticmethod
    def collection(parent, tag, name, **attrs):
        """A list element; its count attribute is filled in when the file is written."""
        node = Node(tag)
        node.attrs['xs.n'] = name
        node.attrs['count'] = None
        node.attrs.update(attrs)
        parent.children.append(node)
        return node

    def serialize(self, root, imports):
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines += [f'<?version {name}:{version}?>' for name, version in VERSION_PIS]
        lines += [f'<?import {name}?>' for name in imports]
        self._write(root, lines)
        return '\n'.join(lines) + '\n'

    def _write(self, node, lines):
        parts = [f'<{node.tag}']
        for key, value in node.attrs.items():
            if key == 'count' and value is None:
                value = len(node.children)
            parts.append(f' {key}="{_escape(str(value), True)}"')
        if not node.children and not node.text:
            lines.append(''.join(parts) + ' />')
            return
        if node.text:
            lines.append(''.join(parts) + f'>{_escape(node.text, False)}</{node.tag}>')
            return
        lines.append(''.join(parts) + '>')
        for child in node.children:
            self._write(child, lines)
        lines.append(f'</{node.tag}>')


class Parameter:
    def __init__(self, param_id, name, minimum, maximum, default, description='', track='NONE'):
        self.id = param_id
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.default = default
        self.description = description
        self.track = track          # cursor tracking in the Editor's viewer: 'X', 'Y' or 'NONE'


class Keyed:
    """An object whose shape is keyed on parameters.

    bindings  [(Parameter, [key values])]; with several parameters there is one shape per
              combination of keys.
    forms     {(key index per binding): shape}; combinations left out use the rest shape.
    """

    def _init_keys(self):
        self.bindings = []
        self.forms = {}
        self.parent = None          # a Warp, or None for the root

    def bind(self, parameter, keys):
        self.bindings.append((parameter, [float(key) for key in keys]))

    def set_form(self, index, shape):
        index = (index,) if isinstance(index, int) else tuple(index)
        shape = np.asarray(shape, np.float32)
        assert len(index) == len(self.bindings), 'one key index per bound parameter'
        assert shape.shape == self.rest_shape().shape, 'a form must have the shape of the rest form'
        self.forms[index] = shape

    def key_indices(self):
        """Every key combination; the first bound parameter varies fastest, like the Editor."""
        ranges = [range(len(keys)) for _, keys in reversed(self.bindings)]
        return [tuple(reversed(combination)) for combination in itertools.product(*ranges)]

    def form_at(self, index):
        return self.forms.get(tuple(index), self.rest_shape())


class Warp(Keyed):
    """A warp deformer: a grid of cols x rows cells; whatever sits inside follows the grid.

    box   (left, top, right, bottom) in canvas pixels: the grid with all parameters at default.
    rest  the grid points of that box, shape (rows + 1, cols + 1, 2); forms have the same shape
          and are also in canvas pixels (where the points would be if the parent stayed at rest).
    """

    def __init__(self, name, deformer_id, box, cols=5, rows=5, parent=None):
        self._init_keys()
        self.name = name
        self.id = deformer_id
        self.box = tuple(float(v) for v in box)
        self.cols = int(cols)
        self.rows = int(rows)
        self.parent = parent
        left, top, right, bottom = self.box
        xs, ys = np.meshgrid(np.linspace(left, right, self.cols + 1), np.linspace(top, bottom, self.rows + 1))
        self.rest = np.stack([xs, ys], axis=-1).astype(np.float32)

    def rest_shape(self):
        return self.rest

    def to_local(self, points):
        """Canvas pixels -> this warp's own space, where 0..1 spans its box.

        Children of a warp (meshes and warps alike) store their points this way; storing canvas
        pixels instead is what made StretchyStudio's meshes vanish.
        """
        left, top, right, bottom = self.box
        return (np.asarray(points, np.float32) - np.float32([left, top])) / np.float32([right - left, bottom - top])


def turn(points, degrees):
    """Points turned about (0, 0) on the canvas (y down): + is clockwise on screen, as in Cubism."""
    angle = np.radians(np.float64(degrees))
    c, s = np.cos(angle), np.sin(angle)
    points = np.asarray(points, np.float64)
    return np.stack([c * points[..., 0] - s * points[..., 1], s * points[..., 0] + c * points[..., 1]], axis=-1)


class Rotation(Keyed):
    """A rotation deformer: whatever sits inside turns and scales about a pivot.

    rest   [pivot x, pivot y, angle, scale] with all parameters at default: the pivot in canvas
           pixels, the angle in degrees (+ = clockwise on screen; 0 = the handle points up).
    forms  the same four numbers per key, in canvas terms as for warps: where the pivot would be
           if the parent stayed at rest, and the angle as seen on the canvas.

    Inside it, points are stored in its own frame: pixels from the pivot, turned back by the rest
    angle and divided by the rest scale (see to_local).  Live2D's rig of Haru and the Editor's own
    files store them this way (pixels relative to the pivot).
    """

    def __init__(self, name, deformer_id, pivot, angle=0.0, scale=1.0, parent=None, handle=200.0):
        self._init_keys()
        self.name = name
        self.id = deformer_id
        self.parent = parent
        self.handle = float(handle)         # the length of the handle drawn in the Editor, pixels
        self.rest = np.float32([pivot[0], pivot[1], angle, scale])

    def rest_shape(self):
        return self.rest

    @staticmethod
    def form(pivot, angle=0.0, scale=1.0):
        return np.float32([pivot[0], pivot[1], angle, scale])

    def to_local(self, points):
        """Canvas pixels (with this deformer at rest) -> its own frame."""
        pivot, angle, scale = self.rest[:2], self.rest[2], self.rest[3]
        return (turn(np.asarray(points, np.float64) - pivot, -angle) / scale).astype(np.float32)

    @staticmethod
    def apply(form, local):
        """Where points of the own frame land for a form (pivot, angle, scale)."""
        form = np.asarray(form, np.float64)
        return form[:2] + form[3] * turn(local, form[2])


class Mesh(Keyed):
    def __init__(self, name, drawable_id, image, origin, vertices, triangles, clip=()):
        self._init_keys()
        self.name = name
        self.id = drawable_id
        self.image = np.ascontiguousarray(image, np.uint8)       # (h, w, 4) RGBA crop of the layer
        self.origin = (int(origin[0]), int(origin[1]))           # canvas position of image[0, 0]
        self.vertices = np.asarray(vertices, np.float32).reshape(-1, 2)   # canvas pixels
        self.triangles = np.asarray(triangles, np.int32).reshape(-1, 3)
        self.clip = list(clip)                                   # meshes that mask this one
        self.draw_order = 500
        self.opacity = 1.0

    def rest_shape(self):
        return self.vertices


class Project:
    def __init__(self, name, width, height, psd_path=None):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.psd_path = psd_path or f'{name}.psd'
        self.parameters = []
        self.meshes = []            # front to back, like the Editor's parts panel
        self.warps = []             # any order; save() writes parents before their children
        self.rotations = []         # the same
        self.atlas_page = None      # set by save(): page size in pixels
        self.atlas_image = None     # set by save(): the page as (h, w, 4) RGBA

    def add_parameter(self, *args, **kwargs):
        parameter = Parameter(*args, **kwargs)
        self.parameters.append(parameter)
        return parameter

    def add_mesh(self, *args, **kwargs):
        mesh = Mesh(*args, **kwargs)
        self.meshes.append(mesh)
        return mesh

    def add_warp(self, *args, **kwargs):
        warp = Warp(*args, **kwargs)
        self.warps.append(warp)
        return warp

    def add_rotation(self, *args, **kwargs):
        rotation = Rotation(*args, **kwargs)
        self.rotations.append(rotation)
        return rotation

    def _ordered_deformers(self):
        """Warps and rotations, parents before their children (a parent may have been set after
        both were added)."""
        everything = self.warps + self.rotations
        ordered = []

        def place(deformer, chain=()):
            assert deformer not in chain, f'deformer {deformer.name} is its own ancestor'
            if deformer in ordered:
                return
            if deformer.parent is not None:
                assert deformer.parent in everything, f'the parent of deformer {deformer.name} is not in the project'
                place(deformer.parent, chain + (deformer,))
            ordered.append(deformer)

        for deformer in everything:
            place(deformer)
        return ordered

    # ------------------------------------------------------------------ atlas

    def _build_atlas(self, pad=2):
        """Packs one region per mesh onto a page; a region covers the layer image and the mesh."""
        regions = []
        for mesh in self.meshes:
            height, width = mesh.image.shape[:2]
            left = min(mesh.origin[0], math.floor(float(mesh.vertices[:, 0].min()))) - pad
            top = min(mesh.origin[1], math.floor(float(mesh.vertices[:, 1].min()))) - pad
            right = max(mesh.origin[0] + width, math.ceil(float(mesh.vertices[:, 0].max()))) + pad
            bottom = max(mesh.origin[1] + height, math.ceil(float(mesh.vertices[:, 1].max()))) + pad
            regions.append((left, top, right - left, bottom - top))
        sizes = [(w, h) for _, _, w, h in regions]
        for page in (256, 512, 1024, 2048, 4096, 8192):
            positions = pack_shelves(sizes, page)
            if positions:
                break
        else:
            raise ValueError('layers do not fit on an 8192 pixel texture')
        image = np.zeros((page, page, 4), np.uint8)
        for mesh, (left, top, _, _), (x, y) in zip(self.meshes, regions, positions):
            height, width = mesh.image.shape[:2]
            ox, oy = x + mesh.origin[0] - left, y + mesh.origin[1] - top
            image[oy:oy + height, ox:ox + width] = mesh.image
            mesh.atlas_to_canvas = (left - x, top - y)                  # canvas = atlas + this
            mesh.material_in_atlas = (ox, oy)
        self.atlas_page = page
        self.atlas_image = extend_edge_colour(image)

    # ------------------------------------------------------------------ writing

    def save(self, path):
        x = XmlBuilder()
        sub, ref, collection = x.sub, x.ref, x.collection
        files = []                  # (archive path, bytes)
        now_ms = int(time.time() * 1000)
        psd_ms = int(os.path.getmtime(self.psd_path) * 1000) if os.path.exists(self.psd_path) else now_ms
        self._build_atlas()
        page = self.atlas_page

        def add_file(prefix, content):
            count = sum(1 for name, _ in files if name.startswith(prefix + '_') or name == prefix + '.png')
            name = f'{prefix}.png' if count == 0 else f'{prefix}_{count - 1}.png'
            files.append((name, content))
            return name

        def image_resource(rgba):
            content = png_bytes(rgba)
            node = x.shared('CImageResource', width=rgba.shape[1], height=rgba.shape[0], type='INT_ARGB',
                            imageFileBuf_size=len(content), previewFileBuf_size=0)
            sub(node, 'file', 'imageFileBuf', path=add_file('imageFileBuf', content))
            return node

        def icon(rgba, size):
            node = x.shared('CImageIcon')
            image = sub(node, 'CWritableImage', 'image', width=size, height=size, type='INT_ARGB')
            sub(image, 'file', 'image', path=add_file('image', png_bytes(thumbnail(rgba, size))))
            return node

        def affine(parent, name, m02=0.0, m12=0.0):
            return sub(parent, 'CAffine', name, m00='1.0', m01='0.0', m02=number(m02),
                       m10='0.0', m11='1.0', m12=number(m12))

        def label_colour(parent):
            node = sub(parent, 'CLabelColor', 'labelColor', customizedColorInt='-1')
            sub(node, 'CLabelColorType', 'labelType', v='UNDEFINED')

        def cached_image_manager(parent, resource, width, height):
            manager = sub(parent, 'CCachedImageManager', 'cachedImageManager')
            sub(manager, 'CachedImageType', 'defaultCacheType', v='SCALE_1')
            ref(manager, resource, 'rawImage')
            cached = sub(collection(manager, 'array_list', 'cachedImages'), 'CCachedImage')
            ref(cached, resource, '_cachedImageResource')
            sub(cached, 'b', 'isSharedImage', 'true')
            sub(cached, 'CSize', 'rawImageSize', width=width, height=height)
            sub(cached, 'i', 'reductionRatio', 1)
            sub(cached, 'i', 'mipmapLevel', 64)
            sub(cached, 'b', 'hasMargin', 'false')
            sub(cached, 'b', 'isCleaned', 'false')
            sub(cached, 'CAffine', 'transformRawImageToCachedImage',
                m00=number(width / (math.ceil(width / 64) * 64)), m01='0.0', m02='0.0',
                m10='0.0', m11=number(height / (math.ceil(height / 64) * 64)), m12='0.0')
            sub(manager, 'i', 'requiredMipmapLevel', 64)

        def fill_grid(grid, cells, binding_nodes):
            cell_list = collection(grid, 'array_list', 'keyformsOnGrid')
            for index, form_guid in cells:
                cell = sub(cell_list, 'KeyformOnGrid')
                access = sub(cell, 'KeyformGridAccessKey', 'accessKey')
                on_parameters = collection(access, 'array_list', '_keyOnParameterList')
                for binding_node, key_index in zip(binding_nodes, index):
                    on_parameter = sub(on_parameters, 'KeyOnParameter')
                    ref(on_parameter, binding_node, 'binding')
                    sub(on_parameter, 'i', 'keyIndex', key_index)
                ref(cell, form_guid, 'keyformGuid')
            bindings = collection(grid, 'array_list', 'keyformBindings')
            for binding_node in binding_nodes:
                ref(bindings, binding_node)

        def key_cells(keyed, note):
            """One form id per key combination: [(key indices, CFormGuid)], plus the grid object.

            An object without keys has one cell and its grid is written in place (grid None here).
            A keyed object's grid and bindings are shared objects, as the Editor writes them.
            """
            if not keyed.bindings:
                return [((), x.shared('CFormGuid', uuid=new_uuid(), note=note))], None
            grid = x.shared('KeyformGridSource')
            binding_nodes = [x.shared('KeyformBindingSource') for _ in keyed.bindings]
            cells = [(index, x.shared('CFormGuid', uuid=new_uuid(), note='Key [  ]')) for index in keyed.key_indices()]
            fill_grid(grid, cells, binding_nodes)
            for node, (parameter, keys) in zip(binding_nodes, keyed.bindings):
                ref(node, grid, '_gridSource')
                ref(node, parameter.guid, 'parameterGuid')
                key_list = collection(node, 'array_list', 'keys')
                for key in keys:
                    sub(key_list, 'f', None, number(key))
                sub(node, 'InterpolationType', 'interpolationType', v='LINEAR')
                sub(node, 'ExtendedInterpolationType', 'extendedInterpolationType', v='LINEAR')
                sub(node, 'i', 'insertPointCount', 1)
                sub(node, 'f', 'extendedInterpolationScale', '1.0')
                sub(node, 's', 'description', '')
            return cells, grid

        def keyform_grid(parent, cells, shared_grid):
            if shared_grid is not None:
                ref(parent, shared_grid, 'keyformGridSource')
            else:
                fill_grid(sub(parent, 'KeyformGridSource', 'keyformGridSource'), cells, [])

        def extension_base(parent, owner):
            node = sub(parent, 'ACExtension', 'super')
            sub(node, 'CExtensionGuid', 'guid', uuid=new_uuid(), note='(no debug info)')
            ref(node, owner, '_owner')

        def in_parent_space(keyed, points):
            return points if keyed.parent is None else keyed.parent.to_local(points)

        def controllable(parent, local_name, parent_part_guid, cells, shared_grid=None):
            node = sub(parent, 'ACParameterControllableSource', 'super')
            sub(node, 's', 'localName', local_name)
            sub(node, 'b', 'isVisible', 'true')
            sub(node, 'b', 'isLocked', 'false')
            if parent_part_guid is None:
                sub(node, 'null', 'parentGuid')
            else:
                ref(node, parent_part_guid, 'parentGuid')
            keyform_grid(node, cells, shared_grid)
            morph = sub(node, 'KeyFormMorphTargetSet', 'keyformMorphTargetSet')
            collection(morph, 'carray_list', '_morphTargets')
            constraints = sub(morph, 'MorphTargetBlendWeightConstraintSet', 'blendWeightConstraintSet')
            collection(constraints, 'carray_list', '_constraints')
            extensions = collection(node, 'carray_list', '_extensions')
            sub(node, 'null', 'internalColor_direct_argb')
            label_colour(node)
            return node, extensions

        def form_base(parent, form_guid, source):
            node = sub(parent, 'ACForm', 'super')
            ref(node, form_guid, 'guid')
            sub(node, 'b', 'isAnimatedForm', 'false')
            sub(node, 'b', 'isLocalAnimatedForm', 'false')
            ref(node, source, '_source')
            sub(node, 'null', 'name')
            sub(node, 's', 'notes', '')

        def draw_form_fields(parent, draw_order, opacity):
            sub(parent, 'i', 'drawOrder', draw_order)
            sub(parent, 'f', 'opacity', number(opacity))
            sub(parent, 'CFloatColor', 'multiplyColor', red='1.0', green='1.0', blue='1.0', alpha='1.0')
            sub(parent, 'CFloatColor', 'screenColor', red='0.0', green='0.0', blue='0.0', alpha='1.0')

        # ---------------------------------------------------------- shared basics
        root_group_guid = x.shared('CParameterGroupGuid', uuid=ROOT_PARAMETER_GROUP_UUID, note='Root Parameter Group')
        root_part_guid = x.shared('CPartGuid', uuid=new_uuid(), note='__RootPart__')
        root_deformer_guid = x.shared('CDeformerGuid', uuid=ROOT_DEFORMER_UUID, note='ROOT')
        coord_basic = x.shared('CoordType')
        sub(coord_basic, 's', 'coordName', 'Basic Coord')
        coord_local = x.shared('CoordType')
        sub(coord_local, 's', 'coordName', 'DeformerLocal')
        blend_normal = x.shared('CBlend_Normal')
        sub(blend_normal, 'ACBlend', 'super')

        for parameter in self.parameters:
            parameter.guid = x.shared('CParameterGuid', uuid=new_uuid(), note=parameter.id)
            parameter.id_node = x.shared('CParameterId', idstr=parameter.id)

        # ---------------------------------------------------------- filter definitions
        selector_def = x.shared('StaticFilterDefGuid', uuid=FILTER_DEF_LAYER_SELECTOR, note='CLayerSelector')
        filter_def = x.shared('StaticFilterDefGuid', uuid=FILTER_DEF_LAYER_FILTER, note='CLayerFilter')
        value_id = {text: x.shared('FilterValueId', idstr=text) for text in (
            'ilf_outputLayerData', 'mi_input_layerInputData', 'ilf_inputLayerData', 'mi_currentImageGuid',
            'ilf_currentImageGuid', 'mi_output_image', 'mi_output_transform', 'ilf_inputLayer')}

        def filter_value(label, id_text, inline=False):
            node = x.shared('FilterValue')
            sub(node, 's', 'name', label)
            if inline:
                sub(node, 'FilterValueId', 'id', idstr=id_text)
            else:
                ref(node, value_id[id_text], 'id')
            sub(node, 'null', 'defaultValueInitializer')
            return node

        value_select = filter_value('Select Layer', 'ilf_outputLayerData')
        value_import = filter_value('Import Layer', 'mi_input_layerInputData')
        value_import_select = filter_value('Import Layer selection', 'ilf_inputLayerData')
        value_current_guid = filter_value('Current GUID', 'mi_currentImageGuid')
        value_selected_guid = filter_value('GUID of Selected Source Image', 'ilf_currentImageGuid')
        value_out_image = filter_value('Output image', 'mi_output_image')
        value_out_resource = filter_value('Output Image (Resource Format)', 'ilf_outputImageRes', inline=True)
        value_out_transform = filter_value('LayerToCanvas', 'mi_output_transform')
        value_out_transform2 = filter_value('LayerToCanvas', 'ilf_outputTransform', inline=True)

        # ---------------------------------------------------------- the layered picture ("PSD")
        layered_guid = x.shared('CLayeredImageGuid', uuid=new_uuid(), note=os.path.basename(self.psd_path))
        layered = x.shared('CLayeredImage')
        layer_group = x.shared('CLayerGroup')
        empty_options = dict(count='0', keyType='string')

        def layer_entry(parent, name, group):
            entry = sub(parent, 'ACLayerEntry', 'super')
            sub(entry, 's', 'name', name)
            sub(entry, 's', 'memo', '')
            sub(entry, 'b', 'isVisible', 'true')
            sub(entry, 'b', 'isClipping', 'false')
            ref(entry, blend_normal, 'blend')
            sub(entry, 'CLayerGuid', 'guid', uuid=new_uuid(), note='(no debug info)')
            if group is None:
                sub(entry, 'null', 'group')
            else:
                ref(entry, group, 'group')
            sub(entry, 'i', 'opacity255', 255)
            sub(entry, 'b', 'isTransparencyShapesLayer', 'true')
            sub(entry, 'hash_map', '_optionOfIOption', **empty_options)
            ref(entry, layered, '_layeredImage')

        for mesh in self.meshes:
            height, width = mesh.image.shape[:2]
            mesh.resource = image_resource(mesh.image)
            mesh.icon16 = icon(mesh.image, 16)
            mesh.icon32 = icon(mesh.image, 32)
            mesh.icon64 = icon(mesh.image, 64)
            layer = mesh.layer = x.shared('CLayer')
            layer_entry(sub(layer, 'ACImageLayer', 'super'), mesh.name, layer_group)
            ref(layer, mesh.resource, 'imageResource')
            bounds = sub(layer, 'CRect', 'boundsOnImageDoc')
            sub(bounds, 'i', 'x', mesh.origin[0])
            sub(bounds, 'i', 'y', mesh.origin[1])
            sub(bounds, 'i', 'width', width)
            sub(bounds, 'i', 'height', height)
            identifier = sub(layer, 'CLayerIdentifier', 'layerIdentifier')
            sub(identifier, 's', 'layerName', mesh.name)
            sub(identifier, 'null', 'layerId')
            sub(identifier, 'i', 'layerIdValue_testImpl', -1)
            ref(layer, mesh.icon16, 'icon16')
            ref(layer, mesh.icon64, 'icon64')
            sub(layer, 'linked_map', 'layerInfo', **empty_options)
            sub(layer, 'hash_map', '_optionOfIOption', **empty_options)

        group_super = sub(layer_group, 'ACLayerGroup', 'super')
        layer_entry(group_super, 'root', None)
        children = collection(group_super, 'carray_list', '_children')
        for mesh in self.meshes:
            ref(children, mesh.layer)
        sub(layer_group, 'null', 'layerIdentifier')

        sub(layered, 's', 'name', os.path.basename(self.psd_path))
        sub(layered, 's', 'memo', '')
        sub(layered, 'i', 'width', self.width)
        sub(layered, 'i', 'height', self.height)
        sub(layered, 'file', 'psdFile', self.psd_path)
        sub(layered, 's', 'description', '')
        ref(layered, layered_guid, 'guid')
        sub(layered, 'null', 'psdBytes')
        sub(layered, 'l', 'psdFileLastModified', psd_ms)
        ref(layered, layer_group, '_rootLayer')
        layer_set = sub(layered, 'LayerSet', 'layerSet')
        ref(layer_set, layered, '_layeredImage')
        entries = collection(layer_set, 'carray_list', '_layerEntryList')
        ref(entries, layer_group)
        for mesh in self.meshes:
            ref(entries, mesh.layer)
        sub(layered, 'null', 'icon16')
        sub(layered, 'null', 'icon64')

        # ---------------------------------------------------------- model images (one per layer)
        image_group = x.shared('CModelImageGroup')
        sub(image_group, 's', 'memo', '')
        sub(image_group, 's', 'groupName', self.name)
        ref(collection(image_group, 'carray_list', '_linkedRawImageGuids'), layered_guid)
        model_images = collection(image_group, 'carray_list', '_modelImages')

        for mesh in self.meshes:
            height, width = mesh.image.shape[:2]
            mesh.image_guid = x.shared('CModelImageGuid', uuid=new_uuid(), note=mesh.name)
            filter_set = x.shared('ModelImageFilterSet')
            selector_id = x.shared('FilterInstanceId', idstr='filter0')
            selector = x.shared('FilterInstance', filterName='CLayerSelector')
            output = x.shared('FilterOutputValueConnector')
            filter_id = x.shared('FilterInstanceId', idstr='filter1')
            layer_filter = x.shared('FilterInstance', filterName='CLayerFilter')

            sub(output, 'AValueConnector', 'super')
            ref(output, selector, 'instance')
            ref(output, value_id['ilf_outputLayerData'], 'id')
            ref(output, value_select, 'valueDef')

            ref(selector, selector_def, 'filterDefGuid')
            sub(selector, 'null', 'filterDef')
            ref(selector, selector_id, 'filterId')
            inputs = collection(selector, 'hash_map', 'inputConnectors')
            for key, env in (('ilf_inputLayerData', 'mi_input_layerInputData'),
                             ('ilf_currentImageGuid', 'mi_currentImageGuid')):
                entry = sub(inputs, 'entry')
                ref(entry, value_id[key], 'key')
                connector = sub(entry, 'EnvValueConnector', 'value')
                sub(connector, 'AValueConnector', 'super')
                ref(connector, value_id[env], 'envValueId')
            entry = sub(collection(selector, 'hash_map', 'outputConnectors'), 'entry')
            ref(entry, value_id['ilf_outputLayerData'], 'key')
            ref(entry, output, 'value')
            ref(selector, filter_set, 'ownerFilterSet')

            ref(layer_filter, filter_def, 'filterDefGuid')
            sub(layer_filter, 'null', 'filterDef')
            ref(layer_filter, filter_id, 'filterId')
            entry = sub(collection(layer_filter, 'hash_map', 'inputConnectors'), 'entry')
            ref(entry, value_id['ilf_inputLayer'], 'key')
            ref(entry, output, 'value')
            sub(layer_filter, 'hash_map', 'outputConnectors', **empty_options)
            ref(layer_filter, filter_set, 'ownerFilterSet')

            set_super = sub(filter_set, 'FilterSet', 'super')
            filter_map = collection(set_super, 'linked_map', 'filterMap')
            for key, value in ((selector_id, selector), (filter_id, layer_filter)):
                entry = sub(filter_map, 'entry')
                ref(entry, key, 'key')
                ref(entry, value, 'value')
            for list_name, rows in (
                    ('_externalInputs', (('mi_input_layerInputData', value_import, selector, value_import_select),
                                         ('mi_currentImageGuid', value_current_guid, selector, value_selected_guid))),
                    ('_externalOutputs', (('mi_output_image', value_out_image, layer_filter, value_out_resource),
                                          ('mi_output_transform', value_out_transform, layer_filter, value_out_transform2)))):
                connections = collection(set_super, 'linked_map', list_name)
                for key, env_def, instance, value_def in rows:
                    entry = sub(connections, 'entry')
                    ref(entry, value_id[key], 'key')
                    connection = sub(entry, 'EnvConnection', 'value')
                    ref(connection, env_def, '_envValueDef')
                    ref(connection, instance, 'filter')
                    ref(connection, value_def, 'filterValueDef')

            model_image = sub(model_images, 'CModelImage', modelImageVersion='0')
            ref(model_image, mesh.image_guid, 'guid')
            sub(model_image, 's', 'name', mesh.name)
            ref(model_image, filter_set, 'inputFilter')
            environment = sub(sub(model_image, 'ModelImageFilterEnv', 'inputFilterEnv'), 'FilterEnv', 'super')
            sub(environment, 'null', 'parentEnv')
            values = collection(environment, 'hash_map', 'envValues')
            entry = sub(values, 'entry')
            ref(entry, value_id['mi_currentImageGuid'], 'key')
            value_set = sub(entry, 'EnvValueSet', 'value')
            ref(value_set, value_id['mi_currentImageGuid'], 'id')
            ref(value_set, layered_guid, 'value')
            sub(value_set, 'l', 'updateTimeMs', now_ms)
            entry = sub(values, 'entry')
            ref(entry, value_id['mi_input_layerInputData'], 'key')
            value_set = sub(entry, 'EnvValueSet', 'value')
            ref(value_set, value_id['mi_input_layerInputData'], 'id')
            selector_map = sub(value_set, 'CLayerSelectorMap', 'value')
            entry = sub(collection(selector_map, 'linked_map', '_imageToLayerInput'), 'entry')
            ref(entry, layered_guid, 'key')
            layer_input = sub(collection(entry, 'array_list', 'value'), 'CLayerInputData')
            ref(layer_input, mesh.layer, 'layer')
            affine(layer_input, 'affine')
            sub(layer_input, 'null', 'clippingOnTexturePx')
            sub(value_set, 'l', 'updateTimeMs', now_ms)
            ref(model_image, mesh.resource, '_filteredImage')
            ref(model_image, mesh.icon16, 'icon16')
            affine(model_image, '_materialLocalToCanvasTransform', mesh.origin[0], mesh.origin[1])
            ref(model_image, image_group, '_group')
            ref(collection(model_image, 'carray_list', 'linkedRawImageGuids'), layered_guid)
            cached_image_manager(model_image, mesh.resource, width, height)
            sub(model_image, 's', 'memo', '')

        # ---------------------------------------------------------- texture atlas
        atlas_resource = image_resource(self.atlas_image)
        atlas_guid = x.shared('CTextureAtlasGuid', uuid=new_uuid(), note='Texture0')
        atlas = x.shared('CTextureAtlas')
        sub(atlas, 's', 'name', 'Texture0')
        sub(atlas, 'i', 'width', page)
        sub(atlas, 'i', 'height', page)
        ref(atlas, atlas_resource, 'cachedAtlasImage')
        sub(atlas, 'b', 'lockCachedAtlasImage', 'false')
        ref(atlas, atlas_guid, 'guid')
        atlas_entries = collection(atlas, 'carray_list', 'modelImages')
        for mesh in self.meshes:
            entry = sub(atlas_entries, 'ModelImageEntry')
            ref(entry, atlas, 'atlas')
            ref(entry, mesh.image_guid, 'modelImageGuid')
            affine(entry, 'atlasLocalToCanvasTransform', *mesh.atlas_to_canvas)
            transform = sub(entry, 'GTransform2', 'materialLocalToAtlasTransform')
            position = sub(transform, 'GVector2', 'position')
            sub(position, 'f', 'x', number(mesh.material_in_atlas[0]))
            sub(position, 'f', 'y', number(mesh.material_in_atlas[1]))
            scale = sub(transform, 'GVector2', 'scale')
            sub(scale, 'f', 'x', '1.0')
            sub(scale, 'f', 'y', '1.0')
            sub(transform, 'f', 'eulerAngle', '0.0')
        cached_image_manager(atlas, atlas_resource, page, page)

        texture = x.shared('GTexture2D')
        texture_super = sub(texture, 'GTexture', 'super')
        sub(texture_super, 's', 'name', 'Texture0')
        sub(texture_super, 'WrapMode', 'wrapMode', v='CLAMP_TO_BORDER')
        filter_mode = sub(texture_super, 'FilterMode', 'filterMode')
        ref(filter_mode, texture, 'owner')
        sub(filter_mode, 'MinFilter', 'minFilter', v='LINEAR_MIPMAP_LINEAR')
        sub(filter_mode, 'MagFilter', 'magFilter', v='LINEAR')
        sub(texture_super, 'GTextureGuid', 'guid', uuid=new_uuid(), note='(no debug info)')
        sub(texture_super, 'Anisotropy', 'anisotropy', v='ON')
        ref(texture, atlas_resource, 'srcImageResource')
        sub(texture, 'CAffine', 'transformImageResource01toLogical01', **IDENTITY)
        sub(texture, 'i', 'mipmapLevel', 64)
        sub(texture, 'b', 'isPremultiplied', 'true')

        # ---------------------------------------------------------- parts
        def part_source(local_name, part_id, guid, parent_guid):
            form_guid = x.shared('CFormGuid', uuid=new_uuid(), note=f'{part_id}_form')
            part = x.shared('CPartSource')
            controllable(part, local_name, parent_guid, [((), form_guid)])
            ref(part, guid, 'guid')
            sub(part, 'CPartId', 'id', idstr=part_id)
            form = sub(collection(part, 'carray_list', 'keyforms'), 'CPartForm')
            form_base(form, form_guid, part)
            draw_form_fields(form, 500, 1.0)
            sub(part, 'b', 'enableDrawOrderGroup', 'false')
            sub(part, 'i', 'defaultOrder_forEditor', 500)
            sub(part, 'b', 'isSketch', 'false')
            sub(part, 'CColor', 'partsEditColor')
            child_guids = collection(part, 'carray_list', '_childGuids')
            sub(part, 'b', 'useOffscreen', 'false')
            collection(part, 'carray_list', 'clipGuidList')
            sub(part, 'b', 'invertClippingMask', 'false')
            sub(part, 'ColorComposition', 'colorComposition', v='NORMAL')
            sub(part, 'AlphaComposition', 'alphaComposition', v='OVER')
            return part, child_guids

        root_part, root_children = part_source('Root Part', '__RootPart__', root_part_guid, None)
        extra_part_guid = x.shared('CPartGuid', uuid=new_uuid(), note='Part')
        extra_part, _ = part_source(self.name, 'Part', extra_part_guid, root_part_guid)
        ref(root_children, extra_part_guid)

        # ---------------------------------------------------------- warp deformers
        stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
        time_text = stamp.strftime('%Y-%m-%dT%H:%M:%S.%f') + '000+08:00[Asia/Shanghai]'

        def write_rotation(rotation):
            cells, shared_grid = key_cells(rotation, '(no debug info)')
            source = rotation.source = x.shared('CRotationDeformerSource')
            original = x.shared('float-array', count=2)
            original.text = numbers(rotation.rest[:2])          # the pivot in canvas pixels
            deformer = sub(source, 'ACDeformerSource', 'super')
            _, extensions = controllable(deformer, rotation.name, root_part_guid, cells, shared_grid)
            shape_extension = sub(extensions, 'CDeformerOriginalShapeExtension')
            extension_base(shape_extension, source)
            shape = sub(shape_extension, 'RotationDeformerOriginalShape', 'originalShape')
            for holder in (sub(shape, 'ACDeformerOriginalShape', 'super'), shape):   # the Editor writes both
                ref(holder, original, 'positions')
                sub(holder, 's', 'lastUpdatedTimeString', time_text)
                sub(holder, 'b', 'autoRefreshIsAvailable', 'true')
                sub(holder, 'CreationMethod', 'creationMethod',
                    v='AFTER_KEY_ADDED' if rotation.bindings else 'DEFORMER_CREATED')
                sub(holder, 'f', 'baseAngle', '0.0')
                sub(holder, 'f', 'globalAngle', number(rotation.rest[2]))
                sub(holder, 'f', 'globalScale', number(rotation.rest[3]))
                sub(holder, 'f', 'circleRadiusTimesScale', number(rotation.handle / 3))
                sub(holder, 'f', 'handleLength', number(rotation.handle))
                sub(holder, 'b', 'isReflectX', 'false')
                sub(holder, 'b', 'isReflectY', 'false')
            ref(deformer, rotation.guid, 'guid')
            sub(deformer, 'CDeformerId', 'id', idstr=rotation.id)
            ref(deformer, rotation.parent.guid if rotation.parent else root_deformer_guid, 'targetDeformerGuid')
            sub(source, 'b', 'useBoneUi_testImpl', 'true')
            # Angle and scale are relative to a rotation parent; the pivot is in the parent's space
            # (pixels from its pivot, 0..1 of a warp's box, canvas pixels at the root).
            parent = rotation.parent
            parent_angle, parent_scale = (parent.rest[2], parent.rest[3]) if isinstance(parent, Rotation) else (0.0, 1.0)
            forms = collection(source, 'carray_list', 'keyforms')
            for index, form_guid in cells:
                values = rotation.form_at(index)
                origin = in_parent_space(rotation, values[:2])
                form = sub(forms, 'CRotationDeformerForm', angle=number(values[2] - parent_angle),
                           originX=number(origin[0]), originY=number(origin[1]),
                           scale=number(values[3] / parent_scale), isReflectX='false', isReflectY='false')
                form_super = sub(form, 'ACDeformerForm', 'super')
                form_base(form_super, form_guid, source)
                sub(form_super, 'f', 'opacity', '1.0')
                sub(form_super, 'CFloatColor', 'multiplyColor', red='1.0', green='1.0', blue='1.0', alpha='1.0')
                sub(form_super, 'CFloatColor', 'screenColor', red='0.0', green='0.0', blue='0.0', alpha='1.0')
                ref(form_super, coord_local, 'coordType')
            sub(source, 'f', 'handleLengthOnCanvas', number(rotation.handle))
            sub(source, 'f', 'circleRadiusOnCanvas', number(rotation.handle / 3))
            sub(source, 'f', 'baseAngle', '0.0')
        deformers = self._ordered_deformers()
        for deformer in deformers:
            deformer.guid = x.shared('CDeformerGuid', uuid=new_uuid(), note=deformer.id)
        for deformer in deformers:
            ref(root_children, deformer.guid)
            if isinstance(deformer, Rotation):
                write_rotation(deformer)
                continue
            warp = deformer
            point_count = (warp.cols + 1) * (warp.rows + 1)
            cells, shared_grid = key_cells(warp, '(no debug info)')
            source = warp.source = x.shared('CWarpDeformerSource')
            original = x.shared('float-array', count=point_count * 2)
            original.text = numbers(warp.rest)                  # canvas pixels, whatever the parent

            deformer = sub(source, 'ACDeformerSource', 'super')
            _, extensions = controllable(deformer, warp.name, root_part_guid, cells, shared_grid)
            for level, bezier_cols, bezier_rows in ((2, 2, 2), (3, 1, 1)):
                bezier = sub(extensions, 'CWarpDeformerBezierExtension')
                extension_base(bezier, source)
                sub(bezier, 'i', 'editLevel', level)
                sub(bezier, 'i', 'bezierCol', bezier_cols)
                sub(bezier, 'i', 'bezierRow', bezier_rows)
            shape_extension = sub(extensions, 'CDeformerOriginalShapeExtension')
            extension_base(shape_extension, source)
            shape = sub(shape_extension, 'WarpDeformerOriginalShape', 'originalShape')
            for holder in (sub(shape, 'ACDeformerOriginalShape', 'super'), shape):   # the Editor writes both
                ref(holder, original, 'positions')
                sub(holder, 's', 'lastUpdatedTimeString', time_text)
                sub(holder, 'b', 'autoRefreshIsAvailable', 'true')
                sub(holder, 'CreationMethod', 'creationMethod',
                    v='AFTER_KEY_ADDED' if warp.bindings else 'DEFORMER_CREATED')
                sub(holder, 'i', 'col', warp.cols)
                sub(holder, 'i', 'row', warp.rows)
            ref(deformer, warp.guid, 'guid')
            sub(deformer, 'CDeformerId', 'id', idstr=warp.id)
            ref(deformer, warp.parent.guid if warp.parent else root_deformer_guid, 'targetDeformerGuid')

            sub(source, 'i', 'col', warp.cols)
            sub(source, 'i', 'row', warp.rows)
            sub(source, 'b', 'isQuadTransform', 'true')
            forms = collection(source, 'carray_list', 'keyforms')
            for index, form_guid in cells:
                form = sub(forms, 'CWarpDeformerForm')
                form_super = sub(form, 'ACDeformerForm', 'super')
                form_base(form_super, form_guid, source)
                sub(form_super, 'f', 'opacity', '1.0')
                sub(form_super, 'CFloatColor', 'multiplyColor', red='1.0', green='1.0', blue='1.0', alpha='1.0')
                sub(form_super, 'CFloatColor', 'screenColor', red='0.0', green='0.0', blue='0.0', alpha='1.0')
                ref(form_super, coord_local, 'coordType')
                # Grid points row by row from the top-left; canvas pixels at the root, 0..1 of
                # the parent's box under another warp.
                sub(form, 'float-array', 'positions', numbers(in_parent_space(warp, warp.form_at(index))),
                    count=point_count * 2)

        # ---------------------------------------------------------- art meshes
        for mesh in self.meshes:
            mesh.guid = x.shared('CDrawableGuid', uuid=new_uuid(), note=mesh.name)
        for mesh in self.meshes:
            ref(root_children, mesh.guid)
            count = len(mesh.vertices)
            cells, shared_grid = key_cells(mesh, f'{mesh.name}_form')
            source = mesh.source = x.shared('CArtMeshSource')
            texture_input = x.shared('CTextureInputExtension')
            region_input = x.shared('CTextureInput_TextureAtlasRegion')

            drawable = sub(source, 'ACDrawableSource', 'super')
            _, extensions = controllable(drawable, mesh.name, root_part_guid, cells, shared_grid)

            editable = sub(extensions, 'CEditableMeshExtension')
            extension_base(editable, source)
            edit_mesh = sub(editable, 'GEditableMesh2', 'editableMesh', nextPointUid=count,
                            useDelaunayTriangulation='true')
            edges = edges_of(mesh.triangles)
            sub(edit_mesh, 'float-array', 'point', numbers(mesh.vertices), count=count * 2)
            sub(edit_mesh, 'byte-array', 'pointPriority', ' '.join(['20'] * count), count=count)
            sub(edit_mesh, 'short-array', 'edge', ' '.join(f'{a} {b}' for a, b in edges), count=len(edges) * 2)
            sub(edit_mesh, 'byte-array', 'edgePriority', ' '.join(['30'] * len(edges)), count=len(edges))
            sub(edit_mesh, 'int-array', 'pointUid', ' '.join(str(i) for i in range(count)), count=count)
            sub(edit_mesh, 'GEditableMeshGuid', 'meshGuid', uuid=new_uuid(), note='(no debug info)')
            ref(edit_mesh, coord_basic, 'coordType')
            sub(editable, 'b', 'isLocked', 'false')

            ref(extensions, texture_input)
            extension_base(texture_input, source)
            inputs = collection(texture_input, 'carray_list', '_textureInputs')
            image_input = sub(inputs, 'CTextureInput_ModelImage')
            image_input_super = sub(image_input, 'ACTextureInput', 'super')
            affine(image_input_super, 'optionalTransformOnCanvas')
            ref(image_input_super, texture_input, '_owner')
            ref(image_input, mesh.image_guid, '_modelImageGuid')
            ref(inputs, region_input)
            ref(texture_input, region_input, 'currentTextureInputData')
            region_super = sub(region_input, 'ACTextureInput', 'super')
            affine(region_super, 'optionalTransformOnCanvas')
            ref(region_super, texture_input, '_owner')
            ref(region_input, atlas_guid, 'textureAtlasGuid')
            affine(region_input, 'inputImageLocalToCanvasTransform', *mesh.atlas_to_canvas)

            generator = sub(extensions, 'CMeshGeneratorExtension')
            extension_base(generator, source)
            setting = sub(generator, 'MeshGenerateSetting', 'meshGenerateSetting')
            for field, value in (('polygonOuterDensity', 100), ('polygonInnerDensity', 100), ('polygonMargin', 20),
                                 ('polygonInnerMargin', 20), ('polygonMinMargin', 5), ('polygonMinBoundsPt', 5),
                                 ('thresholdAlpha', 0)):
                sub(setting, 'i', field, value)

            sub(drawable, 'CDrawableId', 'id', idstr=mesh.id)
            ref(drawable, mesh.guid, 'guid')
            ref(drawable, mesh.parent.guid if mesh.parent else root_deformer_guid, 'targetDeformerGuid')
            clip_list = collection(drawable, 'carray_list', 'clipGuidList')
            for mask in mesh.clip:
                ref(clip_list, mask.guid)
            sub(drawable, 'b', 'invertClippingMask', 'false')
            ref(drawable, mesh.icon32, 'icon32')
            ref(drawable, mesh.icon16, 'icon16')

            sub(source, 'int-array', 'indices', ' '.join(str(i) for i in mesh.triangles.ravel()),
                count=mesh.triangles.size)
            forms = collection(source, 'carray_list', 'keyforms')
            for index, form_guid in cells:
                form = sub(forms, 'CArtMeshForm')
                form_super = sub(form, 'ACDrawableForm', 'super')
                form_base(form_super, form_guid, source)
                draw_form_fields(form_super, mesh.draw_order, mesh.opacity)
                ref(form_super, coord_local, 'coordType')
                # Form positions are in the parent deformer's space; the two arrays below
                # (and the editable mesh above) stay in canvas pixels.
                sub(form, 'float-array', 'positions', numbers(in_parent_space(mesh, mesh.form_at(index))),
                    count=count * 2)
            sub(source, 'float-array', 'positions', numbers(mesh.vertices), count=count * 2)
            uvs = (mesh.vertices - np.array(mesh.atlas_to_canvas, np.float32)) / page
            mesh.uvs = uvs
            sub(source, 'float-array', 'uvs', numbers(uvs), count=count * 2)
            ref(source, texture, 'texture')
            sub(source, 'ColorComposition', 'colorComposition', v='NORMAL')
            sub(source, 'b', 'culling', 'false')
            sub(source, 'TextureState', 'textureState', v='TEXTURE_ATLAS')
            sub(source, 's', 'userData', '')
            sub(source, 'AlphaComposition', 'alphaComposition', v='OVER')

        # ---------------------------------------------------------- parameter group
        root_group_id = x.shared('CParameterGroupId', idstr='ParamGroupRoot')
        root_group = x.shared('CParameterGroup')
        sub(root_group, 's', 'name', 'Root Parameter Group')
        sub(root_group, 's', 'description', '')
        sub(root_group, 'b', 'folderIsOpened', 'false')
        ref(root_group, root_group_guid, 'guid')
        sub(root_group, 'null', 'parentGroupGuid')
        group_children = collection(root_group, 'carray_list', '_childGuids')
        for parameter in self.parameters:
            ref(group_children, parameter.guid)
        ref(root_group, root_group_id, 'id')
        label_colour(root_group)

        # ---------------------------------------------------------- the model
        root = Node('root', {'fileFormatVersion': FILE_FORMAT_VERSION})
        shared_section = sub(root, 'shared')
        model = sub(sub(root, 'main'), 'CModelSource', isDefaultKeyformLocked='false')
        sub(model, 'CModelGuid', 'guid', uuid=new_uuid(), note='(no debug info)')
        sub(model, 's', 'name', self.name)
        sub(sub(model, 'EditorEdition', 'editorEdition'), 'i', 'edition', 12)
        canvas = sub(model, 'CImageCanvas', 'canvas')
        sub(canvas, 'i', 'pixelWidth', self.width)
        sub(canvas, 'i', 'pixelHeight', self.height)
        sub(canvas, 'CColor', 'background')

        sources = collection(sub(model, 'CParameterSourceSet', 'parameterSourceSet'), 'carray_list', '_sources')
        for parameter in self.parameters:
            node = sub(sources, 'CParameterSource')
            sub(node, 'i', 'decimalPlaces', 3)
            ref(node, parameter.guid, 'guid')
            sub(node, 'f', 'snapEpsilon', '0.001')
            sub(node, 'f', 'minValue', number(parameter.minimum))
            sub(node, 'f', 'maxValue', number(parameter.maximum))
            sub(node, 'f', 'defaultValue', number(parameter.default))
            sub(node, 'b', 'isRepeat', 'false')
            ref(node, parameter.id_node, 'id')
            sub(node, 'Type', 'paramType', v='NORMAL')
            sub(node, 's', 'name', parameter.name)
            sub(node, 's', 'description', parameter.description)
            sub(node, 'b', 'combined', 'false')
            ref(node, root_group_guid, 'parentGroupGuid')

        manager = sub(model, 'CTextureManager', 'textureManager')
        collection(sub(manager, 'TextureImageGroup', 'textureList'), 'carray_list', 'children')
        wrapper = sub(collection(manager, 'carray_list', '_rawImages'), 'LayeredImageWrapper')
        ref(wrapper, layered, 'image')
        sub(wrapper, 'l', 'importedTimeMSec', now_ms)
        sub(wrapper, 'l', 'lastModifiedTimeMSec', psd_ms)
        sub(wrapper, 'b', 'isReplaced', 'false')
        ref(collection(manager, 'carray_list', '_modelImageGroups'), image_group)
        ref(collection(manager, 'carray_list', '_textureAtlases'), atlas)
        sub(manager, 'b', 'isTextureInputModelImageMode', 'false')
        sub(manager, 'i', 'previewReductionRatio', 1)
        collection(manager, 'carray_list', 'artPathBrushUsingLayeredImageIds')

        sub(model, 'b', 'useLegacyDrawOrder__testImpl', 'false')
        sources = collection(sub(model, 'CDrawableSourceSet', 'drawableSourceSet'), 'carray_list', '_sources')
        for mesh in self.meshes:
            ref(sources, mesh.source)
        sources = collection(sub(model, 'CDeformerSourceSet', 'deformerSourceSet'), 'carray_list', '_sources')
        for deformer in deformers:
            ref(sources, deformer.source)
        collection(sub(model, 'CAffecterSourceSet', 'affecterSourceSet'), 'carray_list', '_sources')
        sources = collection(sub(model, 'CPartSourceSet', 'partSourceSet'), 'carray_list', '_sources')
        ref(sources, root_part)
        ref(sources, extra_part)
        physics = sub(model, 'CPhysicsSettingsSourceSet', 'physicsSettingsSourceSet')
        collection(physics, 'carray_list', '_sourceCubismPhysics')
        sub(physics, 'null', 'selectedCubismPhysics')
        sub(physics, 'i', 'settingFPS', 60)
        ref(model, root_part, 'rootPart')
        ref(collection(sub(model, 'CParameterGroupSet', 'parameterGroupSet'), 'carray_list', '_groups'), root_group)
        ref(model, root_group, 'rootParameterGroup')

        info = sub(model, 'CModelInfo', 'modelInfo')
        sub(info, 'f', 'pixelsPerUnit', '1.0')
        origin = sub(info, 'CPoint', 'originInPixels')
        sub(origin, 'i', 'x', 0)
        sub(origin, 'i', 'y', 0)
        effect_groups = collection(sub(info, 'CEffectParameterGroups', '_effectParameterGroups'),
                                   'hash_map', '_parameterGroups')
        entry = sub(effect_groups, 'entry')
        sub(entry, 'EffectParameterType', 'key', v='EYE_BLINK')
        collection(entry, 'carray_list', 'value')

        options = collection(model, 'hash_map', 'modelOptions', keyType='string')
        export = sub(options, 'CModelExportSettingDialogData', 'MODEL_EXPORT_SETTING_KEY')
        sub(export, 'FormatVersion', 'version', v='VERSION_50')
        for field, value in (('exportInvisiblePart', 'false'), ('exportSketchPart', 'false'),
                             ('exportInvisibleArtMesh', 'false'), ('exportPhysicsJson', 'true'),
                             ('exportUserDataJson', 'true'), ('exportDisplayInfoJson', 'true'),
                             ('exportTrackCursorJson', 'true')):
            sub(export, 'b', field, value)
        sub(export, 'f', 'normalOriginX', '0.5')
        sub(export, 'f', 'normalOriginY', '0.5')
        sub(export, 'i', 'exportTextureScale', 1)
        sub(export, 'f', 'pixelsPerUnit', number(self.width))
        sub(export, 'CMocExportType', 'exportType', v='SDK')
        sub(export, 'i', 'exportNizimaPreviewTextureScale', 2)
        sub(export, 'b', 'enableExtendEdgeColor', 'true')
        sub(export, 'b', 'exportMotionSyncJson', 'false')

        composite = self.composite()
        for field, size in (('_icon64', 64), ('_icon32', 32), ('_icon16', 16)):
            ref(model, icon(composite, size), field)
        motions = sub(model, 'CGameMotionSet', 'gameMotionSet')
        collection(motions, 'carray_list', 'gameMotions')
        collection(motions, 'carray_list', 'gameMotionGroups')
        tracking = collection(sub(model, 'ModelViewerSetting', 'modelViewerSetting'), 'array_list', 'trackCursorSettings')
        for parameter in self.parameters:
            node = sub(tracking, 'TrackCursorSetting')
            ref(node, parameter.guid, 'guid')
            sub(node, 'TrackCursorSettingItemType', 'value', v=parameter.track)
            sub(node, 'f', 'weight', '1.0')
        collection(sub(model, 'CGuidesSetting', 'guides'), 'carray_list', 'guidesModeling')
        sub(model, 'i', 'targetVersionNo', EDITOR_VERSION_NO)
        sub(model, 'i', 'latestVersionOfLastModelerNo', EDITOR_VERSION_NO)
        collection(sub(model, 'CArtPathBrushSetting', 'artPathBrushesSetting'), 'carray_list', 'brushes')
        random_pose = sub(model, 'CRandomPoseSettingManager', 'randomPoseSetting')
        setting = sub(collection(random_pose, 'array_list', '_settings'), 'CRandomPoseSetting')
        keys = collection(setting, 'array_list', 'parameters.keys')
        for parameter in self.parameters:
            ref(keys, parameter.id_node)
        values = collection(setting, 'array_list', 'parameters.values')
        for parameter in self.parameters:
            sub(sub(values, 'CRandomPoseParamData'), 'b', 'isEnable', 'true')
        ref(collection(setting, 'array_list', 'groups.keys'), root_group_id)
        sub(sub(collection(setting, 'array_list', 'groups.values'), 'CRandomPoseGroupData'), 'b', 'isExpand', 'true')
        sub(random_pose, 'i', 'currentIndex', 0)
        collection(sub(model, 'CMotionSyncSettingSourceSet', 'motionSyncSettingsSet'),
                   'linked_set', '_settingSourceSetMotionSync')

        shared_section.children = x.shared_nodes
        imports_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cmo3_imports.txt')
        imports = open(imports_path, encoding='utf-8').read().split()
        xml_text = x.serialize(root, imports)

        archive = caff.Archive()
        archive.key = 42
        for name, content in files:
            archive.entries.append(caff.Entry(name, '', content, True, caff.COMPRESS_RAW))
        archive.entries.append(caff.Entry('main.xml', 'main_xml', xml_text.encode('utf-8'), True, caff.COMPRESS_FAST))
        caff.write(archive, path)
        return xml_text

    def composite(self):
        """All layers flattened back to front: what the still model should look like."""
        canvas = np.zeros((self.height, self.width, 4), np.float32)
        for mesh in reversed(self.meshes):
            height, width = mesh.image.shape[:2]
            x0, y0 = mesh.origin
            cx0, cy0 = max(x0, 0), max(y0, 0)
            cx1, cy1 = min(x0 + width, self.width), min(y0 + height, self.height)
            if cx0 >= cx1 or cy0 >= cy1:
                continue
            source = mesh.image[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(np.float32) / 255.0
            alpha = source[..., 3:4] * mesh.opacity
            target = canvas[cy0:cy1, cx0:cx1]
            out_alpha = alpha + target[..., 3:4] * (1 - alpha)
            safe = np.where(out_alpha > 0, out_alpha, 1)
            target[..., :3] = (source[..., :3] * alpha + target[..., :3] * target[..., 3:4] * (1 - alpha)) / safe
            target[..., 3:4] = out_alpha
        return (canvas * 255 + 0.5).astype(np.uint8)
