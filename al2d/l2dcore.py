"""Load a .moc3 through Live2D Cubism Core's public C API and render it with numpy.

Used to check generated models without opening Cubism Editor, Cubism Viewer or VTube Studio.
Only the documented Core API (Live2DCubismCore.h from the Cubism SDK) is called.

The Core library is not part of this repository: take it from the Cubism SDK (or the copy
VTube Studio ships) and set core_dll in al2d.toml.
"""
import ctypes as C
import json
import os

import cv2
import numpy as np

from . import config

_core = None


def core():
    """Returns the loaded Core library with return types declared."""
    global _core
    if _core is None:
        lib = C.CDLL(config.need('core_dll'))
        pointer_returning = [
            "csmReviveMocInPlace", "csmInitializeModelInPlace",
            "csmGetParameterIds", "csmGetParameterMinimumValues", "csmGetParameterMaximumValues",
            "csmGetParameterDefaultValues", "csmGetParameterValues",
            "csmGetPartIds", "csmGetPartOpacities", "csmGetPartParentPartIndices",
            "csmGetDrawableIds", "csmGetDrawableConstantFlags", "csmGetDrawableDynamicFlags",
            "csmGetDrawableTextureIndices", "csmGetDrawableDrawOrders", "csmGetDrawableRenderOrders",
            "csmGetDrawableOpacities", "csmGetDrawableMaskCounts", "csmGetDrawableMasks",
            "csmGetDrawableVertexCounts", "csmGetDrawableVertexPositions", "csmGetDrawableVertexUvs",
            "csmGetDrawableIndexCounts", "csmGetDrawableIndices",
            "csmGetDrawableMultiplyColors", "csmGetDrawableScreenColors",
        ]
        for name in pointer_returning:
            getattr(lib, name).restype = C.c_void_p
        for name in ["csmGetLatestMocVersion", "csmGetMocVersion", "csmGetVersion", "csmGetSizeofModel"]:
            getattr(lib, name).restype = C.c_uint
        _core = lib
    return _core


def _aligned(size, alignment):
    """Allocates `size` bytes aligned to `alignment`; returns (owner, address)."""
    owner = C.create_string_buffer(size + alignment)
    address = C.addressof(owner)
    address += (-address) % alignment
    return owner, address


def _array(address, ctype, count):
    return np.ctypeslib.as_array(C.cast(address, C.POINTER(ctype)), shape=(count,)) if count else np.zeros(0)


def _strings(address, count):
    pointers = C.cast(address, C.POINTER(C.c_char_p))
    return [pointers[i].decode("utf-8") for i in range(count)]


def moc_file_version(path):
    with open(path, "rb") as handle:
        head = handle.read(5)
    return head[4]


class Model:
    """One model instance: parameters in, deformed vertices out."""

    def __init__(self, moc_path):
        lib = core()
        data = open(moc_path, "rb").read()
        self.path = moc_path
        self.file_version = data[4]
        self.core_latest = lib.csmGetLatestMocVersion()
        self._moc_owner, moc_address = _aligned(len(data), 64)
        C.memmove(moc_address, data, len(data))
        self.consistent = bool(lib.csmHasMocConsistency(C.c_void_p(moc_address), C.c_uint(len(data))))
        if self.file_version > self.core_latest:
            raise RuntimeError(f"{moc_path}: moc version {self.file_version} is newer than this Core reads ({self.core_latest})")
        if not self.consistent:
            raise RuntimeError(f"{moc_path}: csmHasMocConsistency failed")
        self._moc = lib.csmReviveMocInPlace(C.c_void_p(moc_address), C.c_uint(len(data)))
        if not self._moc:
            raise RuntimeError(f"{moc_path}: csmReviveMocInPlace failed")
        size = lib.csmGetSizeofModel(C.c_void_p(self._moc))
        self._model_owner, model_address = _aligned(size, 16)
        self._m = C.c_void_p(lib.csmInitializeModelInPlace(C.c_void_p(self._moc), C.c_void_p(model_address), C.c_uint(size)))
        if not self._m:
            raise RuntimeError(f"{moc_path}: csmInitializeModelInPlace failed")

        size_px = (C.c_float * 2)()
        origin_px = (C.c_float * 2)()
        ppu = C.c_float()
        lib.csmReadCanvasInfo(self._m, size_px, origin_px, C.byref(ppu))
        self.canvas = (int(round(size_px[0])), int(round(size_px[1])))
        self.origin = (origin_px[0], origin_px[1])
        self.ppu = ppu.value

        m = self._m
        n = self.param_count = lib.csmGetParameterCount(m)
        self.param_ids = _strings(lib.csmGetParameterIds(m), n)
        self.param_min = _array(lib.csmGetParameterMinimumValues(m), C.c_float, n).copy()
        self.param_max = _array(lib.csmGetParameterMaximumValues(m), C.c_float, n).copy()
        self.param_default = _array(lib.csmGetParameterDefaultValues(m), C.c_float, n).copy()
        self.param_values = _array(lib.csmGetParameterValues(m), C.c_float, n)  # live view

        p = self.part_count = lib.csmGetPartCount(m)
        self.part_ids = _strings(lib.csmGetPartIds(m), p)

        d = self.drawable_count = lib.csmGetDrawableCount(m)
        self.drawable_ids = _strings(lib.csmGetDrawableIds(m), d)
        self.constant_flags = _array(lib.csmGetDrawableConstantFlags(m), C.c_ubyte, d).copy()
        self.texture_indices = _array(lib.csmGetDrawableTextureIndices(m), C.c_int, d).copy()
        self.vertex_counts = _array(lib.csmGetDrawableVertexCounts(m), C.c_int, d).copy()
        self.index_counts = _array(lib.csmGetDrawableIndexCounts(m), C.c_int, d).copy()
        mask_counts = _array(lib.csmGetDrawableMaskCounts(m), C.c_int, d).copy()
        mask_pointers = C.cast(lib.csmGetDrawableMasks(m), C.POINTER(C.c_void_p))
        uv_pointers = C.cast(lib.csmGetDrawableVertexUvs(m), C.POINTER(C.c_void_p))
        index_pointers = C.cast(lib.csmGetDrawableIndices(m), C.POINTER(C.c_void_p))
        self.masks, self.uvs, self.indices = [], [], []
        for i in range(d):
            self.masks.append(list(_array(mask_pointers[i], C.c_int, int(mask_counts[i]))) if mask_counts[i] else [])
            self.uvs.append(_array(uv_pointers[i], C.c_float, int(self.vertex_counts[i]) * 2).reshape(-1, 2).copy())
            self.indices.append(_array(index_pointers[i], C.c_ushort, int(self.index_counts[i])).reshape(-1, 3).copy()
                                if self.index_counts[i] else np.zeros((0, 3), np.uint16))
        self.reset()

    # ---- parameters ----
    def reset(self):
        self.param_values[:] = self.param_default
        self.update()

    def set(self, **values):
        """Sets parameters by id (others keep their current value) and updates the model."""
        for name, value in values.items():
            self.param_values[self.param_ids.index(name)] = value
        self.update()

    def pose(self, values):
        """Resets to defaults, applies a {id: value} dict, updates."""
        self.param_values[:] = self.param_default
        for name, value in values.items():
            self.param_values[self.param_ids.index(name)] = value
        self.update()

    def update(self):
        lib = core()
        lib.csmResetDrawableDynamicFlags(self._m)
        lib.csmUpdateModel(self._m)

    # ---- per-frame drawable state ----
    def dynamic_flags(self):
        return _array(core().csmGetDrawableDynamicFlags(self._m), C.c_ubyte, self.drawable_count).copy()

    def render_orders(self):
        return _array(core().csmGetDrawableRenderOrders(self._m), C.c_int, self.drawable_count).copy()

    def draw_orders(self):
        return _array(core().csmGetDrawableDrawOrders(self._m), C.c_int, self.drawable_count).copy()

    def opacities(self):
        return _array(core().csmGetDrawableOpacities(self._m), C.c_float, self.drawable_count).copy()

    def vertices_px(self):
        """Deformed vertex positions per drawable, in canvas pixels (y down)."""
        pointers = C.cast(core().csmGetDrawableVertexPositions(self._m), C.POINTER(C.c_void_p))
        out = []
        for i in range(self.drawable_count):
            xy = _array(pointers[i], C.c_float, int(self.vertex_counts[i]) * 2).reshape(-1, 2).astype(np.float64)
            px = np.empty_like(xy)
            px[:, 0] = xy[:, 0] * self.ppu + self.origin[0]
            px[:, 1] = self.origin[1] - xy[:, 1] * self.ppu
            out.append(px)
        return out


def load_textures(model3_json_path):
    """Loads the atlas pages named by a model3.json as premultiplied float-free uint8 RGBA."""
    folder = os.path.dirname(model3_json_path)
    manifest = json.load(open(model3_json_path, encoding="utf-8"))
    pages = []
    for relative in manifest["FileReferences"]["Textures"]:
        raw = np.fromfile(os.path.join(folder, relative), np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        premultiplied = image.astype(np.float32)
        premultiplied[:, :, :3] *= alpha
        pages.append(premultiplied)
    return pages


def _rasterize(vertices, uvs, triangles, texture, canvas_w, canvas_h):
    """Returns (x0, y0, premultiplied RGBA float patch 0..255) for one drawable, or None."""
    if len(triangles) == 0:
        return None
    x0 = max(int(np.floor(vertices[:, 0].min())) - 1, 0)
    y0 = max(int(np.floor(vertices[:, 1].min())) - 1, 0)
    x1 = min(int(np.ceil(vertices[:, 0].max())) + 2, canvas_w)
    y1 = min(int(np.ceil(vertices[:, 1].max())) + 2, canvas_h)
    if x1 <= x0 or y1 <= y0:
        return None
    width, height = x1 - x0, y1 - y0
    map_x = np.full((height, width), -10.0, np.float32)
    map_y = np.full((height, width), -10.0, np.float32)
    tex_h, tex_w = texture.shape[:2]
    tex_xy = np.empty_like(uvs, dtype=np.float64)
    tex_xy[:, 0] = uvs[:, 0] * tex_w - 0.5
    tex_xy[:, 1] = (1.0 - uvs[:, 1]) * tex_h - 0.5
    local = vertices - np.array([x0, y0], np.float64)
    for a, b, c in triangles:
        pa, pb, pc = local[a], local[b], local[c]
        denominator = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(denominator) < 1e-9:
            continue
        bx0 = max(int(np.floor(min(pa[0], pb[0], pc[0]))), 0)
        by0 = max(int(np.floor(min(pa[1], pb[1], pc[1]))), 0)
        bx1 = min(int(np.ceil(max(pa[0], pb[0], pc[0]))) + 1, width)
        by1 = min(int(np.ceil(max(pa[1], pb[1], pc[1]))) + 1, height)
        if bx1 <= bx0 or by1 <= by0:
            continue
        grid_y, grid_x = np.mgrid[by0:by1, bx0:bx1]
        sample_x = grid_x + 0.5
        sample_y = grid_y + 0.5
        w_a = ((pb[1] - pc[1]) * (sample_x - pc[0]) + (pc[0] - pb[0]) * (sample_y - pc[1])) / denominator
        w_b = ((pc[1] - pa[1]) * (sample_x - pc[0]) + (pa[0] - pc[0]) * (sample_y - pc[1])) / denominator
        w_c = 1.0 - w_a - w_b
        epsilon = -1e-4
        inside = (w_a >= epsilon) & (w_b >= epsilon) & (w_c >= epsilon)
        if not inside.any():
            continue
        ta, tb, tc = tex_xy[a], tex_xy[b], tex_xy[c]
        u = w_a * ta[0] + w_b * tb[0] + w_c * tc[0]
        v = w_a * ta[1] + w_b * tb[1] + w_c * tc[1]
        map_x[by0:by1, bx0:bx1][inside] = u[inside]
        map_y[by0:by1, bx0:bx1][inside] = v[inside]
    patch = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    patch[map_x < -5] = 0
    return x0, y0, patch


def render(model, textures, background=(255, 255, 255), scale=1.0, region=None):
    """Renders the model's current pose.  Returns uint8 RGB (canvas size * scale).

    region  (x0, y0, x1, y1) in canvas pixels: render only that part (much faster for close-ups).
    """
    x0, y0, x1, y1 = region if region else (0, 0, model.canvas[0], model.canvas[1])
    canvas_w = int(round((x1 - x0) * scale))
    canvas_h = int(round((y1 - y0) * scale))
    vertices = [(v - np.array([x0, y0], np.float64)) * scale for v in model.vertices_px()]
    opacities = model.opacities()
    flags = model.dynamic_flags()
    order = np.argsort(model.render_orders(), kind="stable")
    color = np.zeros((canvas_h, canvas_w, 3), np.float32)  # premultiplied
    alpha = np.zeros((canvas_h, canvas_w), np.float32)
    patches = {}

    def patch_of(index):
        if index not in patches:
            patches[index] = _rasterize(vertices[index], model.uvs[index], model.indices[index],
                                        textures[model.texture_indices[index]], canvas_w, canvas_h)
        return patches[index]

    for index in order:
        if not (flags[index] & 1) or opacities[index] <= 0.0:
            continue
        result = patch_of(index)
        if result is None:
            continue
        x0, y0, patch = result
        height, width = patch.shape[:2]
        src = patch * (opacities[index] / 255.0)
        if model.masks[index]:
            mask = np.zeros((height, width), np.float32)
            for mask_index in model.masks[index]:
                if mask_index < 0:
                    continue
                mask_result = patch_of(mask_index)
                if mask_result is None:
                    continue
                mx0, my0, mask_patch = mask_result
                mh, mw = mask_patch.shape[:2]
                ix0, iy0 = max(x0, mx0), max(y0, my0)
                ix1, iy1 = min(x0 + width, mx0 + mw), min(y0 + height, my0 + mh)
                if ix1 > ix0 and iy1 > iy0:
                    region = mask_patch[iy0 - my0:iy1 - my0, ix0 - mx0:ix1 - mx0, 3] / 255.0
                    target = mask[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
                    np.maximum(target, region, out=target)
            if model.constant_flags[index] & 8:
                mask = 1.0 - mask
            src = src * mask[:, :, None]
        src_alpha = src[:, :, 3]
        dst_color = color[y0:y0 + height, x0:x0 + width]
        dst_alpha = alpha[y0:y0 + height, x0:x0 + width]
        blend = model.constant_flags[index] & 3
        if blend == 1:  # additive
            dst_color += src[:, :, :3]
        elif blend == 2:  # multiplicative
            dst_color *= (src[:, :, :3] + (1.0 - src_alpha[:, :, None]))
        else:
            dst_color *= (1.0 - src_alpha[:, :, None])
            dst_color += src[:, :, :3]
            dst_alpha *= (1.0 - src_alpha)
            dst_alpha += src_alpha
    out = color * 255.0 + (1.0 - alpha[:, :, None]) * np.array(background, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_rgb(path, image):
    cv2.imencode(os.path.splitext(path)[1], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tofile(path)
