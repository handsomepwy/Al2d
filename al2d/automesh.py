"""Make a triangle mesh for one picture layer.

The mesh is a regular grid laid over the layer's visible pixels; grid cells that hold no visible
pixel are left out.  Every visible pixel ends up inside some triangle, so nothing is cut off.
"""
import math

import cv2
import numpy as np


def grid_mesh(alpha, origin, spacing=None, margin=3, cells=7.0, min_spacing=8, max_spacing=48):
    """Builds a grid mesh over the visible part of a layer.

    alpha    (h, w) uint8 alpha channel of the layer crop.
    origin   (x, y) canvas position of alpha[0, 0].
    spacing  grid cell size in pixels; by default about sqrt(area) / cells, clamped.
    margin   visible area is grown by this many pixels first, so soft edges stay inside the mesh.

    Returns (vertices, triangles): vertices (n, 2) float32 in canvas pixels, triangles (m, 3) int32.
    """
    padded = np.pad(alpha, margin, constant_values=0)
    mask = (padded > 0).astype(np.uint8)
    if margin:
        mask = cv2.dilate(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError('layer has no visible pixels')
    left, right, top, bottom = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    width, height = right - left, bottom - top
    if spacing is None:
        spacing = min(max(math.sqrt(width * height) / cells, min_spacing), max_spacing)
    columns = max(1, math.ceil(width / spacing))
    rows = max(1, math.ceil(height / spacing))
    grid_x = np.linspace(left, right, columns + 1)
    grid_y = np.linspace(top, bottom, rows + 1)

    index_of = {}
    vertices = []

    def vertex(column, row):
        key = (column, row)
        if key not in index_of:
            index_of[key] = len(vertices)
            vertices.append((grid_x[column] - margin + origin[0], grid_y[row] - margin + origin[1]))
        return index_of[key]

    triangles = []
    for row in range(rows):
        y0, y1 = int(math.floor(grid_y[row])), int(math.ceil(grid_y[row + 1]))
        for column in range(columns):
            x0, x1 = int(math.floor(grid_x[column])), int(math.ceil(grid_x[column + 1]))
            if not mask[y0:y1, x0:x1].any():
                continue
            a, b = vertex(column, row), vertex(column + 1, row)
            c, d = vertex(column, row + 1), vertex(column + 1, row + 1)
            triangles.append((a, b, c))
            triangles.append((b, d, c))
    return np.array(vertices, np.float32), np.array(triangles, np.int32)


def edges_of(triangles):
    """Unique undirected edges of a triangle list, as a flat list of vertex index pairs."""
    seen = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            seen.add((int(min(u, v)), int(max(u, v))))
    return sorted(seen)
