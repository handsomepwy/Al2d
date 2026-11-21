"""Experiment for step 2: rewrite StretchyStudio art-mesh keyforms that sit under a warp deformer from
canvas pixels into the warp's own 0..1 space, and repack the .cmo3.

Usage: python fix_ss_mesh_space.py <broken.cmo3> <fixed.cmo3>
"""
import re
import sys

import numpy as np

from al2d import caff
from al2d import cmo3xml


def fmt(values):
    return ' '.join(repr(float(np.float32(v))) for v in values)


def main(source, target):
    archive = caff.read(source)
    entry = archive.get('main.xml')
    text = entry.data.decode('utf-8')
    scratch = source + '.main.xml'
    open(scratch, 'wb').write(entry.data)
    doc = cmo3xml.Doc(scratch)

    owner = {}
    for tag, kind in (('CArtMeshSource', 'mesh'), ('CWarpDeformerSource', 'warp'), ('CRotationDeformerSource', 'rot')):
        for element in doc.definitions(tag):
            owner[doc.find_field(element, 'guid')] = (kind, element)

    def name_of(element):
        return doc.find_field(element, 'localName').text

    replacements = {}   # positions element -> new text
    for kind, mesh in list(owner.values()):
        if kind != 'mesh':
            continue
        parent = owner.get(doc.find_field(mesh, 'targetDeformerGuid'))
        if not parent or parent[0] != 'warp':
            continue
        warp = parent[1]
        forms = [doc.deref(f) for f in doc.field(mesh, 'keyforms')]
        first = np.array(cmo3xml.floats(doc.field(forms[0], 'positions'))).reshape(-1, 2)
        if first.min() > -0.5 and first.max() < 1.5:
            print(f'  keep   {name_of(mesh):14s} under {name_of(warp):16s} (already 0..1)')
            continue
        columns, rows = int(doc.field(warp, 'col').text), int(doc.field(warp, 'row').text)
        box = None
        for index, warp_form in enumerate(doc.deref(f) for f in doc.field(warp, 'keyforms')):
            grid = np.array(cmo3xml.floats(doc.field(warp_form, 'positions'))).reshape(rows + 1, columns + 1, 2)
            x0, y0 = grid[0, 0]
            x1, y1 = grid[-1, -1]
            ideal_x = np.linspace(x0, x1, columns + 1)[None, :].repeat(rows + 1, 0)
            ideal_y = np.linspace(y0, y1, rows + 1)[:, None].repeat(columns + 1, 1)
            regular = np.abs(grid[:, :, 0] - ideal_x).max() < 1e-3 and np.abs(grid[:, :, 1] - ideal_y).max() < 1e-3
            if regular and box is None:
                box = (x0, y0, x1, y1, index)
        if box is None:
            print(f'  SKIP   {name_of(mesh):14s} under {name_of(warp):16s} (no keyform of the warp is a plain rectangle)')
            continue
        x0, y0, x1, y1, index = box
        for form in forms:
            element = doc.field(form, 'positions')
            points = np.array(cmo3xml.floats(element)).reshape(-1, 2)
            points[:, 0] = (points[:, 0] - x0) / (x1 - x0)
            points[:, 1] = (points[:, 1] - y0) / (y1 - y0)
            replacements[element] = fmt(points.reshape(-1))
        print(f'  fixed  {name_of(mesh):14s} under {name_of(warp):16s} warp box ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}) from its keyform {index}; '
              f'{len(forms)} keyform(s); now x {points[:,0].min():.3f}..{points[:,0].max():.3f} y {points[:,1].min():.3f}..{points[:,1].max():.3f}')

    # Text-level substitution so every other byte of main.xml stays as StretchyStudio wrote it.
    ordered = [e for e in doc.root.iter('float-array') if e.get('xs.n') == 'positions']
    pattern = re.compile(r'(<float-array xs\.n="positions" count="\d+">)([^<]*)(</float-array>)')
    matches = list(pattern.finditer(text))
    assert len(matches) == len(ordered), (len(matches), len(ordered))
    pieces, cursor = [], 0
    for match, element in zip(matches, ordered):
        if element in replacements:
            pieces.append(text[cursor:match.start(2)])
            pieces.append(replacements[element])
            cursor = match.end(2)
    pieces.append(text[cursor:])
    entry.data = ''.join(pieces).encode('utf-8')
    caff.write(archive, target)
    print(f'wrote {target} ({len(replacements)} keyform position arrays rewritten)')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
