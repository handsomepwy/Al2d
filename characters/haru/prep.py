"""Haru (Live2D's sample, receptionist version): the artist's import PSD -> the PSD our build reads.

    python characters/haru/prep.py

Reads  <haru_dir>/haru_import.psd   (Live2D sample data, not in this repository: private research only)
Writes out/haru_prep/haru_head.psd and check pictures beside it.

No clean-up is needed: an artist's PSD already has every part whole.  This only
  - keeps the head and shoulders (the first test is the head rig), cut to HEAD_BOX,
  - renames the layers from the artist's Japanese names to our part names (the table below),
  - leaves out what the rest pose does not show (blush, tears, arm set A, the sketches).
"""
import json
import os

import numpy as np
from PIL import Image
from psd_tools import PSDImage

from al2d import caff, cmo3xml, config
from al2d.artprep import Stack

SOURCE_PSD = os.path.join(config.need('haru_dir'), 'haru_import.psd')
OUT_DIR = config.out_dir('haru_prep')
OUT_PSD = os.path.join(OUT_DIR, 'haru_head.psd')
HEAD_BOX = (800, 190, 1520, 1130)          # left, top, right, bottom on the artist's 2400 x 4500 canvas

# Artist's name -> ours.  L / R are the character's left / right (L is on the right of the screen),
# as in See-through's -l / -r.  Names not listed are left out.
NAMES = {
    'テール': 'tail', '後ろ髪': 'back hair', '上着襟裏': 'collar back', 'スカーフ裏': 'scarf back',
    '首　体': 'neck', '上着': 'jacket', 'インナー': 'inner', '上着襟': 'collar',
    '肩B_R': 'shoulder-r', '肩B_L': 'shoulder-l',
    'スカーフ': 'scarf', 'スカーフリボン左': 'scarf ribbon a', 'スカーフリボン中': 'scarf ribbon b',
    'スカーフリボン右': 'scarf ribbon c',
    '耳L': 'ears-l', '耳R': 'ears-r', '輪郭': 'face',
    '口差分　開き': 'mouth inner', '口下': 'mouth lower', '口上': 'mouth upper', '鼻': 'nose',
    'クリッピングR': 'eye mask-r', 'クリッピングL': 'eye mask-l',
    '白目R': 'eyewhite-r', '白目L': 'eyewhite-l', '目玉L': 'irides-l', '目玉R': 'irides-r',
    '目hi下R': 'eye light low-r', '目hi上R': 'eye light high-r', '目hi下L': 'eye light low-l', '目hi上L': 'eye light high-l',
    **{f'上まつげ{n}{s}': f'lash {n}-{s.lower()}' for n in range(1, 6) for s in 'LR'},
    **{f'下まつげ{n}{s}': f'lower lash {n}-{s.lower()}' for n in range(1, 3) for s in 'LR'},
    '二重影L': 'crease shadow-l', '二重線L': 'crease-l', '二重影R': 'crease shadow-r', '二重線R': 'crease-r',
    '頬_通常R': 'cheek-r', '頬_通常L': 'cheek-l', '顔hi_R': 'face light-r', '顔hi_L': 'face light-l',
    '生え際R': 'hairline-r', '生え際L': 'hairline-l', '横髪R': 'side hair-r', '横髪L': 'side hair-l',
    '前髪': 'front hair', 'まゆ毛R': 'eyebrow-r', 'まゆ毛L': 'eyebrow-l',
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    psd = PSDImage.open(SOURCE_PSD)
    left, top, right, bottom = HEAD_BOX
    size = (right - left, bottom - top)
    layers, dropped = [], []
    for layer in psd:                                   # back to front
        artist_name = layer.name.replace('\x7f', '').strip()      # some names end in DEL characters
        name = NAMES.get(artist_name)
        if name is None or layer.is_group():
            dropped.append(artist_name)
            continue
        canvas = Image.new('RGBA', size, (0, 0, 0, 0))
        canvas.paste(layer.topil().convert('RGBA'), (layer.left - left, layer.top - top))
        image = np.array(canvas)
        if not (image[..., 3] > 0).any():
            dropped.append(artist_name + ' (outside the box)')
            continue
        layers.append((name, image))
    # The artist drew the brows above the front hair; Live2D's rig of her puts the front hair in
    # front (draw order 77 against the brows' 62-63), so the brows show only between the strands.
    layers.sort(key=lambda layer: layer[0] == 'front hair')        # stable: only the front hair moves, to the front
    stack = Stack(size, layers)
    stack.save(OUT_PSD)
    print(f'wrote {OUT_PSD}: {len(layers)} layers, {size[0]} x {size[1]}')
    print('left out:', ', '.join(dropped))
    # What the rest pose looks like: everything but the invisible eye masks.
    picture = stack.flatten(skip=('eye mask-l', 'eye mask-r'))
    Image.fromarray(picture).save(os.path.join(OUT_DIR, 'prep_rest.png'))
    write_drawable_names()


def write_drawable_names():
    """Live2D's runtime model names its drawables ArtMesh1, ArtMesh2...; the artist's layer names are
    only in the Editor project.  build.py needs the pairing to compare layer by layer, so read it
    from the sample's .cmo3 and keep it in out/haru_prep/drawable_names.json."""
    main_xml = os.path.join(OUT_DIR, 'pro_main.xml')
    archive = caff.read(os.path.join(config.need('haru_dir'), 'haru_greeter_t05.cmo3'))
    with open(main_xml, 'wb') as file:
        file.write(archive.get('main.xml').data)
    doc = cmo3xml.Doc(main_xml)
    names = {}
    for element in doc.definitions('CArtMeshSource'):
        identifier = doc.find_field(element, 'id')
        if identifier is not None and identifier.get('idstr'):
            names[identifier.get('idstr')] = doc.find_field(element, 'localName').text
    with open(os.path.join(OUT_DIR, 'drawable_names.json'), 'w', encoding='utf-8') as file:
        json.dump(names, file, ensure_ascii=False, indent=0)
    print(f'wrote drawable_names.json: {len(names)} names')


if __name__ == '__main__':
    main()
