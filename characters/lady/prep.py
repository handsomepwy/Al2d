"""Art clean-up for the second character ('lady'): See-through's PSD -> the PSD the rig is built from.

    python characters/lady/prep.py

Reads  examples/lady/seethrough.psd  (never changed)
Writes out/lady_prep/lady.psd and check pictures beside it.
"""
import os

import numpy as np
from PIL import Image

from al2d import artprep, config
from al2d.artprep import Stack, box_of, pieces

HERE = os.path.join(config.ROOT, 'examples', 'lady')
SOURCE_PSD = os.path.join(HERE, 'seethrough.psd')
SOURCE_PICTURE = os.path.join(HERE, 'source.png')      # the picture See-through worked from
OUT_DIR = config.out_dir('lady_prep')
OUT_PSD = os.path.join(OUT_DIR, 'lady.psd')


def zoom(picture, box, scale):
    crop = Image.fromarray(picture).crop(box)
    return crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)


def side_by_side(path, panels):
    sheet = Image.new('RGB', (sum(panel.width for panel in panels), max(panel.height for panel in panels)), 'white')
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    sheet.save(path, quality=92)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stack = Stack.load(SOURCE_PSD)
    source = np.asarray(Image.open(SOURCE_PICTURE).convert('RGB').resize(stack.size, Image.LANCZOS))
    before = stack.flatten()
    print('layers from See-through, back to front:', stack.names)

    artprep.clean(stack)

    # Loose pieces that See-through filed with the wrong part.  The braid hangs in front of the
    # body: its lower end came out as a separate island of 'front hair', and the ribbon bow tied
    # to it as a separate island of 'headwear' (the hat is the island that reaches the top).
    front_islands = pieces(stack['front hair'][..., 3] > 128, smallest=400)
    hat_islands = pieces(stack['headwear'][..., 3] > 128, smallest=400)
    assert len(front_islands) == 2 and len(hat_islands) == 2, (len(front_islands), len(hat_islands))
    braid_end = artprep.cut_out(stack, 'front hair', front_islands[1][1])
    bow = artprep.cut_out(stack, 'headwear', hat_islands[1][1])
    print(f'  braid end: {front_islands[1][0]} px taken out of front hair; bow: {hat_islands[1][0]} px taken out of headwear')

    # The braid itself was filed under 'back hair', behind the dress: it is the largest piece
    # that recover_hidden() brings forward.  Smaller pieces of back hair found with it stay as they are.
    made = artprep.recover_hidden(stack, source)
    with_braid = max((item for item in made if item[1] == 'back hair'), key=lambda item: item[2])[0]
    braid_mask = pieces(stack[with_braid][..., 3] > 128)[0][1]
    braid = artprep.cut_out(stack, with_braid, braid_mask)
    braid = np.array(Image.alpha_composite(Image.fromarray(braid), Image.fromarray(braid_end)))
    # What was found with it are flat patches where the blue ribbon below the bow happens to
    # match the back hair's colour, not hair: drop them.
    leftover = int((stack[with_braid][..., 3] > 128).sum())
    stack.add('braid', braid, after=with_braid)
    stack.remove(with_braid)
    print(f'  dropped {leftover} px found with the braid (ribbon that matched the hair colour)')
    # The bow sits on the belt and is also drawn in 'bottomwear', so it stays with the body:
    # the braid hangs between the head and the bow.
    stack.add('waist bow', bow, after='braid')

    artprep.repaint_hidden_skin(stack, covers=('front hair', 'headwear'))
    artprep.build_mouth(stack)
    for side in 'rl':
        artprep.build_eyelids(stack, side)

    # The picture shows the eyebrows through the fringe.
    stack.move('eyebrow-r', after='front hair')
    stack.move('eyebrow-l', after='eyebrow-r')

    stack.save(OUT_PSD)
    print('wrote', OUT_PSD)
    print('layers, back to front:')
    for name in stack.names:
        print(f'  {name:22s} {box_of(stack[name], threshold=0)}')

    after = stack.flatten()
    closed = stack.flatten(skip=('mouth inner',))          # the rig rests with the mouth closed
    for label, picture in (('See-through stack', before), ('cleaned, mouth open as in the PSD', after),
                           ('cleaned, mouth closed', closed)):
        difference = np.abs(picture.astype(int) - source.astype(int)).max(axis=2)
        print(f'  {label} against the source picture: mean difference {difference.mean():.2f}, '
              f'{int((difference > 40).sum())} pixels differ by more than 40')
    whole = (0, 0) + stack.size
    side_by_side(os.path.join(OUT_DIR, 'prep_whole.jpg'),
                 [zoom(picture, whole, 1).resize((640, 640), Image.LANCZOS) for picture in (source, before, closed)])
    body = (380, 420, 1000, 1200)
    side_by_side(os.path.join(OUT_DIR, 'prep_body.jpg'), [zoom(picture, body, 1) for picture in (source, before, closed)])
    face = (470, 150, 810, 560)
    no_hair = ('front hair', 'headwear')
    side_by_side(os.path.join(OUT_DIR, 'prep_face.jpg'),
                 [zoom(picture, face, 2) for picture in (source, closed, stack.flatten(skip=no_hair + ('mouth inner',)),
                                                         stack.flatten(skip=no_hair))])
    eyes = (500, 300, 780, 430)
    bare = no_hair + ('mouth inner', 'eyewhite-l', 'eyewhite-r', 'irides-l', 'irides-r')
    tint = lambda name, image: tinted(image) if name.startswith(('eyelid', 'eyelash')) else image
    side_by_side(os.path.join(OUT_DIR, 'prep_eyes.jpg'),
                 [zoom(stack.flatten(skip=no_hair + ('mouth inner',)), eyes, 4),
                  zoom(stack.flatten(skip=bare, change=tint), eyes, 4)])
    mouth = (590, 430, 700, 510)
    side_by_side(os.path.join(OUT_DIR, 'prep_mouth.jpg'),
                 [zoom(picture, mouth, 6) for picture in (source, closed, after)])


def tinted(image):
    """The layer with its skin-coloured parts tinted green, to see where added skin lies."""
    out = image.copy()
    out[..., 1] = np.minimum(255, out[..., 1].astype(int) + 0)
    out[..., 0] = (out[..., 0] * 0.6).astype(np.uint8)
    out[..., 2] = (out[..., 2] * 0.6).astype(np.uint8)
    return out


if __name__ == '__main__':
    main()
