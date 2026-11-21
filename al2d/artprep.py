"""Clean-up of a See-through layer stack before rigging.

See-through gives one layer per body part.  Before the rig recipes can use them a few things
have to be repaired or added, and every step here works from what it measures in the layers, not
from coordinates typed in for one character:

    clean()                 faint haze and stray specks removed
    recover_hidden()        parts filed under the wrong depth (a braid behind the dress) brought forward
    repaint_hidden_skin()   skin that was hidden under hair gets the colour of the visible skin
    build_mouth()           mouth patch -> lip line + drawn interior + skin cover (for open/close)
    build_eyelids()         lid skin added above the lashes, lower-lid strips added (for the blink)

A Stack holds the layers back to front as full-canvas RGBA arrays.
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d


class Stack:
    def __init__(self, size, layers):
        self.size = size                                  # (width, height)
        self.names = [name for name, _ in layers]         # back to front
        self.images = dict(layers)

    @classmethod
    def load(cls, psd_path):
        psd = PSDImage.open(psd_path)
        width, height = psd.size
        layers = []
        for layer in psd:
            if layer.is_group() or not layer.visible:
                continue
            canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            canvas.paste(layer.topil().convert('RGBA'), (layer.left, layer.top))
            layers.append((layer.name, np.array(canvas)))
        return cls((width, height), layers)

    def __getitem__(self, name):
        return self.images[name]

    def __setitem__(self, name, image):
        assert name in self.images, name
        self.images[name] = image

    def __contains__(self, name):
        return name in self.images

    def add(self, name, image, after=None, before=None):
        assert name not in self.images, name
        index = self.names.index(after) + 1 if after is not None else self.names.index(before)
        self.names.insert(index, name)
        self.images[name] = image

    def remove(self, name):
        self.names.remove(name)
        return self.images.pop(name)

    def move(self, name, after):
        self.names.remove(name)
        self.names.insert(self.names.index(after) + 1, name)

    def flatten(self, skip=(), change=None, background=(255, 255, 255)):
        """The layers stacked up as an RGB picture.  change(name, image) may return another image."""
        canvas = Image.new('RGBA', self.size, background + (255,))
        for name in self.names:
            if name in skip:
                continue
            image = self.images[name]
            if change is not None:
                image = change(name, image)
            canvas.alpha_composite(Image.fromarray(image))
        return np.asarray(canvas.convert('RGB'))

    def save(self, psd_path):
        psd = PSDImage.new(mode='RGBA', size=self.size, depth=8)
        for name in self.names:
            left, top, right, bottom = box_of(self.images[name], threshold=0)
            crop = Image.fromarray(self.images[name]).crop((left, top, right + 1, bottom + 1))
            psd.create_pixel_layer(crop, name=name, top=int(top), left=int(left), opacity=255)
        psd.save(psd_path)


def box_of(image, threshold=40):
    """(left, top, right, bottom) of the pixels more opaque than the threshold, inclusive."""
    rows, columns = np.nonzero(image[..., 3] > threshold)
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def pieces(mask, smallest=1):
    """Connected pieces of a mask, largest first, as (pixel count, mask)."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask, labels, range(1, count + 1)) if count else []
    order = np.argsort(sizes)[::-1]
    return [(int(sizes[index]), labels == index + 1) for index in order if sizes[index] >= smallest]


def cut_out(stack, name, mask, reach=3):
    """Takes the pixels of one piece (and its soft edge) out of a layer; returns them as a layer."""
    near = ndimage.binary_dilation(mask, iterations=reach)
    image = stack[name].copy()
    piece = np.zeros_like(image)
    piece[near] = image[near]
    image[near] = 0
    stack[name] = image
    return piece


def clean(stack, speck=12, reach=3, report=print):
    """Removes what is not part of the drawing: See-through leaves a faint veil (alpha under 32)
    over large areas of some layers, and isolated specks.  Soft edges of real strokes are kept."""
    for name in stack.names:
        image = stack[name].copy()
        solid = image[..., 3] > 128
        labels, count = ndimage.label(solid, structure=np.ones((3, 3)))
        sizes = ndimage.sum(solid, labels, range(1, count + 1))
        real = np.isin(labels, [index + 1 for index, size in enumerate(sizes) if size >= speck])
        keep = ndimage.binary_dilation(real, iterations=reach)
        removed = int(((image[..., 3] > 0) & ~keep).sum())
        image[~keep] = 0
        stack[name] = image
        if removed > 200:
            report(f'  clean: {name}: {removed} faint or stray pixels removed')


def recover_hidden(stack, source, smallest=400, report=print):
    """Brings forward what See-through filed too far back.

    A spot counts when a layer is hidden there by the layers above it, its own pixel is what the
    source picture shows, and the stacked layers show something else.  Each such piece is copied
    into a new layer just above whatever hides it; the layer it came from stays whole, so the two
    can never come apart.  Returns [(new layer name, layer it came from, pixel count, box)].
    """
    source = source.astype(int)
    wrong = np.abs(stack.flatten().astype(int) - source).max(axis=2)
    cover = np.zeros(source.shape[:2])
    hidden_by_above = {}
    for name in reversed(stack.names):
        hidden_by_above[name] = cover.copy()
        alpha = stack[name][..., 3] / 255.0
        cover = cover + (1 - cover) * alpha
    made = []
    original = list(stack.names)
    for name in original:
        image = stack[name]
        own = np.abs(image[..., :3].astype(int) - source).max(axis=2)
        found = (image[..., 3] > 200) & (hidden_by_above[name] > 0.8) & (own < 30) & (wrong > own + 25)
        found = ndimage.binary_closing(found, iterations=3) & (image[..., 3] > 200) & (hidden_by_above[name] > 0.5)
        found = ndimage.binary_fill_holes(found)
        found = ndimage.binary_opening(found, iterations=2)
        found = ndimage.median_filter(found.astype(np.uint8), size=7) > 0            # no ragged edge
        # Also cover the soft edge of whatever hid the piece, or that edge stays behind as a thin line.
        found |= ndimage.binary_dilation(found, iterations=4) & (image[..., 3] > 200) & (hidden_by_above[name] < 0.8)
        above = original[original.index(name) + 1:]
        by_place = {}
        for size, piece in pieces(found, smallest):
            covering = [other for other in above if other in stack.images
                        and (stack[other][..., 3][piece] > 128).mean() > 0.05]
            if covering:
                by_place.setdefault(covering[-1], []).append((size, piece))
        for number, (place, found_pieces) in enumerate(by_place.items(), 1):
            mask = np.any([piece for _, piece in found_pieces], axis=0)
            soft = np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8) * 2 - 0.5, 0, 1)
            layer = image.copy()
            layer[..., 3] = (layer[..., 3] * soft).astype(np.uint8)
            new_name = f'{name} over' if len(by_place) == 1 else f'{name} over {number}'
            stack.add(new_name, layer, after=place)
            size = sum(size for size, _ in found_pieces)
            made.append((new_name, name, size, box_of(layer)))
            report(f'  recover: {size} px of {name!r} were hidden behind {place!r}: new layer {new_name!r}, box {box_of(layer)}')
    return made


def repaint_hidden_skin(stack, face='face', covers=('front hair',), report=print):
    """Skin that the picture never showed (it was under hair) comes out of See-through in a
    different colour.  It is invisible until the hair slides over the face on a head turn.
    Repaints it with the colour of the visible skin; the drawn outline of the face is kept."""
    image = stack[face].copy()
    alpha = image[..., 3]
    colour = image[..., :3].astype(np.float64)
    solid = alpha > 250
    covered = np.zeros(alpha.shape, bool)
    for name in covers:
        covered |= stack[name][..., 3] > 128
    outline = solid & (colour.mean(axis=2) < 170)
    visible = solid & ~ndimage.binary_dilation(covered, iterations=6) & ~ndimage.binary_dilation(outline, iterations=3)
    skin = np.median(colour[visible], axis=0)
    was = np.median(colour[solid & ndimage.binary_erosion(covered, iterations=6) & ~outline], axis=0)
    # Every pixel under or next to the hair: take out as much of the colour difference as the
    # pixel has of the hidden colour.  Outline pixels keep their darkness, edge pixels blend.
    repaint = ndimage.binary_dilation(covered, iterations=6) & (alpha > 0)
    tint = was - skin
    share = np.clip(((colour - skin) @ tint) / (tint @ tint), 0, 1) * repaint
    colour = np.clip(colour - share[..., None] * tint, 0, 255)
    # Well inside, anything that is not outline becomes plain skin (the hidden skin is blotchy).
    plain = ndimage.binary_erosion(covered, iterations=2) & (colour.mean(axis=2) > skin.mean() - 30)
    colour[plain] = skin
    image[..., :3] = colour.astype(np.uint8)
    seam = ndimage.binary_dilation(plain, iterations=4) & ~ndimage.binary_erosion(plain, iterations=4) \
        & solid & (colour.mean(axis=2) > skin.mean() - 30)
    blurred = cv2.GaussianBlur(image[..., :3], (0, 0), 2.0)
    image[..., :3][seam] = blurred[seam]
    stack[face] = image
    report(f'  skin: visible skin colour {skin.astype(int).tolist()}, hidden skin was {was.astype(int).tolist()}; '
           f'{int(repaint.sum())} px repainted')
    return skin


def build_mouth(stack, mouth='mouth', face='face', open_depth=0.3, report=print):
    """Replaces the mouth patch by three layers (back to front):

        mouth inner   the open mouth: a drawn dark crescent hanging from the lip line, with tongue
        mouth upper   face skin just above the lip line, which hides the crescent when it is flat
        mouth line    the drawn lip line itself

    The skin of the patch is dropped (the face layer has skin there already).  The PSD shows the
    mouth open; the rig flattens `mouth inner` onto the lip line to close it.
    open_depth: how far the open mouth hangs down, as a share of the lip line's length.
    """
    width, height = stack.size
    yy, xx = np.mgrid[0:height, 0:width]
    patch = stack[mouth]
    colour, alpha = patch[..., :3].astype(np.float64), patch[..., 3]
    skin = np.median(colour[alpha > 250], axis=0)
    away = np.linalg.norm(colour - skin, axis=2) * (alpha > 200)
    dark = pieces(away > 60, smallest=4)
    assert dark, 'no lip line found in the mouth layer'
    seed = np.any([piece for _, piece in dark], axis=0)
    rows, columns = np.nonzero(seed)
    x0, x1 = int(columns.min()), int(columns.max())
    # the lip line: in every column the pixel that differs most from skin, near the dark ends
    band = (yy >= rows.min() - 4) & (yy <= rows.max() + 4) & (xx >= x0) & (xx <= x1)
    strongest = np.where(band, away, 0)
    line_x = np.arange(x0, x1 + 1)
    line_y = strongest[:, x0:x1 + 1].argmax(axis=0).astype(np.float64)
    weight = strongest[:, x0:x1 + 1].max(axis=0)
    fit = np.polyfit(line_x, line_y, 2, w=np.maximum(weight, 1.0))
    curve = lambda x: np.polyval(fit, x)
    line_colour = np.percentile(colour[seed], 10, axis=0)
    report(f'  mouth: lip line x {x0}..{x1}, y {curve(x0):.1f} at the ends, {curve((x0 + x1) / 2):.1f} in the middle; '
           f'skin {skin.astype(int).tolist()}, line {line_colour.astype(int).tolist()}')

    # lip line layer: the patch's own pixels, as opaque as they differ from skin, close to the curve
    # (less far below the curve than above: a lip shade under the line would lie over the open mouth)
    below = yy - curve(xx.astype(np.float64))
    near_curve = np.clip(np.where(below > 0, 2.5 - below, 1.5 + below / 4.0), 0, 1) \
        * np.clip((xx - (x0 - 3)) / 2.0, 0, 1) * np.clip(((x1 + 3) - xx) / 2.0, 0, 1)
    line = patch.copy()
    line[..., 3] = (np.clip((away - 8) / 22.0, 0, 1) * near_curve * 255).astype(np.uint8)

    # open mouth, drawn at 6x and shrunk
    scale, depth = 6, open_depth * (x1 - x0)
    centre, half = (x0 + x1) / 2, (x1 - x0) / 2 - 1.5
    left, top = int(centre - half - 12), int(min(curve(x0), curve(centre)) - 12)
    box_w, box_h = int(2 * half + 24), int(depth + 30)
    canvas = Image.new('RGBA', (box_w * scale, box_h * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    steps = np.linspace(-1, 1, 121)
    upper_edge = [((centre + u * half - left) * scale, (curve(centre + u * half) - top) * scale) for u in steps]
    lower_edge = [((centre + u * half - left) * scale,
                   (curve(centre + u * half) + depth * np.sqrt(max(0.0, 1 - u * u)) ** 0.9 - top) * scale) for u in steps]
    outline = upper_edge + lower_edge[::-1]
    draw.polygon(outline, fill=(120, 48, 56, 255))
    inside = Image.new('L', canvas.size, 0)
    ImageDraw.Draw(inside).polygon(outline, fill=255)
    tongue = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    tongue_y = (curve(centre) + depth * 0.95 - top) * scale
    ImageDraw.Draw(tongue).ellipse([(centre - half * 0.62 - left) * scale, tongue_y - depth * 0.62 * scale,
                                    (centre + half * 0.62 - left) * scale, tongue_y + depth * 0.7 * scale],
                                   fill=(222, 128, 128, 255))
    canvas.paste(tongue, (0, 0), Image.composite(tongue.split()[3], Image.new('L', canvas.size, 0), inside))
    stroke = max(1.6, 0.03 * (x1 - x0))
    draw.line(lower_edge, fill=tuple(int(v) for v in line_colour) + (255,), width=int(stroke * scale), joint='curve')
    inner = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    inner.paste(canvas.resize((box_w, box_h), Image.LANCZOS), (left, top))
    inner = np.array(inner)

    # cover: face skin in a band above the lip curve, faded at its top and sides
    rise = int(depth) + 3
    curve_y = curve(xx.astype(np.float64))
    band = (yy < curve_y + 0.5) & (yy > curve_y - rise - 8) & (xx >= x0 - 5) & (xx <= x1 + 5)
    fade = np.clip((yy - (curve_y - rise - 8)) / 5.0, 0, 1) * np.clip((xx - (x0 - 5)) / 3.0, 0, 1) \
        * np.clip(((x1 + 5) - xx) / 3.0, 0, 1)
    cover = np.zeros_like(patch)
    cover[..., :3] = stack[face][..., :3]
    cover[..., 3] = (np.where(band, 255, 0) * fade * (stack[face][..., 3] > 250)).astype(np.uint8)

    stack.add('mouth inner', inner, after=mouth)
    stack.add('mouth upper', cover, after='mouth inner')
    stack.add('mouth line', line, after='mouth upper')
    stack.remove(mouth)
    return {'lip': (x0, x1), 'depth': depth}


def build_eyelids(stack, side, face='face', lower_share=0.3, report=print):
    """Prepares one eye for the blink (side 'l' or 'r', as in the layer names).

    eyelash-<side>  gets a block of lid skin above the lashes; when the layer comes down the block
                    covers the eye, so nothing has to be squashed.
    eyelid-<side>   new: a strip of skin under the eye that comes up to meet the lashes.
    Both are copies of the face's skin at the same place, so nothing changes in the still picture.
    """
    width, height = stack.size
    yy, xx = np.mgrid[0:height, 0:width]
    lash_name = f'eyelash-{side}'
    lash = stack[lash_name].copy()
    # A pale veil around the strokes smears when the lid closes: keep soft pixels only right next
    # to a clear stroke (their anti-aliased edge), drop the rest.
    haze = (lash[..., 3] > 0) & (lash[..., 3] < 90) & ~ndimage.binary_dilation(lash[..., 3] >= 90, iterations=2)
    lash[..., 3][haze] = 0
    drawn = lash[..., 3] > 120
    eye = (stack[f'eyewhite-{side}'][..., 3] > 40) | (stack[f'irides-{side}'][..., 3] > 40)
    eye_columns = np.nonzero(eye.any(axis=0))[0]
    ex0, ex1 = int(eye_columns.min()), int(eye_columns.max())
    lash_columns = np.nonzero(drawn.any(axis=0))[0]
    lx0, lx1 = int(lash_columns.min()), int(lash_columns.max())
    opening, lash_edge, eye_bottom = np.zeros(width), np.zeros(width), np.zeros(width)
    for x in range(ex0, ex1 + 1):
        eye_rows = np.nonzero(eye[:, x])[0]
        bottom = eye_rows.max()
        lash_rows = np.nonzero(drawn[:, x] & (yy[:, x] < bottom))[0]
        # the main lash stroke: lash pixels above the middle of the eye
        upper = lash_rows[lash_rows <= (eye_rows.min() + bottom) / 2 + 4]
        if len(upper) == 0:
            continue
        lash_edge[x] = upper.max()
        opening[x] = max(0, bottom - upper.max() + 1)
        eye_bottom[x] = bottom
    smooth_opening = gaussian_filter1d(opening, 7)
    smooth_opening[:lx0] = 0
    smooth_opening[lx1 + 1:] = 0
    known = np.nonzero(eye_bottom > 0)[0]
    bottom_line = gaussian_filter1d(np.interp(np.arange(width), known, eye_bottom[known]), 2)
    edge_line = gaussian_filter1d(np.interp(np.arange(width), known, lash_edge[known]), 3)
    skin = stack[face]

    # lower lid: skin from the eye's bottom edge downwards, faded out below and at the sides
    strip_height = max(12, int(0.5 * opening.max()))
    strip = np.zeros_like(lash)
    strip[..., :3] = skin[..., :3]
    # (it starts just under the eye's own bottom edge, which is often a drawn line; smooth top edge)
    in_strip = (xx >= ex0 - 4) & (xx <= ex1 + 4) & (yy <= bottom_line[None, :] + strip_height)
    fade = np.clip((bottom_line[None, :] + strip_height - yy) / 6.0, 0, 1) * np.clip((xx - (ex0 - 4)) / 3.0, 0, 1) \
        * np.clip(((ex1 + 4) - xx) / 3.0, 0, 1) * np.clip(yy - (bottom_line[None, :] + 0.5), 0, 1)
    strip[..., 3] = (np.where(in_strip, 255, 0) * fade).astype(np.uint8)

    # upper lid: a block of skin above the lashes' lower edge, as tall as the lashes have to travel
    block_height = smooth_opening.max() * (1 - lower_share) + 18
    lid = np.zeros_like(lash)
    lid[..., :3] = skin[..., :3]
    # (its lower edge stays 2 px inside the lash stroke, so it never covers the eye below the lashes)
    block_bottom = edge_line[None, :] - 2
    in_block = (xx >= ex0 - 6) & (xx <= ex1 + 6) & (yy <= block_bottom) & (yy >= block_bottom - block_height)
    block_alpha = np.where(in_block, 255, 0).astype(np.uint8)
    soft = cv2.GaussianBlur(block_alpha, (0, 0), 1.0)
    lid[..., 3] = np.where(yy <= block_bottom - 1, soft, block_alpha)
    merged = np.array(Image.alpha_composite(Image.fromarray(lid), Image.fromarray(lash)))

    stack[lash_name] = merged
    stack.add(f'eyelid-{side}', strip, before=lash_name)
    report(f'  eye {side}: x {ex0}..{ex1}, opens {opening.max():.0f} px; lid block {block_height:.0f} px tall, '
           f'lower strip {strip_height} px; {int(haze.sum())} veil pixels dropped from the lashes')
    return {'x': (ex0, ex1), 'opening': float(opening.max())}
