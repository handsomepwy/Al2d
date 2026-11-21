# Workflow

From artwork to a model VTube Studio can load. Every step is a Python script; nothing is clicked.

## 0. Artwork

Two kinds of input:

- **A layered PSD drawn for Live2D** (the normal case for a commission). Each part is on its own
  layer, and hidden areas are painted. See [psd-guide.md](psd-guide.md).
- **One flat picture.** [See-through](https://github.com/shitagaki-lab/see-through) splits it into
  layers (about 10 minutes, needs most of a GPU; run nothing else on the card meanwhile):

  ```bash
  python inference/scripts/inference_psd.py --srcp <input dir> --save_dir <out dir> --save_to_psd --tblr_split --group_offload
  ```

  Pictures made for this work best: face to the front, head upright, face at least 300 px wide,
  brows not covered, mouth clearly drawn, arms away from the body.

## 1. Art prep: `characters/<name>/prep.py`

Turns the input PSD into the PSD the rig is built from, using `al2d/artprep.py`:

| Function | What it fixes |
|---|---|
| `Stack` | Load and save a PSD, add, move and remove layers, flatten |
| `clean` | Faint haze and stray specks (common in See-through output) |
| `recover_hidden` | Pieces that the source picture shows but that ended up behind other layers |
| `repaint_hidden_skin` | Skin hidden under hair that came out the wrong colour |
| `build_mouth` | One mouth patch → interior, skin cover and lip line, so it can open |
| `build_eyelids` | Lid skin above the lashes and a lower-lid strip, so the eye can close |

An artist's PSD usually needs only renaming and cropping (see `characters/haru/prep.py`, which maps
Japanese layer names to ours). Layer names follow See-through's: `face`, `front hair`,
`back hair`, `eyewhite-l`, `irides-l`, `eyelash-l`, `eyebrow-l`, `mouth`, `nose`, `neck`, `ears-l`
and so on. `-l` / `-r` are the character's left and right, so `-l` is on the right of the screen.

Always look at every layer, zoomed in, before rigging.

## 2. Rig: `characters/<name>/build.py`

The character script names the PSD, groups the layers and calls recipes from `al2d/rigkit.py`:

| Recipe | Parameters |
|---|---|
| `rig_blink` / `rig_blink_masked` (artist PSD with an eye mask) | `ParamEyeLOpen`, `ParamEyeROpen`, eye smile |
| `rig_mouth` / `rig_mouth_patches` (artist PSD) | `ParamMouthOpenY` |
| `rig_mouth_form` | `ParamMouthForm` |
| `rig_pupils` | `ParamEyeBallX`, `ParamEyeBallY` |
| `rig_brows` | brow Y, X, angle, form |
| `Head` + `rig_turn` | `ParamAngleX`, `ParamAngleY` (one warp per depth: face, nose, hair, back hair, neck, hat) |
| `rig_tilt` | `ParamAngleZ` (rotation deformer) |
| `rig_sway` | `ParamHairFront`, `ParamHairSide`, `ParamHairBack` |

Recipes measure the artwork (eye opening per column, lip line, face width) and scale their lengths
to the face, so most characters need only grouping and a few amounts. `Tuning` holds the numbers a
character may override.

`rigkit.build()` writes the `.cmo3` through `al2d/cmo3writer.py`. It makes a grid mesh for each
layer (`automesh.py`) and a texture atlas, and packs everything into the CAFF container (`caff.py`).

## 3. Convert: `rigkit.convert()`

- Umamo (patched, see `patches/`) turns the `.cmo3` into `.moc3` format version 5.
- `physics.py` writes `physics3.json`: one pendulum per hair group, driven by head angles.
- `pipeline.fill_model3` adds the `EyeBlink` and `LipSync` groups to `model3.json`, which
  VTube Studio's auto-setup uses.
- `--install` copies the runtime folder into VTube Studio's `Live2DModels`.

## 4. Check: the `report_*` functions

The built model is loaded through Cubism Core (`l2dcore.py`) and rendered with numpy, so no app is
needed to look at it:

- `report_rest`: the rest pose against the flattened PSD (mean pixel difference).
- `report_positions`: every deformer position from Core against our own maths.
- `report_blink`, `report_mouth`: whether anything that should be hidden shows (closed eyes, closed mouth).
- `report_turn`, `report_tilt`, `report_sway`, `report_physics`: picture sheets and GIFs at
  everyday (±15) and extreme angles.

Look at the pictures before handing a model over. Then try it in VTube Studio with face tracking.

## Adding a character

1. Copy the closest example: `characters/girl` for See-through output with straight hair,
   `characters/lady` for a hat, braid and a prep step, `characters/haru` for an artist's PSD with
   expressions and head tilt.
2. Point it at the new PSD, rename its layers in `prep.py`, regroup the layers in `rig()`.
3. Build, read the check pictures, adjust, rebuild. A build takes about a minute.

Client artwork goes in `clients/` (ignored by git), never in `examples/`.
