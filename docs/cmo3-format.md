# Notes on the `.cmo3` format

What we learned writing Cubism Editor 5.3 projects from Python. Everything here was read from
projects the Editor saved and confirmed through Cubism Core's output. Live2D does not publish this
format; treat it as observations, not a specification.

## Container

A `.cmo3` is a CAFF archive (`al2d/caff.py`): a header with an obfuscation key, then entries that
are XOR-obfuscated and optionally zip-compressed. The entries are `main.xml` (a serialized object
graph) and the PNG images. `python -m al2d.caff list|unpack|repack` handles them.

`main.xml` declares the classes it uses in an import list; `al2d/cmo3_imports.txt` is the list an
Editor 5.3 project declares. Shared objects carry `xs.id` and are referenced as `xs.ref`.

## Coordinate spaces

The most important rule, and the one other generators get wrong:

| Object | Parent | Positions are stored as |
|---|---|---|
| mesh or warp | warp | 0..1 across the parent warp's box |
| mesh or warp | rotation deformer | pixels from the rotation's pivot, turned back by its rest angle |
| warp | root | canvas pixels |

A mesh under a warp keeps canvas pixels in `GEditableMesh2.point` and in the source-level
`positions`. Only `keyforms/CArtMeshForm/positions` is in the warp's 0..1 space. A warp also
carries a `CDeformerOriginalShapeExtension` in canvas pixels, whatever its parent.

`tools/cmo3_tree.py <main.xml>` prints the deformer tree with each object's coordinate range.

## Meshes and textures

- UVs are atlas coordinates: `u = (canvas x - m02) / page width`, `v = (canvas y - m12) / page height`,
  from the top, where `m02` / `m12` come from the mesh's
  `CTextureInput_TextureAtlasRegion.inputImageLocalToCanvasTransform`.
- All meshes share one `GTexture2D` whose `srcImageResource` is the atlas page. Without an atlas,
  Umamo writes one canvas-sized texture per mesh, so we write the atlas ourselves.
- Draw order: every form has `drawOrder` 500. Front and back are decided by the order of the root
  part's `_childGuids`, first in front.

## Deformers

- `CWarpDeformerSource`: `col` / `row` count cells, so a form holds `(col+1)*(row+1)` points, row by
  row from the top left. With `isQuadTransform` true, a point inside a cell is placed bilinearly.
- Each warp also has two `CWarpDeformerBezierExtension` (edit level 2 with 2×2, level 3 with 1×1).
- Deformers are listed in `CDeformerSourceSet/_sources`, and their guids in the owning part's
  `_childGuids`, mixed with the drawables. `targetDeformerGuid` names an object's parent.
- Nesting: a parent warp moves only the child's own points. A mesh in a warp in a warp is placed
  inside the child grid after that grid has moved.
- Rotation forms are `[pivot x, pivot y, angle, scale]`. Positive angle is clockwise on screen.

## Keys

An object with keys has a shared `KeyformGridSource` plus one `KeyformBindingSource` per parameter
(`_gridSource`, `parameterGuid`, `keys`, interpolation `LINEAR`). The grid lists one
`KeyformOnGrid` per key combination; with two parameters the first binding's index runs fastest.
Nothing is written on the parameter's side. An object with no keys has a one-cell grid and no
bindings.

## Runtime files

- Cubism Editor 5.3 exports `.moc3` format version 6 by default. VTube Studio ships Cubism Core
  5.0, which reads up to version 5. Byte 4 of a `.moc3` is its version.
- Physics set up in the Editor is stored in the project (`CPhysicsSettingsSource` and friends).
  Umamo does not turn it into `physics3.json`, so `al2d/physics.py` writes the JSON directly.
- A model folder needs `.moc3`, `.model3.json`, `.physics3.json`, `.cdi3.json` and the textures.

## Tools

| Tool | Use |
|---|---|
| `tools/cmo3_tree.py <main.xml>` | Deformer tree with coordinate ranges |
| `tools/xmlshow.py <main.xml> ...` | Outline of any part of `main.xml` |
| `tools/schema_check.py <ref main.xml> <ours>` | Fields we write differently from the Editor |
| `tools/compare_models.py <ref model3.json> <ours> <out>` | Pose two models identically, compare vertices and renders |
| `tools/fix_ss_mesh_space.py` | The experiment that found the coordinate-space bug in StretchyStudio's exports |
