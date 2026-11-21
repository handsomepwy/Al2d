# Third-party work

| Project | Licence | How Al2d uses it |
|---|---|---|
| [Umamo](https://github.com/future2d/umamo) | GPL-3.0 | Run as a separate program to convert `.cmo3` to `.moc3`. Our changes to it are in `patches/umamo.patch`, under GPL-3.0. |
| [StretchyStudio](https://github.com/MangoLion/stretchystudio) | MIT | Its `caffPacker.js` and `cmo3writer.js` were read as a reference for the CAFF container and for which objects a minimal `.cmo3` needs. No code copied. |
| [moc3ingbird](https://github.com/OpenL2D/moc3ingbird) | Free Development Public License | Its `cmo3.hexpat` was read as a reference for the CAFF layout. No code copied. |
| [See-through](https://github.com/shitagaki-lab/see-through) | Apache-2.0 (code) | Optional: splits a single picture into layers. Run separately; not included. Check its model weights' terms before commercial use. |
| Live2D Cubism Core | Live2D Proprietary Software License | Loaded at run time through its public C API to check and render models. Not included; obtain it from Live2D. |

Live2D's sample models (for example Haru) are not included. `characters/haru/` only holds our
scripts; running them needs the sample files, which are covered by Live2D's Free Material License.

Live2D and Cubism are trademarks or registered trademarks of Live2D Inc. Al2d is not affiliated
with or endorsed by Live2D Inc.
