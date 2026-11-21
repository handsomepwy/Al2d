"""Steps shared by the model build scripts: convert a .cmo3 with Umamo, then look at the result.

Umamo (GPLv3, https://github.com/future2d/umamo, patched with patches/umamo.patch) is only run as a program here; none of its code is copied.
"""
import json
import os
import shutil
import subprocess

import numpy as np

from . import config, l2dcore



def convert_with_umamo(cmo3_path, out_dir, name, moc_version='V50'):
    """Runs Umamo's cmo3 -> moc3 conversion.  Returns the path of the written model3.json.

    moc_version V50 gives a file VTube Studio can read (its Core stops at format version 5).
    """
    os.makedirs(out_dir, exist_ok=True)
    source = os.path.abspath(cmo3_path).replace('\\', '/')
    target = os.path.join(os.path.abspath(out_dir), f'{name}.moc3').replace('\\', '/')
    umamo_dir = config.need('umamo_dir')
    environment = dict(os.environ, UMAMO_MOC_VERSION=moc_version)
    if config.get('java_home'):
        environment['JAVA_HOME'] = config.get('java_home')
    gradle = os.path.join(umamo_dir, 'gradlew.bat') if os.name == 'nt' else './gradlew'
    command = [gradle, '-q', ':cli:run', f'--args=convert "{source}" "{target}"']
    result = subprocess.run(command, cwd=umamo_dir, env=environment, capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or not os.path.exists(target):
        raise RuntimeError(f'Umamo convert failed (exit {result.returncode}):\n{output[-4000:]}')
    return os.path.join(out_dir, f'{name}.model3.json'), output


def fill_model3(model3_path, eye_blink=('ParamEyeLOpen', 'ParamEyeROpen'), lip_sync=('ParamMouthOpenY',),
                physics_from=None):
    """Adds what Umamo leaves out of model3.json: the EyeBlink and LipSync groups, physics file.

    physics_from  a physics3.json to copy next to the model and list in model3.json.
    """
    manifest = json.load(open(model3_path, encoding='utf-8'))
    manifest['Groups'] = [
        {'Target': 'Parameter', 'Name': 'EyeBlink', 'Ids': list(eye_blink)},
        {'Target': 'Parameter', 'Name': 'LipSync', 'Ids': list(lip_sync)},
    ]
    if physics_from:
        physics_name = os.path.basename(model3_path).replace('.model3.json', '.physics3.json')
        shutil.copyfile(physics_from, os.path.join(os.path.dirname(model3_path), physics_name))
        manifest['FileReferences']['Physics'] = physics_name
    json.dump(manifest, open(model3_path, 'w', encoding='utf-8'), indent='\t')


def install_in_vtube_studio(runtime_dir, name):
    """Copies a converted model into VTube Studio's model folder.

    Files VTube Studio keeps there itself (its .vtube.json settings, icon.jpg) are left alone.
    Returns the folder.
    """
    target = os.path.join(config.need('vts_models'), name)
    os.makedirs(target, exist_ok=True)
    for entry in os.listdir(runtime_dir):
        source = os.path.join(runtime_dir, entry)
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(target, entry), dirs_exist_ok=True)
        else:
            shutil.copyfile(source, os.path.join(target, entry))
    return target


def load(model3_path):
    manifest = json.load(open(model3_path, encoding='utf-8'))
    moc_path = os.path.join(os.path.dirname(model3_path), manifest['FileReferences']['Moc'])
    return l2dcore.Model(moc_path), l2dcore.load_textures(model3_path)


def over_white(rgba):
    """An RGBA picture flattened onto white, as uint8 RGB."""
    colour = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return (colour * alpha + 255.0 * (1 - alpha) + 0.5).astype(np.uint8)


def difference_report(reference_rgb, rendered_rgb):
    """Per-pixel difference between two RGB pictures: (mean, count over 24, count over 64, map)."""
    difference = np.abs(reference_rgb.astype(np.int16) - rendered_rgb.astype(np.int16)).max(axis=2)
    return float(difference.mean()), int((difference > 24).sum()), int((difference > 64).sum()), difference
