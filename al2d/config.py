"""Where the outside programs and private data are on this machine.

Values come from `al2d.toml` in the repository root (copy `al2d.example.toml`), or from the file
named by the environment variable AL2D_CONFIG.  An environment variable AL2D_<KEY> (for example
AL2D_CORE_DLL) overrides the file.

    umamo_dir   Umamo checkout with patches/umamo.patch applied (cmo3 -> moc3)
    java_home   JDK used to run Umamo (optional; the environment's JAVA_HOME otherwise)
    core_dll    Live2DCubismCore.dll (or .so / .dylib), used to check and render models
    vts_models  VTube Studio's Live2DModels folder, only for --install
    haru_dir    Live2D's sample "Haru" files, only for characters/haru
"""
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ('umamo_dir', 'java_home', 'core_dll', 'vts_models', 'haru_dir')


def _load():
    path = os.environ.get('AL2D_CONFIG') or os.path.join(ROOT, 'al2d.toml')
    values = {}
    if os.path.exists(path):
        with open(path, 'rb') as file:
            values = tomllib.load(file)
    for key in KEYS:
        if os.environ.get('AL2D_' + key.upper()):
            values[key] = os.environ['AL2D_' + key.upper()]
    return values


_values = _load()


def get(key, default=None):
    """The configured value, or default when it is not set."""
    return _values.get(key) or default


def need(key):
    """The configured value; stops with an explanation when it is not set."""
    value = _values.get(key)
    if not value:
        raise RuntimeError(f'"{key}" is not configured.  Copy al2d.example.toml to al2d.toml in '
                           f'{ROOT} and fill it in, or set the environment variable AL2D_{key.upper()}.')
    return value


def out_dir(name):
    """Default output folder for a character: <repo>/out/<name>."""
    return os.path.join(ROOT, 'out', name)
