"""Portable, content-pinned inputs and machine-local generated asset checks."""
from pathlib import Path
import hashlib
import json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError('Asset path escapes the project root')
    return path


def verify_inputs(root):
    root = Path(root)
    manifest = json.loads((root/'data/manifests/runtime_inputs.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        if digest(resolve_inside(root, relative)) != expected:
            raise ValueError('Approved input differs: '+relative)
    return manifest


def verify_model(root, role):
    root = Path(root)
    spec = json.loads((root/'configs/models.json').read_text(encoding='utf-8'))[role]
    folder = resolve_inside(root, spec['local_path'])
    for relative, expected in spec['files'].items():
        if digest(resolve_inside(folder, relative)) != expected:
            raise ValueError('Pinned model file differs: '+role+'/'+relative)
    return spec


def verify_asset(root, relative):
    root = Path(root)
    path = resolve_inside(root, relative)
    manifest_path = root/'data/processed/runtime_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Run python scripts/prepare_data.py first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    inputs_sha = digest(root/'data/manifests/runtime_inputs.json')
    if inputs_sha != manifest['inputs_manifest_sha256']:
        raise ValueError('Input version changed: rebuild runtime data')
    expected = manifest['files'].get(Path(relative).as_posix())
    if expected is None or digest(path) != expected:
        raise ValueError('Generated runtime asset differs: '+relative)
    return path
