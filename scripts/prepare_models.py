"""Fetch pinned model files or verify/copy an explicitly supplied offline cache."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hk_taxi.runtime_assets import digest,resolve_inside


def prepare(root=ROOT, cache=None, verify_only=False):
    root=Path(root)
    models=json.loads((root/'configs/models.json').read_text(encoding='utf-8'))
    report=[]
    for role, spec in models.items():
        target=resolve_inside(root,spec['local_path'])
        missing=[name for name,expected in spec['files'].items() if not (target/name).is_file() or digest(target/name)!=expected]
        if missing and verify_only:
            raise FileNotFoundError('Missing or different '+role+' model files; run scripts/prepare_models.py')
        if missing and cache:
            source=Path(cache)/spec['local_path']
            for relative in missing:
                origin=source/relative
                if not origin.is_file() or digest(origin)!=spec['files'][relative]:
                    raise ValueError('Offline cache does not contain the pinned '+role+' file: '+relative)
                dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(origin,dest)
        elif missing:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=spec['model_id'],revision=spec['revision'],local_dir=target,
                              allow_patterns=list(spec['files']),max_workers=2)
        for relative,expected in spec['files'].items():
            if digest(target/relative)!=expected:
                raise ValueError('Pinned model checksum differs: '+role+'/'+relative)
        report.append({'role':role,'model_id':spec['model_id'],'revision':spec['revision'],
                       'required_bytes':spec['required_bytes'],'verified':True,
                       'source':'existing' if not missing else 'explicit_offline_cache' if cache else 'huggingface'})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-cache',type=Path,help='Directory containing models/<model>/<revision>; file digests are checked before copying')
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    print(json.dumps(prepare(cache=a.from_cache,verify_only=a.verify_only),ensure_ascii=False,indent=2))
