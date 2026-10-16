"""Offline preflight; no API calls and no credential values in output."""
from pathlib import Path
import argparse
import importlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def check(root=ROOT, models=False, credentials=False):
    root=Path(root);checks=[]
    for name in ['numpy','sentence_transformers','rank_bm25','opencc','duckdb','pyproj','openai','dotenv','jsonschema','torch','transformers','streamlit']:
        try:importlib.import_module(name);checks.append({'check':'import:'+name,'ok':True})
        except (ImportError,OSError):checks.append({'check':'import:'+name,'ok':False})
    import torch
    checks.append({'check':'CUDA','ok':torch.cuda.is_available()})
    from hk_taxi.runtime_assets import verify_inputs,verify_asset
    try:verify_inputs(root);checks.append({'check':'approved_inputs','ok':True})
    except (FileNotFoundError,ValueError):checks.append({'check':'approved_inputs','ok':False,'fix':'Restore bundled data from the repository, then run python scripts/setup.py'})
    manifest=root/'data/processed/runtime_manifest.json'
    if manifest.exists():
        try:
            data=json.loads(manifest.read_text(encoding='utf-8'))
            for relative in data['files']:verify_asset(root,relative)
            checks.append({'check':'generated_data','ok':True,'counts':data['counts']})
        except (FileNotFoundError,ValueError):checks.append({'check':'generated_data','ok':False,'fix':'Run python scripts/prepare_data.py --force'})
    else:checks.append({'check':'generated_data','ok':False,'fix':'Run python scripts/prepare_data.py'})
    if models:
        from prepare_models import prepare
        try:prepare(root,verify_only=True);checks.append({'check':'model_files_and_sha256','ok':True})
        except (FileNotFoundError,ValueError):checks.append({'check':'model_files_and_sha256','ok':False,'fix':'Run python scripts/prepare_models.py'})
    if credentials:
        from hk_taxi.generation import GenerationSettings,GenerationConfigError
        try:GenerationSettings.load(root);checks.append({'check':'Qwen_local_config','ok':True})
        except (GenerationConfigError,OSError):checks.append({'check':'Qwen_local_config','ok':False,'fix':'Fill .env following .env.example'})
        from dotenv import dotenv_values
        import os
        local=dotenv_values(root/'.env',interpolate=False) if (root/'.env').exists() else {}
        checks.append({'check':'ORS_key_present','ok':bool(os.environ.get('ORS_API_KEY') or local.get('ORS_API_KEY'))})
    return {'ok':all(c['ok'] for c in checks),'checks':checks,'api_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',action='store_true',help='Hash all required model files; may take several seconds')
    p.add_argument('--credentials',action='store_true',help='Validate local configuration shape, without sending requests')
    a=p.parse_args();r=check(models=a.models,credentials=a.credentials)
    print(json.dumps(r,ensure_ascii=False,indent=2));raise SystemExit(0 if r['ok'] else 1)
