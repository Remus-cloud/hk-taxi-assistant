"""Initialize fixed models, SQL databases and embeddings from bundled inputs."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hk_taxi.runtime_assets import verify_inputs
from prepare_models import prepare
from prepare_data import build
from doctor import check

def initialize(model_cache=None):
    inputs = verify_inputs(ROOT)
    result = {'inputs': {'status': 'bundled_verified',
                         'chunk_count': inputs['chunk_count'],
                         'taxi_stand_count': inputs['taxi_stand_count'],
                         'facility_count': inputs['facility_count']}}
    result['models'] = prepare(cache=model_cache)
    result['data'] = build()
    result['doctor'] = check()
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache', type=Path,
                        help='Optional root containing pinned models/<name>/<revision>')
    args = parser.parse_args()
    result = initialize(args.model_cache)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['doctor']['ok'] else 1)
