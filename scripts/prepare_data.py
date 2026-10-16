"""Verify reviewed inputs, build SQL tables and encode the approved current corpus.

No cloud model, geocoding, ORS or search requests. Model files must be prepared first.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hk_taxi.runtime_assets import digest, verify_inputs, verify_asset


def build(root=ROOT, force=False):
    import duckdb
    import numpy as np
    from hk_taxi.fare_calculator import initialize_fares
    from hk_taxi.harbour_tolls import initialize
    from hk_taxi.retrieval import DenseEncoder

    root = Path(root)
    inputs = verify_inputs(root)
    from prepare_models import prepare
    prepare(root, verify_only=True)
    existing = root/'data/processed/runtime_manifest.json'
    if existing.exists() and not force:
        state = json.loads(existing.read_text(encoding='utf-8'))
        for relative in state['files']:
            verify_asset(root, relative)
        return {'status':'already_verified', 'chunk_count':inputs['chunk_count']}
    stage = root/'data/processed'/('build_'+uuid4().hex)
    stage.mkdir(parents=True)
    files = {}
    def publish(source, relative):
        dest = root/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        files[relative] = digest(dest)
    def read(relative):
        return json.loads((root/relative).read_text(encoding='utf-8'))
    try:
        stands = read('data/curated/taxi_stands.json')
        if len(stands) != inputs['taxi_stand_count']:
            raise ValueError('Stand count differs')
        stands_file = stage/'taxi_stands.json'
        stands_file.write_text(json.dumps(stands, ensure_ascii=False), encoding='utf-8')
        db = stage/'hk_taxi.duckdb'
        with duckdb.connect(str(db)) as con:
            con.execute('CREATE TABLE taxi_stands AS SELECT * FROM read_json_auto(?)', [str(stands_file)])
        initialize_fares(db, root/'configs/fare_distance_v1.json')
        publish(db, 'data/processed/hk_taxi.duckdb')
        publish(stands_file, 'data/processed/taxi_stands.json')

        facilities = read('data/reference/passenger_facilities_v2.json')
        cfg = read('configs/passenger_tools_v2.json')
        db = stage/'facilities.duckdb'
        with duckdb.connect(str(db)) as con:
            con.execute('CREATE TABLE pickup_facilities(facility_id VARCHAR PRIMARY KEY,name VARCHAR,kind VARCHAR,service_modes VARCHAR,taxi_type_scope VARCHAR,area VARCHAR,longitude DOUBLE,latitude DOUBLE,source_url VARCHAR,evidence_id VARCHAR,place_keys VARCHAR)')
            con.executemany('INSERT INTO pickup_facilities VALUES(?,?,?,?,?,?,?,?,?,?,?)', [[r['facility_id'],r['name'],r['kind'],json.dumps(r['service_modes']),json.dumps(r['taxi_type_scope']),r['area'],r['longitude'],r['latitude'],r['source_url'],r['evidence_id'],json.dumps(r['place_keys'],ensure_ascii=False)] for r in facilities])
        publish(db, cfg['facility_database'])

        guard = read('data/reference/structured_rules.json')
        cfg = read('configs/structured_guard_v1.json')
        db = stage/'guard.duckdb'
        with duckdb.connect(str(db)) as con:
            for name, rows in guard.items():
                if name == 'guard_rules':
                    con.execute('CREATE TABLE guard_rules(id VARCHAR PRIMARY KEY,approval_status VARCHAR,payload VARCHAR)')
                    con.executemany('INSERT INTO guard_rules VALUES (?,?,?)', rows)
                else:
                    if name not in {'guard_coverage','guard_evidence','guard_sources'}:
                        raise ValueError('Unexpected rule table')
                    con.execute('CREATE TABLE '+name+'(id VARCHAR PRIMARY KEY,payload VARCHAR)')
                    con.executemany('INSERT INTO '+name+' VALUES (?,?)', rows)
        publish(db, cfg['database'])
        # Harbour builder consumes only the approved config and sourced geometry.
        initialize(root)
        tolls = 'data/processed/taxi_surcharges_v1.duckdb'
        files[tolls] = digest(root/tolls)

        corpus = root/'data/curated/chunks.jsonl'
        chunks = [json.loads(line) for line in corpus.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(chunks) != inputs['chunk_count']:
            raise ValueError('Corpus count differs')
        config = read('configs/retrieval_v1.json')
        model = read('configs/models.json')['embedding']
        print('Encoding reviewed corpus locally on CPU...', flush=True)
        encoder = DenseEncoder(root/model['local_path'])
        vectors, ms, lengths = encoder.encode([c['text'] for c in chunks])
        if vectors.shape != (len(chunks), 1024):
            raise ValueError('Unexpected embedding dimensions')
        np.save(stage/'embeddings.npy', vectors, allow_pickle=False)
        publish(corpus, 'artifacts/retrieval_index/chunks.jsonl')
        publish(stage/'embeddings.npy', 'artifacts/retrieval_index/embeddings.npy')
        index_manifest = {'version':'passenger-runtime-v1','chunk_count':len(chunks),'dimension':1024,
                          'corpus_sha256':digest(corpus),'model_revision':model['revision'],
                          'local_model':model['local_path'],'config_sha256':digest(root/'configs/retrieval_v1.json')}
        p = stage/'index_manifest.json'
        p.write_text(json.dumps(index_manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        publish(p, 'artifacts/retrieval_index/manifest.json')
        result = {'version':'runtime-build-v1','inputs_manifest_sha256':digest(root/'data/manifests/runtime_inputs.json'),
                  'files':files,'counts':{'chunks':len(chunks),'stands':len(stands),'facilities':len(facilities)},
                  'embedding_ms':ms, 'max_chunk_tokens':max(lengths), 'cloud_calls':0}
        existing.parent.mkdir(parents=True, exist_ok=True)
        tmp = existing.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(existing)
        return {'status':'built',**result['counts'],'generated_files':len(files),'cloud_calls':0}
    finally:
        # Never recursively delete a calculated directory. Staging is tiny and retained for diagnostics.
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Explicitly rebuild generated data; never changes approved inputs')
    args = parser.parse_args()
    print(json.dumps(build(force=args.force), ensure_ascii=False))
