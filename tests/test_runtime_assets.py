"""Portable loading and manifest guards; no cloud service dependencies."""
import hashlib,json
from pathlib import Path
import pytest
from hk_taxi.runtime_assets import digest,resolve_inside,verify_inputs,verify_asset

def test_asset_paths_cannot_escape_project(tmp_path):
    with pytest.raises(ValueError):resolve_inside(tmp_path,'../outside.txt')

def test_modified_approved_input_is_rejected(tmp_path):
    source=tmp_path/'data/curated/value.txt';source.parent.mkdir(parents=True);source.write_text('approved')
    m=tmp_path/'data/manifests/runtime_inputs.json';m.parent.mkdir(parents=True)
    m.write_text(json.dumps({'files':{'data/curated/value.txt':digest(source)}}))
    verify_inputs(tmp_path);source.write_text('changed')
    with pytest.raises(ValueError):verify_inputs(tmp_path)

def test_database_hash_is_machine_local_but_input_version_is_pinned(tmp_path):
    p=tmp_path/'data/manifests/runtime_inputs.json';p.parent.mkdir(parents=True);p.write_text('{"files":{}}')
    db=tmp_path/'data/processed/db.duckdb';db.parent.mkdir(parents=True);db.write_bytes(b'local database')
    m=tmp_path/'data/processed/runtime_manifest.json';m.write_text(json.dumps({'inputs_manifest_sha256':digest(p),'files':{'data/processed/db.duckdb':digest(db)}}))
    assert verify_asset(tmp_path,'data/processed/db.duckdb')==db
    db.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_asset(tmp_path,'data/processed/db.duckdb')
