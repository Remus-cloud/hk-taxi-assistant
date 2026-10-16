from dataclasses import replace
from pathlib import Path
import json
import pytest
from hk_taxi.fare_calculator import DistanceRule, calculate_distance, initialize_fares, load_distance_rule

RULE=DistanceRule('urban-distance-20240714-v1','2024-07-14','2026-10-15','https://www.td.gov.hk/',2000,2900,200,10250,210,140)

@pytest.mark.parametrize('distance,expected',[(1,2900),(1999.9,2900),(2000,2900),('2000.01',3110),
 (2200,3110),('2200.01',3320),(5000,6050),(8800,10040),(9000,10250),('9000.01',10390),(9200,10390),(9400,10530),(10000,10950)])
def test_independent_tariff_examples(distance,expected):
    result=calculate_distance(distance,RULE,'2026-10-15')
    assert result['fare_cent']==expected
    assert result['final_payable_total'] is None
    assert result['breakdown']['base_cent']+result['breakdown']['first_tier_cent']+result['breakdown']['later_tier_cent']==expected

@pytest.mark.parametrize('value',[0,-1,True,'NaN','Infinity','unknown',None])
def test_invalid_distance(value):
    with pytest.raises(ValueError): calculate_distance(value,RULE,'2026-10-15')

@pytest.mark.parametrize('as_of',['2024-07-13'])
def test_before_effective_date(as_of):
    with pytest.raises(ValueError): calculate_distance(2000,RULE,as_of)

def test_bad_threshold():
    with pytest.raises(ValueError): calculate_distance(3000,replace(RULE,meter_threshold_cent=10251),'2026-10-15')

def test_monotonic_and_exact_integer_units():
    values=[calculate_distance(d,RULE,'2026-10-15')['fare_cent'] for d in range(1,15001,17)]
    assert values==sorted(values)

def test_sql_roundtrip_and_idempotency(tmp_path):
    import duckdb
    source=Path(__file__).resolve().parents[1]/'configs/fare_distance_v1.json'
    db=tmp_path/'fares.duckdb'
    with duckdb.connect(str(db)) as conn:
        conn.execute('CREATE TABLE preserved_data(i INTEGER)')
        conn.execute('INSERT INTO preserved_data VALUES (7)')
    initialize_fares(db,source)
    initialize_fares(db,source)
    assert load_distance_rule(db)==DistanceRule(**json.loads(source.read_text(encoding='utf-8'))['rule'])
    with duckdb.connect(str(db),read_only=True) as conn:
        assert conn.execute('SELECT * FROM preserved_data').fetchall()==[(7,)]
        assert conn.execute('SELECT COUNT(*) FROM fare_versions').fetchone()==(1,)
        assert conn.execute('SELECT waiting_execution_enabled FROM fare_steps').fetchone()==(False,)
    edited=json.loads(source.read_text(encoding='utf-8'))
    edited['rule']['base_fare_cent']=3000
    edited['rule']['meter_threshold_cent']=10350
    changed=tmp_path/'changed.json'
    changed.write_text(json.dumps(edited),encoding='utf-8')
    with pytest.raises(ValueError,match='different rules'): initialize_fares(db,changed)
