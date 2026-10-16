from copy import deepcopy
from pathlib import Path
import pytest
from test_unified import populated,plan,FakeBackend,choose
from hk_taxi.unified import apply_plan
from hk_taxi.multi_journey import MapError
from hk_taxi.harbour_tolls import quote

ROOT=Path(__file__).resolve().parents[1]

def exempt():
    s=populated();s['conditions']['boarding_type']='cross_harbour_stand'
    s['origin']=choose(s['origin'],'A0');s['destination']=choose(s['destination'],'B0')
    return s

@pytest.mark.parametrize('p',[
    plan(origin={'query':'C','kind':'place'}),plan(action='reverse')])
def test_changed_pickup_resets_exemption_preserves_luggage(p):
    s=exempt();before=deepcopy(s);n=apply_plan(s,p,FakeBackend())
    assert n['conditions']['boarding_type']=='ordinary'
    assert n['conditions']['luggage']==s['conditions']['luggage']
    assert n['conditions']['taxi_type']==s['conditions']['taxi_type']
    assert s==before

@pytest.mark.parametrize('p',[
    plan(origin={'query':'A','kind':'place'}),
    plan(origin={'query':'A','kind':'area'}),
    plan(origin_choice='A1'),
    plan(destination={'query':'C','kind':'place'}),
    plan(conditions={'luggage':'none','boarding_type':None})])
def test_same_place_refinement_destination_and_luggage_keep_exemption(p):
    assert apply_plan(exempt(),p,FakeBackend())['conditions']['boarding_type']=='cross_harbour_stand'

@pytest.mark.parametrize('p',[
    plan(origin={'query':'C','kind':'place'},conditions={'boarding_type':'cross_harbour_stand'}),
    plan(action='reverse',conditions={'boarding_type':'cross_harbour_stand'})])
def test_explicit_new_pickup_fact_overrides_default(p):
    assert apply_plan(exempt(),p,FakeBackend())['conditions']['boarding_type']=='cross_harbour_stand'

def test_failed_pickup_update_keeps_old_state():
    s=exempt();before=deepcopy(s)
    with pytest.raises(MapError):apply_plan(s,plan(origin={'query':'missing','kind':'place'}),FakeBackend())
    assert s==before

def test_rule_only_cannot_reset_exemption():
    s=exempt()
    assert apply_plan(s,plan(tasks=['rules'],origin={'query':'C','kind':'place'}),FakeBackend())==s

@pytest.mark.parametrize('reverse',[False,True])
def test_reset_recalculates_surcharge_and_round_trip_still_exempt(monkeypatch,reverse):
    import hk_taxi.harbour_tolls as m
    monkeypatch.setattr(m,'load',lambda root:({'version':'test','source_url':'https://www.td.gov.hk'}, {},
        {'western':{'name':'西隧','toll_cent':2500,'return_cent':2500}}))
    crossings=[{'tunnel_id':'western'}]
    monkeypatch.setattr(m,'identify',lambda *a:{'status':'identified','crossings':crossings})
    s=exempt();route={'estimate':{'fare_cent':7940}}
    assert quote(ROOT,route,s)['subtotal_cent']==10440
    n=apply_plan(s,plan(action='reverse') if reverse else plan(origin={'query':'C','kind':'place'}),FakeBackend())
    assert quote(ROOT,route,n)['subtotal_cent']==12940
    crossings.append({'tunnel_id':'western'})
    q=quote(ROOT,route,n)
    assert q['return_cent']==0 and q['outbound_cent']==5000
    assert q['exemption']=='same_side_final_destination'
