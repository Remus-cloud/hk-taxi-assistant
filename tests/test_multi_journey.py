import json
from types import SimpleNamespace
import pytest
from hk_taxi.multi_journey import (select_anchor,parse_osm,forbidden,candidates_for,
 route_pairs,inspect_route,make_answer,MapClient,MapError,estimate_pairs)

CFG={'search_radius_m':400,'dedup_distance_m':40,'max_candidates':3,'max_routes_per_request':3,'timeout_seconds':20}

def candidate(identifier,x=114.17,y=22.3,kind='ground_road_reference'):
    return {'id':identifier,'name':identifier,'coordinate':[x,y],'kind':kind}

def endpoint(kind='place',count=3):
    return {'anchor':{'query':'sample','kind':kind},'candidates':[candidate(str(i)) for i in range(count)]}

def test_exact_anchor_excludes_aed():
    rows=[{'nameZH':'自动心脏除颤器 - 香港中央圖書館','nameEN':'Automated Defibrillator - Hong Kong Central Library','x':837586,'y':815670},
          {'nameZH':'香港中央圖書館','nameEN':'Hong Kong Central Library','x':837586,'y':815670}]
    assert select_anchor('香港中央图书馆',rows)['candidate_index']==1

def test_region_station_fallback_is_explicit():
    row={'nameZH':'九龍塘站','nameEN':'Kowloon Tong Station','x':836500,'y':821000}
    a=select_anchor('九龙塘',[row],'area')
    assert '同名车站' in a['assumption']
    with pytest.raises(ValueError): select_anchor('九龙塘',[row],'place')

def test_no_unrelated_first_result():
    with pytest.raises(ValueError): select_anchor('酒店',[{'nameZH':'香港中央圖書館','x':837586,'y':815670}])

@pytest.mark.parametrize('tags',[{'bridge':'yes'},{'layer':'1'},{'vehicle':'no','bus':'yes'},
 {'access':'private'},{'access:conditional':'yes @ something'},{'highway':'motorway'},{'highway':'footway'},{'highway':'busway'}])
def test_bad_access_rejected(tags): assert forbidden(tags)

def test_public_road_compatible_not_permission_certificate():
    assert forbidden({'highway':'tertiary'}) is None

def fixture_geometry():
    nodes={'1':{'coordinate':[114.17,22.3],'tags':{}},
      '2':{'coordinate':[114.1701,22.3],'tags':{}},
      '3':{'coordinate':[114.1702,22.3],'tags':{'barrier':'gate'}},
      '4':{'coordinate':[114.1703,22.3],'tags':{}},
      '5':{'coordinate':[114.1704,22.3],'tags':{}}}
    ways=[{'id':'a','tags':{'name':'Ground Road','highway':'tertiary'},'node_ids':['1','2']},
      {'id':'b','tags':{'highway':'service'},'node_ids':['1','3','4']},
      {'id':'c','tags':{'name':'Flyover','highway':'secondary','bridge':'yes','layer':'1'},'node_ids':['4','5']}]
    return nodes,ways

def test_extract_dedup_gate_and_bridge():
    n,w=fixture_geometry()
    result=candidates_for({'coordinate':[114.17,22.3],'query':'Sample','kind':'place'},n,w,[],CFG)
    assert [c['id'] for c in result['candidates']]==['osm-node-1']
    assert result['candidates'][0]['kind']=='service_road_junction'
    assert not result['candidates'][0]['curb_permission_verified']
    assert any(r['reason']=='same_road_duplicate' for r in result['rejected'])

def test_stand_and_access_diversity():
    n,w=fixture_geometry()
    stands=[{'stand_id':i,'longitude':114.17,'latitude':22.3001+i*0.0001,'status_tc':'市區的士站','location_tc':'S'+str(i)} for i in range(3)]
    result=candidates_for({'coordinate':[114.17,22.3],'query':'Sample','kind':'place'},n,w,stands,CFG)
    assert len(result['candidates'])==3
    assert any(c['kind']=='service_road_junction' for c in result['candidates'])

def test_radius_excludes_far_away_stand():
    n,w=fixture_geometry()
    stands=[{'stand_id':1,'longitude':115,'latitude':22.3,'status_tc':'市區的士站','location_tc':'far'}]
    r=candidates_for({'coordinate':[114.17,22.3],'query':'Sample','kind':'place'},n,w,stands,CFG)
    assert not any(c['kind']=='official_taxi_stand' for c in r['candidates'])

def test_xml_and_json_osm_equivalence():
    raw=b'<osm><node id="1" lat="22.3" lon="114.17"/><way id="2"><nd ref="1"/><tag k="highway" v="tertiary"/></way></osm>'
    data={'elements':[{'type':'node','id':1,'lat':22.3,'lon':114.17},{'type':'way','id':2,'nodes':[1],'tags':{'highway':'tertiary'}}]}
    assert parse_osm(raw)==parse_osm(json.dumps(data).encode())

def test_no_nine_pair_explosion():
    assert len(route_pairs(endpoint(),endpoint()))==3
    assert len(route_pairs(endpoint('area'),endpoint('area')))==1
    assert len(route_pairs(endpoint(count=1),endpoint()))==3
    assert route_pairs(endpoint(count=0),endpoint())==[]

def response(name='Ground Road'):
    return {'features':[{'geometry':{'coordinates':[[114.17,22.3],[114.18,22.31]]},'properties':{
        'summary':{'distance':3000,'duration':500},'segments':[{'steps':[{'name':name}]}]}}]}

def test_bridge_at_endpoint_not_quoted():
    result=inspect_route(response('大坑道天橋'),candidate('a'),candidate('b',114.18,22.31))
    assert result['review_flags'] and result['status']=='needs_route_review'

def test_valid_route_stays_provisional():
    result=inspect_route(response(),candidate('a'),candidate('b',114.18,22.31))
    assert result['status']=='provisional_distance_estimate'
    assert result['taxi_stopping_permission_verified'] is False

@pytest.mark.parametrize('data',[{}, {'features':[]},response()])
def test_invalid_response(data):
    if data.get('features'): data['features'][0]['properties']['summary']['distance']=float('nan')
    with pytest.raises(MapError): inspect_route(data,candidate('a'),candidate('b'))

def test_region_answer_estimate_then_short_question():
    r={'origin_id':'A','destination_id':'B','origin_name':'旺角站附近','destination_name':'九龙塘站附近',
       'distance_m':3000,'estimate':{'fare_cent':3950,'fare_hkd':'39.50'}}
    text=make_answer(endpoint('area'),endpoint('area'),[r])
    assert text.index('HK$39.50')<text.index('你有更具体')
    assert '不代表整个区域' in text and '咪表计时' in text

def test_no_data_no_fabricated_price():
    text=make_answer(endpoint(),endpoint(),[])
    assert '不会猜测距离' in text and 'HK$' not in text

def test_budget_before_reading_key(tmp_path):
    client=MapClient(tmp_path,CFG,budget=0)
    with pytest.raises(MapError,match='budget'): client.route(candidate('a'),candidate('b'))

def test_api_failure_stops_without_retry():
    class Fake:
        config=CFG
        calls=0
        def route(self,*args):
            self.calls+=1
            raise MapError('HTTP 429; no automatic retry.')
    client=Fake()
    results=estimate_pairs(client,endpoint(),endpoint(),None)
    assert client.calls==1 and results[0]['status']=='tool_error'

def test_confirmed_selection_does_not_ask_same_question_again():
    text=make_answer(endpoint(),endpoint(),[],selection_confirmed=True)
    assert '你准备从哪个' not in text
    assert '本会话' in text

