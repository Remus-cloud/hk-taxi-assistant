from pathlib import Path
from copy import deepcopy
from hk_taxi.journey_feedback import fare_date_feedback,present_journey_failure,TARIFF_INTERVAL_ERROR
from hk_taxi.unified import journey_text,empty_state
ROOT=Path(__file__).resolve().parents[1]

def test_known_date_failure_no_false_estimate_heading():
    text,refs=journey_text(empty_state(),[],TARIFF_INTERVAL_ERROR,'zh-Hans')
    assert '暂时没查到可用路线' in text and '按以下' not in text
    assert 'Requested date' not in text and refs==[] and '核验' not in text
def test_names_and_date_are_explained_in_saved_history():
    r={'question':'从豉油街到香港城市大学要多少钱','answer':'按以下估算：'+TARIFF_INTERVAL_ERROR,'state_after':{'origin':{'name':'豉油街'},'destination':{'name':'香港城市大學'}}}
    before=deepcopy(r);text=present_journey_failure(ROOT,r)
    assert '现在已恢复估算' in text and '2026-10-15' not in text
    assert r==before
def test_unrelated_answer_unchanged():
    assert present_journey_failure(ROOT,{'answer':'打印机坏了可提供手写收据。'})=='打印机坏了可提供手写收据。'
def test_mixed_rule_subanswer_preserved():
    tail='打印机坏了应给手写收据。 [S1](https://www.td.gov.hk)'
    r={'question':'行程多少钱，打印机坏了有收据吗','answer':TARIFF_INTERVAL_ERROR+'\n\n'+tail,'generation':{'rendered_answer':tail}}
    assert tail in present_journey_failure(ROOT,r)
def test_language_variants():
    assert '沒有算出車費' in fare_date_feedback(ROOT,'zh-Hant')
    assert 'Estimates are available again' in fare_date_feedback(ROOT,'en')
