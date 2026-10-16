"""Approved urban distance-only estimate. No time, toll, or extra-charge inference."""
from dataclasses import dataclass, asdict
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
import hashlib
import json
import duckdb


@dataclass(frozen=True)
class DistanceRule:
    version_id: str
    effective_from: str
    verified_as_of: str
    source_url: str
    base_distance_m: int
    base_fare_cent: int
    increment_distance_m: int
    meter_threshold_cent: int
    increment_before_threshold_cent: int
    increment_after_threshold_cent: int

    def validate(self):
        values = [self.base_distance_m, self.base_fare_cent, self.increment_distance_m,
                  self.meter_threshold_cent, self.increment_before_threshold_cent,
                  self.increment_after_threshold_cent]
        if any(type(x) is not int or x <= 0 for x in values):
            raise ValueError('Rule units must be positive integers.')
        delta = self.meter_threshold_cent - self.base_fare_cent
        if delta <= 0 or delta % self.increment_before_threshold_cent:
            raise ValueError('Unsupported meter threshold: review tariff before calculation.')
        date.fromisoformat(self.effective_from)
        date.fromisoformat(self.verified_as_of)


def calculate_distance(distance_m, rule, as_of=None):
    rule.validate()
    if isinstance(distance_m, bool):
        raise ValueError('Distance must be positive metres.')
    try:
        distance = Decimal(str(distance_m))
    except (InvalidOperation, ValueError):
        raise ValueError('Invalid distance.') from None
    if not distance.is_finite() or distance <= 0:
        raise ValueError('Distance must be positive finite metres; identical endpoints need clarification.')
    query_date = date.fromisoformat(as_of) if as_of else date.today()
    if query_date < date.fromisoformat(rule.effective_from):
        raise ValueError('Requested date predates the effective tariff; use an appropriate historical rule.')
    units = int((max(Decimal(0), distance - rule.base_distance_m) / rule.increment_distance_m)
                .to_integral_value(rounding=ROUND_CEILING))
    first_capacity = (rule.meter_threshold_cent - rule.base_fare_cent) // rule.increment_before_threshold_cent
    first_units, later_units = min(units, first_capacity), max(0, units-first_capacity)
    cents = rule.base_fare_cent + first_units*rule.increment_before_threshold_cent + later_units*rule.increment_after_threshold_cent
    return {'status':'distance_only_estimate','taxi_type':'urban','distance_m':str(distance),
            'fare_cent':cents, 'fare_hkd':format(Decimal(cents)/100,'.2f'),
            'breakdown':{'base_cent':rule.base_fare_cent,'first_tier_units':first_units,
                         'first_tier_cent':first_units*rule.increment_before_threshold_cent,
                         'later_tier_units':later_units,'later_tier_cent':later_units*rule.increment_after_threshold_cent},
            'rule_version':rule.version_id,'source_url':rule.source_url,'verified_as_of':rule.verified_as_of,
            'excluded':['time_based_meter_charge','tolls_and_return_tolls','luggage_and_other_surcharges'],
            'final_payable_total':None,'route_legality_validated':False}


def initialize_fares(db_path, config_path):
    raw=Path(config_path).read_bytes()
    config=json.loads(raw)
    if config.get('approval_status')!='approved' or config.get('execution_scope')!='urban_distance_only':
        raise ValueError('Executable distance rules must be explicitly approved.')
    rule=DistanceRule(**config['rule'])
    rule.validate()
    digest=hashlib.sha256(raw).hexdigest()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute('BEGIN TRANSACTION')
        try:
            conn.execute('''CREATE TABLE IF NOT EXISTS fare_versions (
              version_id VARCHAR PRIMARY KEY, taxi_type VARCHAR NOT NULL,
              effective_from DATE NOT NULL, effective_until DATE, verified_as_of DATE NOT NULL,
              source_url VARCHAR NOT NULL, approval_status VARCHAR NOT NULL, config_sha256 VARCHAR NOT NULL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS fare_steps (
              version_id VARCHAR PRIMARY KEY REFERENCES fare_versions(version_id),
              base_distance_m INTEGER NOT NULL, base_fare_cent INTEGER NOT NULL,
              increment_distance_m INTEGER NOT NULL, meter_threshold_cent INTEGER NOT NULL,
              increment_before_threshold_cent INTEGER NOT NULL, increment_after_threshold_cent INTEGER NOT NULL,
              waiting_execution_enabled BOOLEAN NOT NULL DEFAULT FALSE)''')
            existing=conn.execute('SELECT config_sha256 FROM fare_versions WHERE version_id=?',[rule.version_id]).fetchone()
            if existing and existing[0]!=digest:
                raise ValueError('Existing version has different rules; create a new version instead.')
            if not existing:
                conn.execute('INSERT INTO fare_versions VALUES (?, ?, ?, NULL, ?, ?, ?, ?)',
                   [rule.version_id,'urban',rule.effective_from,rule.verified_as_of,rule.source_url,'approved',digest])
                conn.execute('INSERT INTO fare_steps VALUES (?, ?, ?, ?, ?, ?, ?, FALSE)',
                   [rule.version_id,rule.base_distance_m,rule.base_fare_cent,rule.increment_distance_m,
                    rule.meter_threshold_cent,rule.increment_before_threshold_cent,rule.increment_after_threshold_cent])
            conn.execute('COMMIT')
        except Exception:
            conn.execute('ROLLBACK')
            raise
    return rule


def load_distance_rule(db_path, version_id='urban-distance-20240714-v1'):
    with duckdb.connect(str(db_path),read_only=True) as conn:
        row=conn.execute('''SELECT v.version_id, CAST(v.effective_from AS VARCHAR), CAST(v.verified_as_of AS VARCHAR),
          v.source_url,s.base_distance_m,s.base_fare_cent,s.increment_distance_m,s.meter_threshold_cent,
          s.increment_before_threshold_cent,s.increment_after_threshold_cent
          FROM fare_versions v JOIN fare_steps s USING(version_id)
          WHERE v.version_id=? AND v.taxi_type='urban' AND v.approval_status='approved'
          AND s.waiting_execution_enabled=FALSE''',[version_id]).fetchone()
    if row is None: raise ValueError('No approved distance tariff found. Initialize explicitly first.')
    return DistanceRule(*row)
