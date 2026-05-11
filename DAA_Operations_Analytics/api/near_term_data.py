import csv
import os
from datetime import datetime, timedelta

# Resolve project root: go up from api/ to project root
_HERE = os.path.dirname(os.path.abspath(__file__))  # …/api
_ROOT = os.path.dirname(_HERE)                       # …/project root

def _csv_path(root, *parts):
    return os.path.join(root, 'data', 'inputs', *parts)


def _parse_time(t):
    try:
        h, m = map(int, t.strip().split(':')[:2])
        return h * 60 + m  # minutes from midnight
    except Exception:
        return None


def load_schedule():
    """Load both CSVs into a unified list suitable for the Gantt grid."""
    schedule = []

    try:
        with open(_csv_path(_ROOT, 'flights_arrivals.csv'), encoding='cp1252') as f:
            for row in csv.DictReader(f):
                t = row.get('sta', '').strip()
                minutes = _parse_time(t)
                if minutes is None:
                    continue
                gate = row.get('gate', '').strip()
                # Estimate on-stand duration: arrival touch-down to off-blocks ~80 min
                schedule.append({
                    'type': 'ARR',
                    'flight': row.get('flight_no', '').strip(),
                    'airline': row.get('airline_name', '').strip().split('(')[0].strip(),
                    'route': row.get('origin', '').strip().replace('\xa0', ' ').split('(')[0].strip(),
                    'origin_code': row.get('origin_code', '').strip(),
                    'gate': gate,
                    'time': t,
                    'start_min': max(0, minutes - 15),   # on stand ~15 min before STA
                    'end_min': minutes + 80,              # off stand ~80 min after STA
                    'aircraft': row.get('aircraft_type', '').strip(),
                    'cbp': row.get('cbp_flag', 'FALSE').strip().upper() == 'TRUE',
                    'cat': row.get('icao_cat', '').strip(),
                })
    except Exception:
        pass

    try:
        with open(_csv_path(_ROOT, 'flights_departures.csv'), encoding='cp1252') as f:
            for row in csv.DictReader(f):
                t = row.get('std', '').strip()
                minutes = _parse_time(t)
                if minutes is None:
                    continue
                gate = row.get('gate', '').strip()
                # On stand: ~70 min before departure until push-back
                schedule.append({
                    'type': 'DEP',
                    'flight': row.get('flight_no', '').strip(),
                    'airline': row.get('airline_name', '').strip().split('(')[0].strip(),
                    'route': row.get('destination', '').strip().replace('\xa0', ' ').split('(')[0].strip(),
                    'dest_code': row.get('destination_code', '').strip(),
                    'gate': gate,
                    'time': t,
                    'start_min': max(0, minutes - 70),   # on stand 70 min before STD
                    'end_min': minutes + 5,               # off stand ~5 min after pushback
                    'aircraft': row.get('aircraft_type', '').strip(),
                    'cbp': False,
                    'cat': row.get('icao_cat', '').strip(),
                })
    except Exception:
        pass

    return schedule


def _filter_flights(schedule, start_h, end_h, airline_kw=None, cbp_only=False, route_codes=None):
    out = []
    for f in schedule:
        t = _parse_time(f['time'])
        if t is None:
            continue
        h = t // 60
        if not (start_h <= h < end_h):
            continue
        if airline_kw and airline_kw.lower() not in f['airline'].lower():
            continue
        if cbp_only and not f.get('cbp'):
            continue
        if route_codes:
            code = f.get('origin_code', '') + f.get('dest_code', '')
            if not any(c in code for c in route_codes):
                continue
        out.append(f)
    return out


def _fmt_sample(flights, max_n=8):
    """Format compact affected flight records for detail cards."""
    out = []
    for f in flights[:max_n]:
        rec = {
            'dir': f['type'],
            'flight': f['flight'],
            'airline': f['airline'],
            'route': f'{f.get("route","")[:18]} → DUB' if f['type'] == 'ARR' else f'DUB → {f.get("route","")[:18]}',
            'time': f['time'],
            'gate': f['gate'],
            'aircraft': f['aircraft'],
            'cbp': f.get('cbp', False),
        }
        out.append(rec)
    if len(flights) > max_n:
        out.append({'_more': len(flights) - max_n})
    return out


def get_near_term_data():
    """
    Returns (risk_14_days, full_schedule_json) — the full schedule is for the Gantt grid.
    Risk events define impact windows that the frontend will highlight on the grid.
    """
    schedule = load_schedule()
    today = datetime.now()
    days = []

    for i in range(1, 15):
        target_date = today + timedelta(days=i)
        base_movements = 820 if target_date.weekday() in [4, 6] else 780

        day = {
            'day_index': i,
            'date': target_date.strftime('%d %b'),
            'weekday': target_date.strftime('%a').upper(),
            'weekday_full': target_date.strftime('%A'),
            'risk_level': 'green',
            'risk_score': 15,
            'risk_category': 'Normal Operations',
            'base_movements': base_movements,
            'llm_headline': 'No significant disruptions. Normal operations expected.',
            'probability': '10%',
            'impact_window': 'All day',
            'impact_start_h': 0,
            'impact_end_h': 24,
            'flight_scope': 'All flights operating normally.',
            'schedule_impact': 'No delays projected.',
            'stand_pressure': 'Normal buffer margins maintained.',
            'contingency': 'Standard SOPs apply. No advance action required.',
            'affected_flights': [],
            'affected_gate_filter': None,   # None = no filter; list of gates = only those
            'affected_airline_kw': None,
            'sources': ['DAA NOTAM feed', 'Met Éireann — clear'],
        }

        # ── Day 3: UK ATC Congestion (Yellow) ────────────────────────────────
        if i == 3:
            uk_codes = ['LHR','LCY','MAN','LPL','STN','GLA','EDI','BFS','BHX','NCL']
            flt = _filter_flights(schedule, 14, 20, route_codes=uk_codes)
            day.update({
                'risk_level': 'yellow',
                'risk_score': 42,
                'risk_category': 'ATC / Airspace',
                'llm_headline': 'UK NATS sector congestion — network flow rate restriction issued 14:00–20:00 UTC.',
                'probability': '78%',
                'impact_window': '14:00 – 20:00',
                'impact_start_h': 14,
                'impact_end_h': 20,
                'flight_scope': f'{len(flt)} UK-sector flights in affected window (arrivals & departures)',
                'schedule_impact': '+15–25 min avg delay on UK-sector movements. Arrival bunching 16:00–17:00.',
                'stand_pressure': 'Pier A and B stand buffers may drop below 20 min margin during 16:00–18:00.',
                'contingency': 'Monitor T1 Pier A turnaround buffers. Pre-brief handlers on potential slot swaps. Issue Crew Delay Notifications to affected airlines before 13:00.',
                'affected_flights': _fmt_sample(flt),
                'sources': ['UK NATS NOTAM #2026/0412', 'Eurocontrol CFMU advisory #DUB-04-12'],
            })

        # ── Day 6: Ryanair Schedule Add (Orange) ─────────────────────────────
        elif i == 6:
            flt = _filter_flights(schedule, 16, 22, airline_kw='Ryanair')
            day.update({
                'risk_level': 'orange',
                'risk_score': 71,
                'risk_category': 'Airline Schedule Change',
                'llm_headline': 'Ryanair: 8 unplanned repositioning movements filed 16:00–22:00. Pier D saturation risk.',
                'probability': '100%',
                'impact_window': '16:00 – 22:00',
                'impact_start_h': 16,
                'impact_end_h': 22,
                'flight_scope': f'{len(flt)} Ryanair movements in window + 8 new additions',
                'schedule_impact': 'Pier D stand demand exceeds plan by ~6 stands. T1 remote apron will reach 94% utilisation.',
                'stand_pressure': 'Stands 201–210 fully committed. Remote apron required for at least 4 aircraft overnight.',
                'contingency': 'Pre-assign remote stands 114–118 for overnight positioning. Move 2 Ryanair turns to Pier B. Ground ops brief at 14:00. Alert slot coordinators.',
                'affected_flights': _fmt_sample(flt),
                'affected_airline_kw': 'Ryanair',
                'sources': ['Ryanair slot submission received 11 Apr', 'IATA STS system alert #IE-2026-4481'],
            })

        # ── Day 8: Runway 28L Maintenance (Red) ──────────────────────────────
        elif i == 8:
            flt = _filter_flights(schedule, 9, 15)
            day.update({
                'risk_level': 'red',
                'risk_score': 93,
                'risk_category': 'Airport Maintenance / NOTAM',
                'llm_headline': 'NOTAM: Rwy 28L threshold inspection 10:00–14:00. Single-runway ops — 25% throughput reduction.',
                'probability': '98%',
                'impact_window': '10:00 – 14:00',
                'impact_start_h': 10,
                'impact_end_h': 14,
                'flight_scope': f'All {len(flt)} movements 09:00–15:00 affected by reduced runway throughput',
                'schedule_impact': 'GDP expected. +20–40 min avg departure delay. 8–10 arrivals held in stack during peak.',
                'stand_pressure': 'Stand occupancy extends 30–45 min beyond plan 10:00–13:00. Critical saturation window at 11:00.',
                'contingency': 'Activate Ground Delay Programme (GDP). Pre-hold 5 departures at gate until 10:30. Assign T1 remote overflow. Notify all airlines by 07:00.',
                'affected_flights': _fmt_sample(flt),
                'sources': ['DAA ATM NOTAM #DUB/2026/0387', 'Scheduled engineering ops brief'],
            })

        # ── Day 11: Wind Warning (Yellow) ────────────────────────────────────
        elif i == 11:
            flt = _filter_flights(schedule, 8, 18, cbp_only=False)
            day.update({
                'risk_level': 'yellow',
                'risk_score': 52,
                'risk_category': 'Meteorological',
                'llm_headline': 'Met Éireann Yellow Wind Warning: gusts 28–35kt from 280°. Elevated go-around risk and extended turnarounds.',
                'probability': '63%',
                'impact_window': '08:00 – 18:00',
                'impact_start_h': 8,
                'impact_end_h': 18,
                'flight_scope': f'All {len(flt)} movements during window — CBP Preclearance flights are time-critical',
                'schedule_impact': '+10–20 min occupancy extension per stand caused by crosswind caution. Possible diversions to SNN.',
                'stand_pressure': 'T2 Pier E CBP stands at risk of overrun if 2+ diversions return late.',
                'contingency': 'Suspend apron maintenance. Brief tug drivers on windsock compliance. Hold CBP overflow plan on standby. Open comms with Shannon Airport for potential diversions.',
                'affected_flights': _fmt_sample(flt),
                'sources': ['Met Éireann bulletin 13:00Z', 'DAA ATM Met watch #DUB-MW-0041'],
            })

        days.append(day)

    return days, schedule
