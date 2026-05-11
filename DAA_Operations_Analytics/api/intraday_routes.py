from flask import Blueprint, jsonify, request, Response
import os, json, csv, io
from datetime import datetime

intraday_blueprint = Blueprint('intraday', __name__)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLIGHTS_CSV  = os.path.join(BASE_DIR, 'data', 'inputs', 'intraday_flights.csv')
STANDS_CSV   = os.path.join(BASE_DIR, 'data', 'inputs', 'intraday_stands.csv')
ALLOC_JSON   = os.path.join(BASE_DIR, 'data', 'outputs', 'intraday_allocation.json')
LOG_JSON     = os.path.join(BASE_DIR, 'data', 'outputs', 'intraday_event_log.json')

# In-memory simulation state (reset on each /reset call)
_state = {'flights': None, 'stands': None, 'event_log': [], 'golden': None}


def _load_state():
    from engine.intraday_engine import load_intraday_flights, load_intraday_stands
    _state['flights'] = load_intraday_flights(FLIGHTS_CSV)
    _state['stands']  = load_intraday_stands(STANDS_CSV)
    _state['event_log'] = []


def _save_alloc():
    from engine.intraday_engine import flights_to_gantt, detect_conflicts, generate_recommendation
    gantt = flights_to_gantt(_state['flights'], _state['stands'])
    payload = {
        'gantt': gantt,
        'event_log': _state['event_log'],
        'recommendation': generate_recommendation(gantt)
    }
    os.makedirs(os.path.dirname(ALLOC_JSON), exist_ok=True)
    with open(ALLOC_JSON, 'w') as f:
        json.dump(payload, f, indent=2)


# ── GET /api/intraday/golden-plan ─────────────────────────────────────────────
@intraday_blueprint.route('/golden-plan', methods=['GET'])
def golden_plan():
    try:
        _load_state()
        from engine.intraday_engine import solve_milp, flights_to_gantt, detect_conflicts, generate_recommendation

        result = solve_milp(_state['flights'], _state['stands'])
        _state['golden'] = result

        gantt      = flights_to_gantt(_state['flights'], _state['stands'])
        conflicts  = detect_conflicts(_state['flights'])

        # KPIs
        total      = len(_state['flights'])
        remote_ct  = sum(1 for g in gantt if g['stand'].startswith('R'))
        contact_ct = total - remote_ct
        cbp_ct     = sum(1 for g in gantt if g['cbp'])

        # Generate operational decisions for each flight
        ops_decisions = []
        for fl in _state['flights']:
            g = next((x for x in gantt if x['flight_no'] == fl.flight_no), None)
            if not g:
                continue
            stand = g['stand']
            badge = g['badge']
            # Operational language per scenario
            if badge == 'QUICK_TURN':
                mins = int((fl.etd - fl.eta).total_seconds() / 60) if fl.eta and fl.etd else 45
                ops_decisions.append({'flight': fl.flight_no, 'badge': badge,
                    'action': f'Push quick turnaround — {mins} min gate window on Stand {stand}. Ground crew priority dispatch.'})
            elif badge == 'ARR_PARK':
                ops_decisions.append({'flight': fl.flight_no, 'badge': badge,
                    'action': f'Arrival parking on Stand {stand} — hold at gate until next departure slot.'})
            elif badge == 'DEP_HOLD':
                ops_decisions.append({'flight': fl.flight_no, 'badge': badge,
                    'action': f'Departure hold — aircraft pre-positioned at Stand {stand}. Taxi clearance at ETD-10.'})
            elif badge == 'CBP_CONTACT':
                ops_decisions.append({'flight': fl.flight_no, 'badge': badge,
                    'action': f'CBP Preclearance routing — assigned Stand {stand} (T2 CBP-eligible pier).'})
            elif badge == 'WIDEBODY_PIER':
                ops_decisions.append({'flight': fl.flight_no, 'badge': badge,
                    'action': f'Widebody {fl.aircraft_type} assigned MARS Stand {stand} — adjacent stands blocked.'})
            elif badge == 'DEGRADED_REMOTE':
                ops_decisions.append({'flight': fl.flight_no, 'badge': badge,
                    'action': f'Remote apron {stand} — PAX bus transfer required (~8 min). No contact stand available.'})
            elif fl.preferred_stand and stand != fl.preferred_stand:
                ops_decisions.append({'flight': fl.flight_no, 'badge': badge,
                    'action': f'Reassigned from preferred {fl.preferred_stand} to {stand} — taxi hold 5 min after landing.'})

        stand_rows = [{'stand_id': s.stand_id, 'type': s.stype,
                       'icao_max': s.icao_cat_max, 'pier': s.pier,
                       'cbp': s.cbp_eligible, 'mars': s.mars_group, 'active': s.active}
                      for s in _state['stands']]

        payload = {
            'status':       'success',
            'solver_status': result.get('status'),
            'gantt':        gantt,
            'stands':       stand_rows,
            'conflict_log': result.get('conflict_log', []),
            'live_conflicts': conflicts,
            'ops_decisions': ops_decisions,
            'recommendation': generate_recommendation(gantt),
            'kpi': {
                'total_flights': total,
                'contact_assigned': contact_ct,
                'remote_assigned': remote_ct,
                'remote_pct': round(remote_ct / total * 100, 1),
                'cbp_flights': cbp_ct,
            }
        }
        _save_alloc()
        return jsonify(payload)

    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e),
                        'trace': traceback.format_exc()}), 500


# ── POST /api/intraday/event ──────────────────────────────────────────────────
@intraday_blueprint.route('/event', methods=['POST'])
def apply_event():
    try:
        if _state['flights'] is None:
            return jsonify({'status': 'error', 'message': 'Run golden-plan first'}), 400

        data        = request.get_json(silent=True) or {}
        trigger     = data.get('trigger', 'ARRIVAL_DELAY')
        flight_no   = data.get('flight_no', '')
        delta       = data.get('delta_minutes', 0)
        stand_id    = data.get('stand_id', '')
        w_start_str = data.get('weather_start', '')
        w_end_str   = data.get('weather_end', '')

        from engine.intraday_engine import (
            resolve_event, flights_to_gantt, detect_conflicts,
            generate_recommendation, solve_milp, SIM_START
        )
        from datetime import datetime, timedelta

        log_entries = []

        # ── Map new trigger keys → engine event types ──
        ENGINE_MAP = {
            'FLIGHT_DELAY':    'FLIGHT_DELAY',
            'ARRIVAL_DELAY':   'ETA_SHIFT',
            'DEPARTURE_DELAY': 'ETD_SHIFT',
            'STAND_CLOSE':     'STAND_CLOSE',
            'MAYDAY':          'MAYDAY',
        }

        if trigger == 'WEATHER_STOP':
            # Push all flights inside window; then re-solve
            if w_start_str and w_end_str:
                base = SIM_START.replace(hour=0, minute=0)
                def parse_t(s):
                    h, m = map(int, s.split(':'))
                    return SIM_START.replace(hour=h, minute=m)
                w_start = parse_t(w_start_str)
                w_end   = parse_t(w_end_str)
                affected = []
                for fl in _state['flights']:
                    changed = False
                    if fl.eta and w_start <= fl.eta <= w_end:
                        fl.eta = w_end
                        changed = True
                    if fl.etd and w_start <= fl.etd <= w_end:
                        fl.etd = w_end + timedelta(minutes=10)
                        changed = True
                    if changed:
                        affected.append(fl.flight_no)
                        log_entries.append({
                            'flight': fl.flight_no, 'event_type': 'WEATHER_STOP',
                            'reason': {'step_1_pass': f'Pushed past weather window {w_start_str}–{w_end_str}'},
                            'stand': None, 'badge': 'BUFFER_HOLD'
                        })
                # Re-run full MILP after weather displacement
                _state['golden'] = solve_milp(_state['flights'], _state['stands'])
            _save_alloc()

        elif trigger == 'STAND_CLOSE':
            for st in _state['stands']:
                if st.stand_id == stand_id:
                    st.active = False
            # Grace: only re-route flights whose occupancy has ENDED (or not started)
            affected = [f for f in _state['flights']
                        if f.assigned_stand == stand_id]
            for fl in affected:
                entry = resolve_event(_state['flights'], _state['stands'],
                                      fl.flight_no, 0, 'STAND_CLOSE')
                log_entries.append(entry)
            _state['event_log'].extend(log_entries)
            _save_alloc()

        else:
            engine_type = ENGINE_MAP.get(trigger, trigger)
            entry = resolve_event(_state['flights'], _state['stands'],
                                  flight_no, delta, engine_type)
            log_entries.append(entry)
            _state['event_log'].append(entry)

        # Full MILP re-solve after every event to globally re-optimise
        _state['golden'] = solve_milp(_state['flights'], _state['stands'])
        _save_alloc()

        gantt      = flights_to_gantt(_state['flights'], _state['stands'])
        conflicts  = detect_conflicts(_state['flights'])
        rec        = generate_recommendation(gantt)

        return jsonify({
            'status':         'success',
            'gantt':          gantt,
            'live_conflicts': conflicts,
            'log_entries':    log_entries,
            'recommendation': rec,
        })

    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e),
                        'trace': traceback.format_exc()}), 500



# ── GET /api/intraday/export ──────────────────────────────────────────────────
@intraday_blueprint.route('/export', methods=['GET'])
def export_csv():
    try:
        if _state['flights'] is None:
            return jsonify({'status': 'error', 'message': 'No plan to export'}), 400

        from engine.intraday_engine import flights_to_gantt
        gantt = flights_to_gantt(_state['flights'], _state['stands'])

        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(['Flight', 'Airline', 'Type', 'Stand', 'Badge',
                    'ETA', 'ETD', 'Start_Min', 'End_Min', 'Terminal', 'CBP'])
        for g in gantt:
            w.writerow([g['flight_no'], g['airline'], g['ftype'], g['stand'],
                        g['badge'], g['eta'], g['etd'],
                        g['start_min'], g['end_min'], g['terminal'], g['cbp']])

        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        buf.seek(0)
        return Response(
            buf.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition':
                     f'attachment; filename=intraday_allocation_{ts}.csv'}
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── POST /api/intraday/reset ──────────────────────────────────────────────────
@intraday_blueprint.route('/reset', methods=['POST'])
def reset_simulation():
    _state['flights']   = None
    _state['stands']    = None
    _state['event_log'] = []
    _state['golden']    = None
    return jsonify({'status': 'success', 'message': 'Simulation reset to T=0'})


# ── GET /api/intraday/log ─────────────────────────────────────────────────────
@intraday_blueprint.route('/log', methods=['GET'])
def get_log():
    return jsonify({'status': 'success', 'event_log': _state['event_log']})
