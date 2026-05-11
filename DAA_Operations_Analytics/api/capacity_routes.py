"""
api/capacity_routes.py
Flask blueprint exposing the Capacity Intelligence Engine for the Policy Studio UI.
"""
from flask import Blueprint, jsonify, request
import os
import json

capacity_blueprint = Blueprint('capacity', __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDS_FIXED_PATH   = os.path.join(BASE_DIR, 'data', 'inputs', 'stands_final_fixed.csv')
ARRIVALS_HIST_PATH  = os.path.join(BASE_DIR, 'data', 'inputs', 'flights_arrivals.csv')
DEPARTURES_HIST_PATH= os.path.join(BASE_DIR, 'data', 'inputs', 'flights_departures.csv')
ARRIVALS_FWD_PATH   = os.path.join(BASE_DIR, 'data', 'inputs', 'flights_arrivals_next3days_match.csv')
AIRLINE_TERM_PATH   = os.path.join(BASE_DIR, 'data', 'inputs', 'airline_terminal.csv')


@capacity_blueprint.route('/run', methods=['GET'])
def run_capacity_study():
    """
    Run the full 7-study capacity intelligence analysis.
    Returns a JSON report to be consumed by the Policy Studio UI.
    """
    try:
        from engine.capacity_study import run_full_study

        # Prefer the MARS-corrected stand file if available
        stands_path = STANDS_FIXED_PATH if os.path.exists(STANDS_FIXED_PATH) else \
                      os.path.join(BASE_DIR, 'data', 'inputs', 'stands_final.csv')

        report = run_full_study(
            stands_path=stands_path,
            arrivals_hist_path=ARRIVALS_HIST_PATH,
            departures_hist_path=DEPARTURES_HIST_PATH,
            arrivals_fwd_path=ARRIVALS_FWD_PATH,
            airline_terminal_path=AIRLINE_TERM_PATH
        )
        return jsonify({'status': 'ok', 'report': report})

    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500

@capacity_blueprint.route('/save-policies', methods=['POST'])
def save_policies():
    try:
        data = request.get_json()
        out_path = os.path.join(BASE_DIR, 'data', 'inputs', 'selected_policies.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
