from flask import Blueprint, jsonify, request
from engine.passenger_engine import process_3day_plan, process_intraday_pulse, simulate_intraday, _load_intraday_flights
import gc

passenger_blueprint = Blueprint('passenger', __name__, url_prefix='/api/passenger')

@passenger_blueprint.route('/3day')
def get_3day_passenger_flow():
    """Returns 72 hours of aggregated passenger flow data."""
    try:
        data = process_3day_plan()
        res = jsonify({"status": "success", "data": data})
        gc.collect()
        return res
    except Exception as e:
        print(f"Error in 3-day passenger flow: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@passenger_blueprint.route('/intraday')
def get_intraday_passenger_flow():
    """Returns granular 90-min slice of the passenger waves."""
    try:
        data = process_intraday_pulse()
        res = jsonify({"status": "success", "data": data})
        gc.collect()
        return res
    except Exception as e:
        print(f"Error in intraday passenger flow: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@passenger_blueprint.route('/flights')
def get_intraday_flights():
    """Returns list of departure flights for the simulation flight selector."""
    try:
        _, departures, _ = _load_intraday_flights()
        flights = departures[['Flight_No', 'ETD', 'ICAO_Cat']].dropna(subset=['Flight_No']).copy()
        flights = flights[flights['Flight_No'].astype(str).str.strip() != ''].copy()
        result = [
            {"flight_no": str(r['Flight_No']), "etd": str(r['ETD']), "icao": str(r.get('ICAO_Cat','C'))}
            for _, r in flights.iterrows()
        ]
        res = jsonify({"status": "success", "flights": result})
        gc.collect()
        return res
    except Exception as e:
        print(f"Error fetching flights: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@passenger_blueprint.route('/simulate', methods=['POST'])
def run_simulation():
    """Run a what-if simulation scenario over the intraday pulse."""
    try:
        body     = request.get_json(force=True)
        scenario = body.get('scenario', '')
        params   = body.get('params', {})
        if not scenario:
            return jsonify({"status": "error", "message": "No scenario specified"}), 400
        data = simulate_intraday(scenario, params)
        res = jsonify({"status": "success", "data": data})
        gc.collect()
        return res
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
