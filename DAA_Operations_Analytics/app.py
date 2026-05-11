from flask import Flask, render_template, jsonify, send_from_directory, request, redirect, url_for
from dotenv import load_dotenv
import os
import json

from api.routes import api_blueprint
from api.capacity_routes import capacity_blueprint
from api.intraday_routes import intraday_blueprint
from api.passenger_routes import passenger_blueprint
from api.disruption_routes import disruption_blueprint

# Load Environment Variables (e.g. LLM/News API keys)
load_dotenv()

# Robust path management
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "web", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "web", "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Register the API endpoints handling Engine data
app.register_blueprint(api_blueprint, url_prefix='/api')
app.register_blueprint(capacity_blueprint, url_prefix='/api/capacity')
app.register_blueprint(intraday_blueprint, url_prefix='/api/intraday')
app.register_blueprint(passenger_blueprint, url_prefix='/api/passenger')
app.register_blueprint(disruption_blueprint, url_prefix='/api/disruption')

@app.route('/policy-studio')
def policy_studio():
    """Stand Analysis Policy Intelligence page."""
    return render_template('policy_studio.html')

@app.route('/')
def landing_portal():
    """Redirect root to the 2026 Plan Ahead homepage."""
    return redirect(url_for('dashboard_plan_ahead_2026'))

@app.route('/3day')
def dashboard_3day():
    """3-Day Planning Horizon view."""
    return render_template('3day.html')

@app.route('/seasonal')
@app.route('/seasonal/2025')
def dashboard_seasonal():
    """2025 Seasonal Operations Report."""
    try:
        from api.data_v2 import get_v2_data
        v2_context = get_v2_data()
        return render_template('seasonal_report_2025.html', **v2_context)
    except Exception as e:
        import traceback
        return f"<h1>Internal Server Error</h1><p>{str(e)}</p><pre>{traceback.format_exc()}</pre>", 500

@app.route('/seasonal/2026')
def dashboard_forecast_2026():
    """2026 Forecast Plan report."""
    return render_template('forecast_2026.html')

@app.route('/api/forecast2026data')
def forecast2026_data():
    """Serve pre-built 2026 analytics JSON payload."""
    data_path = os.path.join(BASE_DIR, 'data', 'outputs', 'dashboard_data.json')
    with open(data_path, 'r') as f:
        return jsonify(json.load(f))

@app.route('/seasonal/near-term')
def dashboard_near_term():
    """14-Day Near-Term Intelligence View."""
    from api.near_term_data import get_near_term_data
    risk_data, _ = get_near_term_data()   # schedule fetched lazily via /api/near-term-schedule
    return render_template('near_term_intel.html', risk_data=risk_data)

@app.route('/api/near-term-schedule')
def api_near_term_schedule():
    """Serve the 3-day flight schedule as JSON for the Gantt grid."""
    from api.near_term_data import load_schedule
    from flask import jsonify
    return jsonify(load_schedule())

@app.route('/seasonal/simulate')
def dashboard_simulation_2026():
    """2026 GenAI Simulation Co-pilot view."""
    return render_template('simulation_2026.html')

@app.route('/seasonal/plan-ahead')
def dashboard_plan_ahead_2026():
    """2026 Plan Ahead: Three-Module analytical suite."""
    from api.near_term_data import get_near_term_data
    risk_data, _ = get_near_term_data()
    return render_template('plan_ahead_2026.html', risk_data=risk_data)

@app.route('/api/simulate/chat', methods=['POST'])
def simulate_chat():
    """Process natural language planning requests into structured JSON + narrative story."""
    user_input = request.json.get('message', '')
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return jsonify({
            "error": "Missing API Key",
            "message": "Please add GOOGLE_API_KEY to your .env file to enable the AI Co-pilot."
        }), 400

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')

        system_prompt = """You are the Dublin Airport (DAA) 2026 Strategic Planning Co-pilot.
Given a planning scenario, respond ONLY with a valid JSON object. No preamble, no markdown code fences.

Required schema:
{
  "params": {
    "carrier_adjustments": [{"airline": "Ryanair", "value": 1.15, "type": "percentage"}],
    "asset_availability": {"Pier B": 0.5},
    "global_modifiers": {"total_stand_capacity_delta": -5, "buffer_minutes_delta": 0},
    "cbp_volume_multiplier": 1.0,
    "active_months": ["Jun", "Jul", "Aug"],
    "timeline": "Yearly"
  },
  "story": {
    "headline": "One crisp sentence summarising the scenario impact at Dublin Airport",
    "demand_impact": "Specific impact on flight/passenger demand volumes at DAA",
    "gate_impact": "Which specific piers (A, B, D) or terminals (T1, T2) face pressure and why",
    "schedule_impact": "Which peak hours or seasons are most affected and projected congestion",
    "risk_level": "Low | Moderate | High | Critical",
    "recommendation": "One actionable mitigation step for DAA planners"
  }
}

Rules: omit any params key that is not applicable. story fields must always be present and Dublin-Airport-specific."""

        response = model.generate_content(f"{system_prompt}\n\nUser scenario: {user_input}")
        raw_text = response.text.strip()
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.split("```")[0].strip()

        payload = json.loads(raw_text)
        simulation_params = payload.get("params", {})
        story = payload.get("story", {})

        # --- DELTA ENGINE ---
        base_util = 82.4
        delta_multiplier = 1.0

        for adj in simulation_params.get("carrier_adjustments", []):
            if adj.get("type") == "percentage":
                delta_multiplier += (float(adj.get("value", 1.0)) - 1.0) * 0.3

        for asset, level in simulation_params.get("asset_availability", {}).items():
            lvl = float(level)
            if lvl < 1.0:
                delta_multiplier += (1.0 - lvl) * 0.5

        cbp_impact = float(simulation_params.get("cbp_volume_multiplier", 1.0))
        delta_multiplier += (cbp_impact - 1.0) * 0.2

        modifiers = simulation_params.get("global_modifiers", {})
        buf_min = float(modifiers.get("buffer_minutes_delta", 0))
        delta_multiplier += (buf_min / 60) * 1.5

        simulated_util = min(100.0, base_util * delta_multiplier)

        # Clear memory explicitly for Render free tier
        del raw_text
        del payload
        import gc
        gc.collect()

        return jsonify({
            "params": simulation_params,
            "story": story,
            "metrics": {
                "peak_utilization": round(float(simulated_util), 1),
                "utilization_delta": round(float(simulated_util - base_util), 1),
                "stress_level": story.get("risk_level", "High" if simulated_util > 90 else "Moderate" if simulated_util > 85 else "Low"),
                "cbp_delta": round(float((cbp_impact - 1.0) * 100), 0)
            }
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"LLM Error: {error_details}")
        return jsonify({
            "error": str(e),
            "message": "Failed to connect to LLM. Check your API key and Internet connection.",
            "traceback": error_details if app.debug else None
        }), 500


@app.route('/intraday')
def dashboard_intraday():
    """Intra-day Planning: MILP + Event Simulation Command Centre."""
    return render_template('intraday.html')

@app.route('/passenger')
def dashboard_passenger():
    """Passenger Flow Digital Twin Module."""
    return render_template('passenger.html')

@app.route('/test')
def test_plain():
    return "FLASK OK"

# Pre-load data into memory for instant report loading on Render
try:
    print("Pre-loading data for Render optimization...")
    from api.data_v2 import get_v2_data
    get_v2_data()
    
    print("Warm-up: Generating Passenger Flow Cache...")
    from engine.passenger_engine import process_3day_plan
    process_3day_plan()
except Exception as e:
    print(f"Pre-load failed: {e}")

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5010)
