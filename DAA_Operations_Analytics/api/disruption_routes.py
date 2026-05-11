"""
disruption_routes.py
Flask blueprint for disruption simulation API endpoints.
All responses are served from pre-baked static JSON files — zero computation on request.
"""
from flask import Blueprint, jsonify
import json
import os

disruption_blueprint = Blueprint('disruption', __name__, url_prefix='/api/disruption')

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "outputs", "disruption_cache")

# ── Hardcoded bulletin data ───────────────────────────────────────────────────
BULLETIN = [
    {
        "type": "notam",
        "icon": "🔴",
        "title": "NOTAM Ground Stop",
        "probability": "12%",
        "status": "Not Expected",
        "status_color": "green",
        "detail": "No active NOTAMs for next 72 hours. Scenario anchored at 10:00 Day 1 if activated.",
        "intensities": [
            {"label": "1h Stop (10:00–11:00)", "key": "notam_1h"},
            {"label": "2h Stop (10:00–12:00)", "key": "notam_2h"},
            {"label": "3h Stop (10:00–13:00)", "key": "notam_3h"},
        ]
    },
    {
        "type": "weather",
        "icon": "🌧️",
        "title": "Weather Disruption",
        "probability": "63%",
        "status": "Possible — Day 1, 16:00–18:00",
        "status_color": "yellow",
        "detail": "Met Éireann Yellow Wind Warning. Gusts 28–35kt forecast 16:00–18:00. Ground times extended across all Day 1 stands in window.",
        "intensities": [
            {"label": "Mild +15% Ground Time",     "key": "weather_15"},
            {"label": "Moderate +25% Ground Time", "key": "weather_25"},
            {"label": "Severe +40% Ground Time",   "key": "weather_40"},
        ]
    },
    {
        "type": "runway",
        "icon": "🛬",
        "title": "Reduced Runway Capacity",
        "probability": "8%",
        "status": "Not Expected",
        "status_color": "green",
        "detail": "Both runways 28L and 10R fully operational. No scheduled maintenance in 72h window.",
        "intensities": [
            {"label": "Light −25% Capacity", "key": "runway_25"},
            {"label": "Severe −50% Capacity","key": "runway_50"},
        ]
    },
    {
        "type": "schedule",
        "icon": "✈️",
        "title": "Schedule Change",
        "probability": "85%",
        "status": "Likely — Day 1",
        "status_color": "orange",
        "detail": "3 Day 1 departures with amended slots: FR2971 07:52→09:00 · EI204 08:07→10:30 · EI404 09:12→11:45.",
        "intensities": [
            {"label": "FR2971 Ryanair  07:52 → 09:00 (+68 min)", "key": "sched_FR2971"},
            {"label": "EI204 Aer Lingus 08:07 → 10:30 (+143 min)", "key": "sched_EI204"},
            {"label": "EI404 Aer Lingus 09:12 → 11:45 (+153 min)", "key": "sched_EI404"},
            {"label": "ALL 3 Flights Combined",                     "key": "sched_all"},
        ]
    },
]


@disruption_blueprint.route('/status')
def get_bulletin():
    """Return hardcoded disruption bulletin for the 3-day page top ribbon."""
    return jsonify({"status": "success", "bulletin": BULLETIN})


@disruption_blueprint.route('/scenario/<key>')
def get_scenario(key):
    """Return pre-baked disruption gantt for the given scenario key."""
    path = os.path.join(_CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": f"Scenario '{key}' not found"}), 404
    with open(path) as f:
        data = json.load(f)
    return jsonify({"status": "success", "data": data})


@disruption_blueprint.route('/scenarios')
def list_scenarios():
    """List all available pre-baked scenario keys."""
    keys = []
    if os.path.isdir(_CACHE_DIR):
        keys = [f.replace(".json", "") for f in os.listdir(_CACHE_DIR) if f.endswith(".json")]
    return jsonify({"status": "success", "scenarios": sorted(keys)})
