# DAA Passenger Planning Suite — Architecture Reference Prompt

## Purpose
This document describes the **exact architecture and data flow** for two self-contained passenger planning views from the DAA Strategic Planning Suite (Flask/Python + Chart.js). Use this as a complete reference to reproduce or integrate these views into another solution.

---

## Quick-Reference: All Files at a Glance

| Category | Exact Files |
|---|---|
| **HTML Templates** | `web/templates/plan_ahead_2026.html` (Strategic 2026 Long-Term)<br>`web/templates/passenger.html` (Tactical Pulse — 3-Day & Intra-Day) |
| **Python Engines** | `engine/passenger_engine.py` (Short-Term wave engine + simulation)<br>`engine/passenger_ml_engine.py` (Long-Term ML ensemble models)<br>`engine/ingestion.py` (Data loader utilities) |
| **API Blueprints** | `api/passenger_routes.py` (Exposes `/3day`, `/intraday`, `/simulate`, `/flights`) |
| **Input Data (CSV)** | `data/inputs/flights_arrivals_next3days_match.csv`<br>`data/inputs/flights_departures_next3days.csv`<br>`data/inputs/intraday_flights.csv`<br>`data/outputs/forecast_2026.csv` (Load Factor seasonal index) |
| **Pre-Calculated Data** | `data/outputs/pax_3day_cache.json` (3-day simulation cache)<br>`web/static/js/_data_arrays.js` (Historical Pax baselines: H2023–H2025)<br>`web/static/js/_pax_ml_results.js` (ML P10 / P50 / P90 results) |

---

## The Two Views

| View | URL Route | Template File |
|---|---|---|
| **2026 Long-Term Passenger Forecast** | `/seasonal/plan-ahead` | `web/templates/plan_ahead_2026.html` |
| **Short-Term Passenger Plan** | `/passenger` | `web/templates/passenger.html` |

---

## View 1: 2026 Long-Term Passenger Forecast

### What It Does
A dual-mode strategic dashboard (Flight Movements / Passenger Flow). The **Passenger Flow side** shows a 52-week ML ensemble forecast for 2026 with confidence bands, a demand heatmap, weekly pax trend by 3-hour block, a terminal processor stress heatmap, and a carrier market dominance analysis.

### How to Activate the Passenger View
The dashboard defaults to **Flight Movements** mode. The Passenger view is switched by calling:
```javascript
window.setMasterMetric('pax');  // in plan_ahead_lt.js
```
The UI toggle button is `id="btn-metric-pax"`. The global state lives in `window.MASTER_METRIC`.

### Sections Visible ONLY in Passenger Mode (`plan_ahead_2026.html`)
| Section ID | Description |
|---|---|
| `#decomp-panel-pax` | Weekly Pax Trend chart (3-hour block intensity, Mon–Sun) |
| `#sec-carrier-dominance` | Carrier Market Share Donut + Absolute Growth Leaderboard |
| `#processor-heatmap-card` | T1/T2 Security & Baggage RAG stress heatmap |

### Sections HIDDEN in Passenger Mode
`#sec-historical-overview`, `#sec-risk-index`, `#sec-carrier-growth`, `#sec-strategic-intelligence`, `#decomp-panel-mov`

The JS visibility engine is in `plan_ahead_lt.js`, lines 79–88:
```javascript
const hiddenInPax = ['sec-historical-overview','sec-risk-index','sec-carrier-growth','sec-strategic-intelligence'];
const hiddenInMov = ['sec-strategic-intelligence','sec-pier-rag'];
[...hiddenInPax,...hiddenInMov].forEach(id => {
    el.classList.toggle('hidden-sec', shouldHide);
});
```

### Data Flow — Long-Term Passenger View

```
STEP 1 — ENGINE RUN (offline, run once)
  engine/passenger_ml_engine.py
    INPUT:  data/historical_weekly_movements_2021_2025.csv
              Columns: year, week, flight_mov
    PROCESS:
      - Derives pax from movements using seasonal ratio (~125 pax/mov ± 15 sine wave)
      - Trains 4 models on 2023–2024, validates on 2025:
          SARIMA (1,1,1), MLP/LSTM proxy, Prophet (seasonal_decompose), Monte Carlo (1000 paths)
      - Ensemble = inverse-MAPE-weighted combination of all 4
      - Forecasts 2026 anchored on 2025 actuals × growth multipliers:
          SARIMA ×1.06, MLP ×1.065, Prophet ×1.05, MC ×1.055 (with ±2% noise)
      - Produces P10 = P50×0.94, P50 = Ensemble, P90 = P50×1.08
    OUTPUT: web/static/js/_pax_ml_results.js
              window.PAX_ML_RESULTS = { accuracies, p10_pax[52], p50_pax[52], p90_pax[52] }

STEP 2 — STATIC DATA ARRAYS (pre-built, loaded at page load)
  web/static/js/_data_arrays.js
    Contains: H2023_PAX[52], H2024_PAX[52], H2025_PAX[52]  ← historical weekly pax
              P10[52], P50[52], P90[52]                      ← flight movement forecasts
              KPI_5YR object                                   ← total26_pax, avg_lf, cagr, etc.
              T1_PEAK_HR[52], T2_PEAK_HR[52]                 ← hourly pax load for processor heatmap

STEP 3 — FRONTEND RENDERING (plan_ahead_lt.js)
  On page load → DOMContentLoaded → setMasterMetric('mov', true) → renderLT()
  On PAX toggle → setMasterMetric('pax') → renderLT()
  
  renderLT() draws:
    - 52-week main chart:   P10/P50/P90 from PAX_ML_RESULTS + H2025_PAX as baseline
    - Quarterly bar chart:  H2025_PAX vs P50 aggregated to Q1–Q4
    - Demand heatmap:       P50_PAX coloured by intensity (blue cold → red hot)
    - KPI strip:            total P50 pax, P10/P90 range, avg load factor from KPI_5YR
  
  renderWeeklyPaxTrend()  → '#weeklyPaxTrendChart' — hardcoded daily 3-hr patterns
  renderCarrierDominance() → '#carrierShareDonut' + '#carrierAbsGrowthBar'
                             Uses AIRLINES[], AL_TOTAL_2025[], AL_GROWTH_PCT[] (hardcoded in JS)
  Processor Heatmap:       T1_PEAK_HR / T2_PEAK_HR vs hardcoded capacity thresholds
```

### No Live API Call for This View
The Long-Term view is **fully static** — it loads 2 JS files at page boot and needs no backend API call at runtime.

---

## View 2: Short-Term Passenger Plan (`passenger.html`)

### What It Does
Two sub-panels:
- **3-Day Outlook** — 72-hour passenger flow across 7 touchpoints in 15-min bins
- **Intra-Day Pulse** — 90-minute high-resolution wave for the current morning peak (07:00–08:30)

### Data Flow — Short-Term / 3-Day

```
INPUT FILES (CSV)
  data/inputs/flights_arrivals_next3days_match.csv
    Columns: date (DD-Mon-YY), sta (HH:MM), ICAO_Cat, aircraft_type, ...

  data/inputs/flights_departures_next3days.csv
    Columns: date (DD-Mon-YY), std (HH:MM), ICAO_Cat, aircraft_type, CBP_Required, destination_code, ...

ENGINE: engine/passenger_engine.py → process_3day_plan()
  - Parses datetimes, filters 05:00 Day1 → midnight Day3
  - For each departure flight:
      pax = seat_capacity(aircraft_type, ICAO_Cat) × load_factor(date)
        load_factor pulled from data/outputs/forecast_2026.csv [Week_In_Year → seasonal_index]
      Generates 7 Gamma/uniform touchpoint waves (per-minute resolution):
        Check-In:   Gamma(-180m to -60m pre-ETD, lead=180m intl, 120m domestic)
        Security:   Check-In shifted +15 min
        CBP:        Security shifted +30 min (US destinations only)
        Boarding:   Uniform pulse -40m to -15m pre-ETD
        Lounge:     Cumulative (Security inflow − Boarding outflow)
  - For each arrival flight:
        Immigration: Uniform +15m to +75m post-ETA (intl only)
        Baggage:     Gamma +25m to +80m post-ETA (75% of pax)
  - Resamples from 1-min to 15-min bins
  - Caches output to data/outputs/pax_3day_cache.json

API ENDPOINT
  GET /api/passenger/3day  (api/passenger_routes.py)
  
  RESPONSE JSON:
  {
    "status": "success",
    "data": {
      "labels": ["05:00", "05:15", ...],         // 15-min HH:MM ticks
      "day_labels": ["Mon 11 May", ...],          // parallel day strings
      "checkin":    [float, ...],                 // 15-min pax counts
      "security":   [float, ...],
      "cbp":        [float, ...],
      "lounge":     [float, ...],                 // mean concurrent occupancy
      "boarding":   [float, ...],
      "immigration":[float, ...],
      "baggage":    [float, ...],
      "days_data":  [                             // 3 items, one per day
        { "date": "Mon 11 May 2026",
          "labels": ["05:00",...],
          "checkin": [int,...], ... }
      ]
    }
  }

FRONTEND (passenger.html → render3DayChart())
  - Chart.js multi-line area chart with 7 datasets
  - Custom plugin draws date labels below x-axis at noon of each day
  - KPI strip: peak pax/15-min per touchpoint + peak time window
  - Day matrix: 3 per-day tables at 30-min resolution
```

### Data Flow — Intra-Day Pulse

```
INPUT FILE
  data/inputs/intraday_flights.csv
    Columns: Type (DEP/ARR/TURN), Flight_No, ETD, ETA, ICAO_Cat, aircraft_type, CBP_Required

ENGINE: engine/passenger_engine.py → process_intraday_pulse()
  - Uses hardcoded sim_date = '31-Mar-26'
  - Filters window: 04:00 to 09:30 (view), waves computed from 03:00
  - Same Gamma wave kernels as 3-Day, freq_min=1 (minute resolution)
  - No cache — computed on each request

API ENDPOINT
  GET /api/passenger/intraday
  
  RESPONSE JSON:
  {
    "status": "success",
    "data": {
      "labels": ["04:00","04:01",...],  // 1-min resolution
      "checkin":    [float,...],
      "security":   [float,...],
      "cbp":        [float,...],
      "lounge":     [float,...],
      "boarding":   [float,...],
      "immigration":[float,...],
      "baggage":    [float,...],
      "table_labels": ["04:30","04:45",...],  // 15-min bins
      "table_data": { "checkin":[int,...], ... }
    }
  }

FRONTEND (passenger.html → renderIntraChart())
  - Chart.js multi-line chart at 1-min resolution
  - Canvas destroyed and recreated on each render to prevent "Canvas already in use" error
  - Dual Y-axis: left=Pax/min, right=Lounge Concurrent Occupancy
  - Side table: 15-min bin summary
```

### What-If Simulation Engine

```
POST /api/passenger/simulate
  Body: { "scenario": "flight_delay"|"ground_stop"|"security_reduction"|"checkin_reduction",
          "params": { ... } }

Scenarios:
  flight_delay:       Shifts specific flight datetime +N minutes
  ground_stop:        Freezes all flights in window, bursts on lift
  security_reduction: Applies throughput cap via _apply_throughput_cap() 
                      (models queue buildup and drain)
  checkin_reduction:  Applies cap to check-in wave

RESPONSE JSON:
  { "baseline": { checkin:[...], ... },
    "simulated": { checkin:[...], ... },
    "impact": { checkin: { baseline_peak, simulated_peak, delta, delta_pct }, ... },
    "cascade": ["narrative string",...],
    "at_risk_flights": [{ flight, type, detail },...],
    "table_labels": [...], "table_data": {...} }

Helper endpoints:
  GET /api/passenger/flights  → list of departures for the flight selector dropdown
```

---

## Complete File Manifest

```
engine/
  passenger_ml_engine.py     ← ML ensemble, run offline, writes _pax_ml_results.js
  passenger_engine.py        ← Short-term wave engine + simulation
  ingestion.py               ← CSV load utilities

api/
  passenger_routes.py        ← Flask Blueprint: /api/passenger/3day, /intraday, /simulate, /flights

data/
  historical_weekly_movements_2021_2025.csv   ← ML training input
  inputs/flights_arrivals_next3days_match.csv ← 3-day arrivals
  inputs/flights_departures_next3days.csv     ← 3-day departures
  inputs/intraday_flights.csv                 ← intraday schedule
  outputs/forecast_2026.csv                   ← seasonal_index by Week_In_Year (load factor source)
  outputs/pax_3day_cache.json                 ← 3-day result cache

web/
  static/js/
    _data_arrays.js       ← H2023_PAX, H2024_PAX, H2025_PAX, T1_PEAK_HR, T2_PEAK_HR, KPI_5YR
    _pax_ml_results.js    ← window.PAX_ML_RESULTS (generated by passenger_ml_engine.py)
    plan_ahead_lt.js      ← All rendering logic for plan_ahead_2026.html
  templates/
    plan_ahead_2026.html  ← Long-term dual-mode dashboard
    passenger.html        ← Short-term 3-day + intraday pulse dashboard
```

---

## Key Integration Notes

1. **Long-Term view needs no live API** — serve the 2 static JS files and the HTML. Run `passenger_ml_engine.py` once to regenerate `_pax_ml_results.js` whenever the training data changes.
2. **Short-Term view needs the Flask API** — the 3-day view checks the cache file first; delete `pax_3day_cache.json` to force recomputation when input CSVs change.
3. **Load Factor dependency** — `passenger_engine.py` reads `forecast_2026.csv` to compute seasonal load factors. If this file is missing it falls back to `DEFAULT_LOAD_FACTOR = 0.85`.
4. **Aircraft Seat Capacity** — hardcoded in `passenger_engine.py` in the `SEAT_CAPACITY` dict and `ICAO_FALLBACK` dict. No external lookup required.
5. **Chart.js version** — loaded from CDN (`cdn.jsdelivr.net/npm/chart.js`). No specific version pinned — use latest stable.
