from flask import Blueprint, jsonify, request
import os
import json
import pandas as pd
import numpy as np
from engine.ingestion import load_flights, load_airline_terminals, load_stands
from engine.rotation import build_rotations
from engine.config_loader import get_sys_config
from engine.allocator import generate_allocation, OperatorPreferences

api_blueprint = Blueprint('api', __name__)

CONFIG_PATH = r'data\inputs\config.json'
STANDS_PATH = r'data\inputs\stands_final_fixed.csv'
AIRLINE_PATH = r'data\inputs\airline_terminal.csv'
ARRIVALS_PATH = r'data\inputs\flights_arrivals.csv'
DEPARTURES_PATH = r'data\inputs\flights_departures.csv'
SEASONAL_ARR = r'data\inputs\DAA_Weekly_Flight_Demand_Arrivals_2025.csv'
SEASONAL_DEP = r'data\inputs\DAA_Weekly_Flight_Demand_Departures_2025.csv'
SEASONAL_UTIL = r'data\inputs\DAA_Weekly_Stand_Utilisation_2025.csv'

# --- Growth category maps ---
LCC_AIRLINES      = ['FR', 'W6', 'TO', 'U2', 'VY', 'DY', 'D8']
TRANSATLANTIC     = ['AA', 'DL', 'UA', 'AC', 'EI', 'BA']
LONGHAUL          = ['EK', 'QR', 'LH', 'TK', 'ET', 'EY']

AIRCRAFT_TO_ICAO = {
    'B737': 'C', 'A320': 'C', 'A319': 'C', 'A321': 'C', 'B738': 'C',
    'A330': 'E', 'A350': 'E', 'B787': 'E', 'B777': 'E', 'B789': 'E',
    'ATR': 'B', 'E190': 'C', 'E195': 'C', 'DH4': 'B'
}

def _get_icao_cat(family):
    for k, v in AIRCRAFT_TO_ICAO.items():
        if k.lower() in str(family).lower():
            return v
    return 'C'


@api_blueprint.route('/seasonal/forecast', methods=['GET', 'POST'])
def get_seasonal_forecast():
    """
    Returns the full forecast dataset for the Seasonal UI.
    POST body (optional): { lcc_growth: 0.08, transatlantic_growth: 0.04, charter_growth: 0.0, acl_cap: 3000 }
    """
    body = request.get_json(silent=True) or {}
    lcc_g   = float(body.get('lcc_growth', 0.08))
    trans_g = float(body.get('transatlantic_growth', 0.04))
    lh_g    = float(body.get('longhaul_growth', 0.02))
    other_g = float(body.get('other_growth', 0.03))
    acl_cap = int(body.get('acl_cap', 3000))

    try:
        # ── 1. Load raw 2025 data ──────────────────────────────────────────
        df_arr  = pd.read_csv(SEASONAL_ARR)
        df_dep  = pd.read_csv(SEASONAL_DEP)
        df_util = pd.read_csv(SEASONAL_UTIL)

        df_arr['Week_Start']  = pd.to_datetime(df_arr['Week_Start'])
        df_dep['Week_Start']  = pd.to_datetime(df_dep['Week_Start'])
        df_util['Week_Start'] = pd.to_datetime(df_util['Week_Start'])

        # ── 2. Apply growth multipliers per airline segment ────────────────
        def get_multiplier(airline_code):
            if str(airline_code) in LCC_AIRLINES:      return 1 + lcc_g
            if str(airline_code) in TRANSATLANTIC:     return 1 + trans_g
            if str(airline_code) in LONGHAUL:         return 1 + lh_g
            return 1 + other_g

        df_arr['multiplier']  = df_arr['Airline'].apply(get_multiplier)
        df_arr['Forecast_Movements'] = df_arr['Weekly_Movements'].astype(float) * df_arr['multiplier']
        df_arr['icao_cat'] = df_arr['Aircraft_Family'].apply(_get_icao_cat)

        df_dep['multiplier'] = df_dep['Airline'].apply(get_multiplier)
        df_dep['Forecast_Movements'] = df_dep['Weekly_Movements'].astype(float) * df_dep['multiplier']

        # ── 3. ZONE B: Main Demand Forecast (weekly totals) ────────────────
        actual_by_week = (
            df_arr.groupby('Week_Start')['Weekly_Movements'].sum()
            + df_dep.groupby('Week_Start')['Weekly_Movements'].sum()
        ).reset_index()
        actual_by_week.columns = ['week', 'actual']

        forecast_by_week = (
            df_arr.groupby('Week_Start')['Forecast_Movements'].sum()
            + df_dep.groupby('Week_Start')['Forecast_Movements'].sum()
        ).reset_index()
        forecast_by_week.columns = ['week', 'forecast']

        merged = actual_by_week.merge(forecast_by_week, on='week').sort_values('week')
        merged['week_label'] = merged['week'].dt.strftime('W%V %b')

        zone_b = {
            'weeks':    merged['week_label'].tolist(),
            'actual':   merged['actual'].round(0).tolist(),
            'forecast': merged['forecast'].round(0).tolist(),
            'acl_cap':  acl_cap
        }

        # ── 4. ZONE D: Stand × Week Heatmap ───────────────────────────────
        df_util = df_util.dropna(subset=['Week_Start', 'Pier'])

        # Apply segment growth to utilisation by mapping pier to traffic type heuristic
        df_util['occupancy_pct'] = (
            df_util['Avg_Daily_Occupancy_Hrs'].astype(float) /
            (df_util['Avg_Daily_Occupancy_Hrs'].astype(float) + df_util['Headroom_Hrs'].astype(float).clip(lower=0.01))
        ) * 100

        # Apply blanket forecast multiplier to utilisation
        df_util['forecast_pct'] = (df_util['occupancy_pct'] * (1 + other_g)).clip(upper=110)

        # Group by Pier + Week (for heatmap rows = pier clusters)
        util_pivot = df_util.groupby(['Week_Start', 'Pier'])['forecast_pct'].mean().reset_index()
        piers = sorted(df_util['Pier'].dropna().unique().tolist())
        weeks_sorted = sorted(df_util['Week_Start'].dt.strftime('W%V %b').unique().tolist())

        heatmap_z = []
        for pier in piers:
            pier_data = util_pivot[util_pivot['Pier'] == pier].sort_values('Week_Start')
            heatmap_z.append(pier_data['forecast_pct'].round(1).tolist())

        zone_d = {
            'x': weeks_sorted,
            'y': piers,
            'z': heatmap_z
        }

        # ── 5. ZONE C: Auto-generated Insights ────────────────────────────
        insights = []

        # Breach alerts: weeks where forecast exceeds ACL cap
        breach_weeks = merged[merged['forecast'] > acl_cap]
        if not breach_weeks.empty:
            peak = breach_weeks.iloc[breach_weeks['forecast'].values.argmax()]
            peak_week = peak['week_label']
            peak_val  = int(peak['forecast'])
            # Find top driver that week
            top_airline = df_arr[df_arr['Week_Start'] == peak['week']].groupby('Airline')['Forecast_Movements'].sum().idxmax()
            insights.append({
                'type': 'BREACH',
                'title': f'ACL Cap Breached — {len(breach_weeks)} Weeks',
                'body': f'Peak at {peak_week}: {peak_val:,} movements vs cap of {acl_cap:,}. Primary driver: {top_airline}.',
                'week': peak_week
            })

        # Opportunity: weeks with high headroom
        low_util = util_pivot.groupby('Week_Start')['forecast_pct'].mean()
        easy_weeks = low_util[low_util < 45].sort_values()
        if not easy_weeks.empty:
            easy_win  = easy_weeks.index[0].strftime('W%V %b')
            easy_pct  = round(easy_weeks.iloc[0], 1)
            insights.append({
                'type': 'OPPORTUNITY',
                'title': 'Route Development Window Identified',
                'body': f'{easy_win} shows avg stand utilisation of only {easy_pct}%. Capacity available for up to 3 additional route allocations.',
                'week': easy_win
            })

        # Maintenance window
        maint_week = low_util.idxmin().strftime('W%V %b')
        insights.append({
            'type': 'MAINTENANCE',
            'title': f'Safest Maintenance Window: {maint_week}',
            'body': f'Lowest combined utilisation across all piers. Recommended window for stand closures or resurfacing work.',
            'week': maint_week
        })

        # 32M Pax cap
        # Estimate: assume avg 185 seats, Avg_Load_Factor from demand file
        df_arr['est_pax'] = (
            df_arr['Forecast_Movements'].astype(float) * 185 *
            df_arr['Avg_Load_Factor_Pct'].str.replace('%', '').astype(float) / 100
        )
        cumulative_pax = df_arr.groupby('Week_Start')['est_pax'].sum().cumsum()
        breach_mask = cumulative_pax > 32_000_000
        if breach_mask.any():
            breach_at = breach_mask.idxmax().strftime('W%V %b')
            insights.append({
                'type': 'CAP_BREACH',
                'title': f'32M Pax Cap Breached at {breach_at}',
                'body': f'At current growth trajectory, T1+T2 cumulative passengers cross the 32.0M government-imposed annual cap.',
                'week': breach_at
            })
        else:
            final_pax = int(cumulative_pax.iloc[-1] / 1_000_000 * 10) / 10
            insights.append({
                'type': 'COMPLIANT',
                'title': '32M Cap: Compliant',
                'body': f'Forecast reaches {final_pax}M passengers, within the 32.0M annual limit.',
                'week': None
            })

        # ── 6. ZONE E: KPI Scalars ─────────────────────────────────────────
        avg_util = float(df_util['forecast_pct'].mean())
        cat_e_demand = int(df_arr[df_arr['icao_cat'] == 'E']['Forecast_Movements'].sum())
        mars_capacity_weekly = 11 * 52  # 11 MARS centres, one per week
        final_pax_m = round(cumulative_pax.iloc[-1] / 1_000_000, 1)

        zone_e = {
            'avg_stand_util_pct': round(avg_util, 1),
            'cat_e_total_forecast': cat_e_demand,
            'mars_capacity_total': mars_capacity_weekly,
            'mars_pressure_pct': round(cat_e_demand / mars_capacity_weekly * 100, 1),
            'forecast_pax_m': final_pax_m,
            'pax_cap_m': 32.0
        }

        return jsonify({
            'status': 'success',
            'zone_b': zone_b,
            'zone_c': insights,
            'zone_d': zone_d,
            'zone_e': zone_e
        })

    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@api_blueprint.route('/plan/predictive', methods=['GET', 'POST'])
def generate_predictive_3day_plan():
    CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'outputs', '3day_plan_cache.json')
    
    # On GET, try to load from cache
    if request.method == 'GET':
        if os.path.exists(CACHE_FILE):
            try:
                if os.path.getsize(CACHE_FILE) > 0:
                    with open(CACHE_FILE, 'r') as f:
                        data = json.load(f)
                        if data.get('status') == 'success' and 'gantt' in data:
                            return jsonify(data)
            except Exception as e:
                print(f"Cache read error: {e}")
                pass
    
    prefs = OperatorPreferences()
    config_overrides = {}
    closed_stands = []

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}

        # Soft weight overrides
        prefs.w_contact             = float(data.get('w_contact', prefs.w_contact))
        prefs.w_seasonal_preference = float(data.get('w_seasonal_preference', prefs.w_seasonal_preference))
        prefs.w_pier_match          = float(data.get('w_pier_match', prefs.w_pier_match))
        prefs.w_long_stay_penalty   = float(data.get('w_long_stay_penalty', prefs.w_long_stay_penalty))

        # Hard config overrides from UI simulation panel
        if 'stand_buffer_min' in data:
            config_overrides['stand_buffer_min'] = int(data['stand_buffer_min'])
        if 'long_stay_threshold_min' in data:
            config_overrides['long_stay_threshold_min'] = int(data['long_stay_threshold_min'])
        if 'tow_threshold_min' in data:
            config_overrides['tow_threshold_min'] = int(data['tow_threshold_min'])

        # Closed stand list from UI
        closed_stands = data.get('closed_stands', [])

    ARR_MATCH_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'inputs', 'flights_arrivals_next3days_match.csv')
    DEP_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'inputs', 'flights_departures_next3days.csv')

    from engine.predictive_engine import run_predictive_engine
    blocks, decisions = run_predictive_engine(
        CONFIG_PATH, STANDS_PATH, AIRLINE_PATH,
        ARR_MATCH_PATH, DEP_PATH,
        prefs,
        config_overrides=config_overrides,
        closed_stands=closed_stands
    )

    gantt_data = []
    for b in blocks:
        if b.assigned_stand and b.assigned_stand != 'UNRESOLVED':
            arr_f = b.arrival_flight
            dep_f = b.departure_flight

            arr_id = arr_f.flight_no if arr_f else ''
            dep_id = dep_f.flight_no if dep_f else ''
            airline = arr_f.airline_name if arr_f else (dep_f.airline_name if dep_f else 'UNK')

            import datetime
            start_str = b.occupancy_start.strftime("%Y-%m-%d %H:%M:%S")
            end_str = b.occupancy_end.strftime("%Y-%m-%d %H:%M:%S")

            rtype = 'ARR'
            if arr_f and dep_f:
                rtype = 'TURN'
            elif dep_f and not arr_f:
                rtype = 'DEP'

            gantt_data.append({
                'gate': b.assigned_stand,
                'start_time': start_str,
                'end_time': end_str,
                'arr_flight': arr_id,
                'dep_flight': dep_id,
                'airline': airline,
                'tail_reg': b.tail_reg,
                'terminal': b.terminal,
                'icao_cat': b.icao_cat,
                'aircraft_type': arr_f.aircraft_type if arr_f else (dep_f.aircraft_type if dep_f else 'UNK'),
                'origin': arr_f.airport_code if arr_f else '',
                'destination': dep_f.airport_code if dep_f else '',
                'cbp_flag': b.cbp_required,
                'type': rtype
            })

    # Build all_stands list for UI vacant-gate rendering
    from engine.ingestion import load_stands as _load_stands
    all_stands_raw = _load_stands(STANDS_PATH)
    all_stands_list = [
        {'stand_id': s.stand_id, 'pier': s.pier, 'terminal': s.terminal,
         'icao_cat_max': s.icao_cat_max, 'has_airbridge': s.has_airbridge,
         'cbp_eligible': s.cbp_eligible, 'stand_type': s.stand_type}
        for s in all_stands_raw.values()
    ]

    # Build KPI data from gantt_data
    WIDEBODY_CATS = {'E', 'F'}
    PIER_MAP = {
        '1': 'Pier 1', '2': 'Pier 2', '3': 'Pier 3', '4': 'Pier 4',
    }
    def _gate_to_pier(gate):
        """Map gate ID to pier label for KPI summary."""
        try:
            n = int(str(gate).rstrip('CLRTlcrt'))
            if 100 <= n <= 199: return 'Pier 1'
            if 200 <= n <= 249: return 'Pier 2'
            if 300 <= n <= 399: return 'Pier 3'
            if 400 <= n <= 499: return 'Pier 4'
            if 500 <= n <= 599: return 'Pier E'
            return 'Apron'
        except:
            return 'Apron'

    by_pier = {}
    nb_count = 0
    wb_count = 0
    cbp_count = 0
    for g in gantt_data:
        pier_lbl = _gate_to_pier(g['gate'])
        by_pier[pier_lbl] = by_pier.get(pier_lbl, 0) + 1
        if g.get('icao_cat') in WIDEBODY_CATS:
            wb_count += 1
        else:
            nb_count += 1
        if g.get('cbp_flag'):
            cbp_count += 1

    top_pier = max(by_pier, key=by_pier.get) if by_pier else 'N/A'

    kpi = {
        'total_flights': len(gantt_data),
        'by_pier': by_pier,
        'top_pier': top_pier,
        'narrowbody': nb_count,
        'widebody': wb_count,
        'cbp': cbp_count
    }

    unresolved = len([d for d in decisions if d.assigned_stand == 'UNRESOLVED'])
    # Save to cache before returning
    res_payload = {
        'status': 'success',
        'gantt': gantt_data,
        'all_stands': all_stands_list,
        'kpi': kpi,
        'metrics': {'total_blocks': len(blocks), 'unresolved': unresolved, 'resolved': len(blocks) - unresolved}
    }
    
    response = jsonify(res_payload)
    
    try:
        with open(CACHE_FILE, 'wb') as f:
            f.write(response.get_data())
    except Exception as e:
        print(f"Error caching plan: {e}")

    return response

@api_blueprint.route('/seasonal/baseline_2025', methods=['GET'])
def get_seasonal_baseline():
    """
    Returns the deep forensic diagnostics for the 2025 Baseline Report (Tab 1).
    """
    try:
        # Load 2025 actuals
        df_arr  = pd.read_csv(SEASONAL_ARR)
        df_dep  = pd.read_csv(SEASONAL_DEP)
        df_util = pd.read_csv(SEASONAL_UTIL)

        df_arr['Week_Start']  = pd.to_datetime(df_arr['Week_Start'])
        df_dep['Week_Start']  = pd.to_datetime(df_dep['Week_Start'])
        df_util['Week_Start'] = pd.to_datetime(df_util['Week_Start'])

        # ── 1. Top-Level KPIs ─────────────────────────────────
        total_arr = int(df_arr['Weekly_Movements'].sum())
        total_dep = int(df_dep['Weekly_Movements'].sum())
        total_mov = total_arr + total_dep

        # ── 2. Insights ───────────────────────────────────────
        df_util['Month'] = df_util['Week_Start'].dt.month_name()
        busiest_month = df_util.groupby('Month')['Weekly_Movements'].sum().idxmax()
        
        busiest_pier = df_util.groupby('Pier')['Weekly_Movements'].sum().idxmax()
        busiest_pier_util = df_util[df_util['Pier'] == busiest_pier]['Avg_Daily_Occupancy_Hrs'].mean()
        
        # Max peak week absolute movements
        weekly_totals = df_arr.groupby('Week_Start')['Weekly_Movements'].sum() + df_dep.groupby('Week_Start')['Weekly_Movements'].sum()
        peak_week_date = weekly_totals.idxmax()
        peak_week_movs = int(weekly_totals.max())

        # ── 3. Fleet & Airline Mix ───────────────────────────
        df_combined = pd.concat([df_arr, df_dep])
        fleet_mix = df_combined.groupby('Aircraft_Family')['Weekly_Movements'].sum().sort_values(ascending=False).head(5)
        airline_mix = df_combined.groupby('Airline_Name')['Weekly_Movements'].sum().sort_values(ascending=False).head(5)

        fleet_data = [{"family": k, "count": int(v)} for k, v in fleet_mix.items()]
        airline_data = [{"airline": k, "count": int(v)} for k, v in airline_mix.items()]

        # ── 4. The 52-Week Demand Curve ──────────────────────
        # Sort chronologically
        weekly_totals = weekly_totals.sort_index()
        demand_curve = {
            "x": [d.strftime('%Y-%m-%d') for d in weekly_totals.index],
            "y": [int(v) for v in weekly_totals.values]
        }

        # ── 5. The Massive Stand Utilisation Heatmap ─────────
        # Pivot: Rows = Gate_ID, Cols = Week_Start, Values = Avg_Daily_Occupancy_Hrs
        # We sort by Pier first to group stands logically on the Y axis
        df_util_sorted = df_util.sort_values(by=['Pier', 'Gate_ID'])
        heatmap_pivot = df_util_sorted.pivot_table(
            index='Gate_ID', 
            columns='Week_Start', 
            values='Avg_Daily_Occupancy_Hrs', 
            aggfunc='mean'
        )
        
        # We need the y axis labels (stands) ordered nicely
        y_labels = [str(g) for g in heatmap_pivot.index]
        x_labels = [d.strftime('%Y-%m-%d') for d in heatmap_pivot.columns]
        z_data = np.nan_to_num(heatmap_pivot.values).tolist()

        heatmap = {
            "x": x_labels,
            "y": y_labels,
            "z": z_data
        }

        # ── 6. Construct Insights Payload ────────────────────
        insights = [
            {
                "type": "TOTAL_VOLUME",
                "title": "2025 Annual Baseline Recorded",
                "body": f"The airport processed {total_mov:,} total movements. Arrivals ({total_arr:,}) perfectly mirrored Departures ({total_dep:,})."
            },
            {
                "type": "PEAK_PRESSURE",
                "title": f"Busiest Period: {busiest_month}",
                "body": f"Week of {peak_week_date.strftime('%b %d')} recorded the highest throughput with {peak_week_movs:,} movements in a 7-day window."
            },
            {
                "type": "PIER_BOTTLENECK",
                "title": f"Most Saturated Infrastructure: {busiest_pier}",
                "body": f"{busiest_pier} experienced an average daily occupancy of {busiest_pier_util:.1f} hours per stand throughout the year."
            }
        ]

        return jsonify({
            "kpis": {
                "total_movements": total_mov,
                "total_arrivals": total_arr,
                "total_departures": total_dep,
                "busiest_pier": busiest_pier
            },
            "fleet_mix": fleet_data,
            "airline_mix": airline_data,
            "demand_curve": demand_curve,
            "heatmap": heatmap,
            "insights": insights
        })

    except Exception as e:
        print("Error serving 2025 baseline:", str(e))
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/seasonal/story_2025', methods=['GET'])
def get_seasonal_story_2025():
    """
    Returns the four-section horizontal story data for the 2025 Baseline.
    Section 1: Overall Demand Trend (Monthly + Weekly)
    Section 2: Operational Breakdown (Airlines, Fleet, Category)
    Section 3: Infrastructure Pressure (Gate & Stand)
    Section 4: Terminal Comparison (T1 vs T2)
    """
    try:
        df_arr = pd.read_csv(SEASONAL_ARR)
        df_dep = pd.read_csv(SEASONAL_DEP)
        df_util = pd.read_csv(SEASONAL_UTIL)

        df_arr['Week_Start'] = pd.to_datetime(df_arr['Week_Start'])
        df_dep['Week_Start'] = pd.to_datetime(df_dep['Week_Start'])
        df_util['Week_Start'] = pd.to_datetime(df_util['Week_Start'])

        # ── SECTION 1: Demand Story ──────────────────────────
        df_combined = pd.concat([df_arr, df_dep])
        df_combined['Month'] = df_combined['Week_Start'].dt.strftime('%b')
        df_combined['Month_Num'] = df_combined['Week_Start'].dt.month

        monthly = df_combined.groupby(['Month_Num', 'Month'])['Weekly_Movements'].sum().reset_index().sort_values('Month_Num')
        weekly = df_combined.groupby('Week_Start')['Weekly_Movements'].sum().reset_index().sort_values('Week_Start')

        demand_story = {
            'monthly_labels': monthly['Month'].tolist(),
            'monthly_values': monthly['Weekly_Movements'].astype(int).tolist(),
            'weekly_labels': [w.strftime('%V') for w in weekly['Week_Start']], # Week numbers
            'weekly_values': weekly['Weekly_Movements'].astype(int).tolist()
        }

        # ── SECTION 2: Operational Story ──────────────────────
        airline_share = df_combined.groupby('Airline_Name')['Weekly_Movements'].sum().sort_values(ascending=False).head(10)
        fleet_mix = df_combined.groupby('Aircraft_Family')['Weekly_Movements'].sum().sort_values(ascending=False).head(8)
        category_mix = df_combined.groupby('Flight_Category')['Weekly_Movements'].sum().sort_values(ascending=False)
        
        operational_story = {
            'airlines': [{'name': k, 'value': int(v)} for k, v in airline_share.items()],
            'fleet': [{'family': k, 'value': int(v)} for k, v in fleet_mix.items()],
            'categories': [{'name': k, 'value': int(v)} for k, v in category_mix.items()],
            'traffic_mix': {
                'arrivals': int(df_arr['Weekly_Movements'].sum()),
                'departures': int(df_dep['Weekly_Movements'].sum())
            }
        }

        # ── SECTION 3: Infrastructure Story ───────────────────
        pier_pressure = df_util.groupby('Pier')['Avg_Daily_Occupancy_Hrs'].mean().sort_values(ascending=False)
        gate_pressure = df_util.groupby('Gate_ID')['Weekly_Movements'].sum().sort_values(ascending=False).head(10)
        
        infrastructure_story = {
            'piers': [{'name': str(k), 'value': round(float(v), 1)} for k, v in pier_pressure.items()],
            'gates': [{'id': str(int(float(k))), 'value': int(v)} for k, v in gate_pressure.items()]
        }

        # ── SECTION 4: Terminal Story ────────────────────────
        t_stats = df_util.groupby('Terminal')['Weekly_Movements'].sum().to_dict()
        t_pax = {
            'Terminal 1': int(df_arr[df_arr['Terminal'] == 'T1']['Weekly_Movements'].sum() * 180 * 0.85), # Heuristic
            'Terminal 2': int(df_arr[df_arr['Terminal'] == 'T2']['Weekly_Movements'].sum() * 240 * 0.88)
        }
        
        terminal_story = {
            't1': {'movements': int(t_stats.get('Terminal 1', 0)), 'pax_est': t_pax['Terminal 1']},
            't2': {'movements': int(t_stats.get('Terminal 2', 0)), 'pax_est': t_pax['Terminal 2']}
        }

        return jsonify({
            'status': 'success',
            'demand_story': demand_story,
            'operational_story': operational_story,
            'infrastructure_story': infrastructure_story,
            'terminal_story': terminal_story
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =====================================================================
# LIVE OPERATIONS SIMULATOR ENDPOINTS
# =====================================================================

@api_blueprint.route('/live/generate', methods=['GET'])
def live_generate_scenario():
    """Generates a slice of flights falling within the 07:00-08:00 peak hour window."""
    try:
        from api.near_term_data import load_schedule
        import random
        schedule = load_schedule()
        
        # 07:00 = 420 mins, 08:00 = 480 mins
        start_min = 420
        end_min = 480
        
        active_flights = []
        seen = set()
        for f in schedule:
            if not f.get('start_min') or not f.get('end_min') or not f.get('flight'):
                continue
            if f['flight'] in seen:
                continue
            # If the flight's presence overlaps the 7:00-8:00 window
            if f['start_min'] < end_min and f['end_min'] > start_min:
                active_flights.append(f)
                seen.add(f['flight'])
                
        occupied = []
        arriving = []
        for f in active_flights:
            # We want to show both Arrivals and Departures
            if f['type'] == 'DEP':
                occupied.append(f)
            else:
                arriving.append(f)
                
        occupied = occupied[:8]
        arriving = sorted(arriving, key=lambda x: x['start_min'])[:15]
        
        # Add visual metadata (Passengers, Top 3 Suggestions)
        for fl_list in [occupied, arriving]:
            for f in fl_list:
                cat = f.get('cat', 'C')
                pax = random.randint(140, 189) if cat == 'C' else (random.randint(200, 250) if cat == 'D' else random.randint(250, 350))
                f['pax'] = pax
                
                planned = f.get('gate', '').strip()
                if not planned:
                    planned = '101'
                
                prefix = planned[0] if planned[0].isdigit() else '1'
                opts = [planned]
                for _ in range(5): # Generate a few, then pick first 3 uniques
                    g = f"{prefix}0{random.randint(1,6)}"
                    if g not in opts: opts.append(g)
                f['suggestions'] = opts[:3]
        
        return jsonify({
            'status': 'success',
            'scenario': {
                'occupied': occupied,
                'arriving': arriving
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@api_blueprint.route('/live/score', methods=['POST'])
def live_score():
    """
    Evaluates a user drag-and-drop against the allocator AI core constraints.
    Returns: { valid: bool, penalty: float, breakdown: list }
    """
    try:
        from engine.allocator import score_stand
        from engine.models import Flight, Stand, OperatorPreferences
        from data.loader import load_stands_final
        
        data = request.json
        flight_data = data.get('flight')
        gate_str = data.get('gate')
        
        # Handle aprons generic drops by defaulting to True if it exists
        if gate_str.startswith('Apron'):
             return jsonify({'status': 'success', 'valid': True, 'score': 0, 'breakdown': {'generic_apron': 0}})

        # Ensure stands csv uses the exact path via ROOT
        import os
        _HERE = os.path.dirname(os.path.abspath(__file__))
        _ROOT = os.path.dirname(_HERE)
        stands_dict = load_stands_final(os.path.join(_ROOT, 'data', 'inputs', 'stands_final.csv'))
        
        if gate_str not in stands_dict:
            return jsonify({'status': 'success', 'valid': False, 'score': -500, 'breakdown': {'unknown_gate': -500}})
            
        stand = stands_dict[gate_str]
        
        # Build mock flight
        f = Flight(
            flight_no=flight_data.get('flight', 'UNKNOWN'),
            airline=flight_data.get('airline', 'RYR'),
            aircraft_type=flight_data.get('aircraft', 'B738'),
            icao_cat=flight_data.get('cat', 'C'),
            is_cbp=flight_data.get('cbp', False),
            is_arrival=(flight_data.get('type') == 'ARR'),
            scheduled_time=flight_data.get('time', '00:00'),
            route_port=flight_data.get('route', 'DUB')
        )
        
        prefs = OperatorPreferences()
        
        # Fix DummyBlock attributes required by score_stand
        class DummyBlock:
            def __init__(self, f):
                self.arrival_flight = f if f.is_arrival else None
                self.departure_flight = f if not f.is_arrival else None
                self.terminal = 'T1' if 'RYR' in f.airline else 'T2'
                self.original_gate = None
                self.pier_preference = 'Pier 1' if 'RYR' in f.airline else 'Pier 4'
                self.cbp_required = f.is_cbp
                self.departure_flight = f if not f.is_arrival else None
        
        scores = score_stand(f, stand, prefs, DummyBlock(f), {}, [])
        
        total_p = sum(scores.values())
        return jsonify({
            'status': 'success',
            'valid': total_p >= -20.0,  # Allows soft violations, hard violations are -100
            'score': total_p,
            'breakdown': scores
        })
        
    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500
