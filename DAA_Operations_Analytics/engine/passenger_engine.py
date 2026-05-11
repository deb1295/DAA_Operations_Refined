import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import gamma
import math
import os
import json

# --- Configuration & Constants ---
SEAT_CAPACITY = {
    'B38M': 189, 'A20N': 186, 'A21N': 230, 'B738': 189, 'A320': 180, 'A319': 150,
    'E190': 98, 'E295': 132, 'E195': 118, 'BCS3': 145, 'BCS1': 125, 'AT76': 72,
    'AT72': 72, 'AT75': 72, 'DH8D': 78,
    'B788': 242, 'B789': 290, 'A332': 271, 'A333': 317, 'B77W': 350, 'B772': 273,
    'B763': 226, 'B764': 246, 'B752': 169
}

ICAO_FALLBACK = {
    'A': 19, 'B': 70, 'C': 180, 'D': 250, 'E': 300, 'F': 450
}

DEFAULT_LOAD_FACTOR = 0.85

def get_seat_capacity(aircraft_type, icao_cat):
    if pd.isna(aircraft_type): aircraft_type = ''
    if pd.isna(icao_cat): icao_cat = 'C'
    
    if aircraft_type in SEAT_CAPACITY:
        return SEAT_CAPACITY[aircraft_type]
    
    # Prefix matching (e.g. A32X -> A320)
    for key, val in SEAT_CAPACITY.items():
        if str(aircraft_type).startswith(key[:3]):
            return val
            
    return ICAO_FALLBACK.get(icao_cat, 180)

_forecast_df = None

def calculate_load_factor(flight_date_str, csv_path="data/outputs/forecast_2026.csv"):
    """Estimates the load factor based on historical seasonality."""
    global _forecast_df
    if _forecast_df is None:
        try:
            _forecast_df = pd.read_csv(csv_path)
        except Exception as e:
            print("Error loading forecast:", e)
            _forecast_df = pd.DataFrame()
            
    base_lf = DEFAULT_LOAD_FACTOR
    try:
        if isinstance(flight_date_str, str):
            dt = pd.to_datetime(flight_date_str)
        else:
            dt = flight_date_str
            
        if not _forecast_df.empty:
            week_num = dt.isocalendar().week
            match = _forecast_df[_forecast_df['Week_In_Year'] == week_num]
            if not match.empty:
                seasonal_index = match.iloc[0]['seasonal_index']
                base_lf = 0.82 * seasonal_index
    except:
        pass
    
    # Add +/- 2% noise
    noise = np.random.uniform(-0.02, 0.02)
    return min(1.0, max(0.4, base_lf + noise))

# --- Wave Generators (Kernels) ---
# Returns arrays of pax per minute relative to T=0
def generate_checkin_wave(pax_count, lead_time=180):
    # spans from -lead_time to -60 mins
    t = np.arange(-lead_time, -59)
    # Dynamic scale based on the window size to keep the shape consistent
    # Peak will be at loc + (a-1)*scale
    scale = (lead_time - 60) / 4
    y = gamma.pdf(-t, a=2, loc=60, scale=scale)
    # Normalize to total pax
    if np.sum(y) > 0:
        y = y / np.sum(y) * pax_count
    return t, y

def generate_security_wave(t_chk, checkin_y):
    # Lag Check-in by exactly 15 mins
    t_sec = t_chk + 15
    return t_sec, checkin_y

def generate_boarding_wave(pax_count):
    # -40 to -15 mins
    t = np.arange(-40, -14)
    y = np.full_like(t, pax_count / len(t), dtype=float)
    return t, y

def generate_cbp_wave(t_sec, checkin_y):
    # CBP lags Security by 30 mins, uses exact same Skewed Gamma shape as Check-in
    t_cbp = t_sec + 30
    return t_cbp, checkin_y

def generate_immigration_wave(pax_count):
    # Linear flow extended: +15 to +75 mins (60 mins duration instead of 35)
    t = np.arange(15, 76)
    # Uniform linear distribution
    y = np.full_like(t, pax_count / len(t), dtype=float)
    return t, y

def generate_baggage_wave(pax_count):
    # +25 to +80 mins
    t = np.arange(25, 81)
    # Right skew Gamma
    y = gamma.pdf(t, a=2, loc=25, scale=12)
    # Assume 75% have checked bags
    baggage_pax = pax_count * 0.75
    if np.sum(y) > 0:
        y = y / np.sum(y) * baggage_pax
    return t, y

# --- Aggregation Engine ---
from collections import defaultdict

def aggregate_flows(flights_df, is_arrival=False, start_time=None, end_time=None, freq_min=15):
    """
    Process a dataframe of flights into time-series waves.
    Returns a dict of aggregated timeseries.
    """
    if start_time is None:
        start_time = pd.to_datetime(flights_df['datetime']).min() - pd.Timedelta(hours=4)
    if end_time is None:
        end_time = pd.to_datetime(flights_df['datetime']).max() + pd.Timedelta(hours=2)
        
    checkin_dict = defaultdict(float)
    security_dict = defaultdict(float)
    boarding_dict = defaultdict(float)
    cbp_dict = defaultdict(float)
    immigration_dict = defaultdict(float)
    baggage_dict = defaultdict(float)
    lounge_inflow = defaultdict(float)
    lounge_outflow = defaultdict(float)
    
    US_DESTS = {'JFK', 'EWR', 'BOS', 'ORD', 'IAD', 'PHL', 'CLT', 'ATL', 'MCO', 'MIA', 'DFW', 'LAX', 'SFO', 'SEA', 'IND', 'MSP', 'DTW', 'DEN', 'BWI', 'CVG'}
    
    for _, row in flights_df.iterrows():
        dt = row['datetime']
        seats = get_seat_capacity(row.get('aircraft_type', ''), row.get('icao_cat', 'C'))
        lf = calculate_load_factor(dt)
        pax = seats * lf
        
        # Round dt to minute
        dt = dt.replace(second=0, microsecond=0)
        
        if not is_arrival:
            # Differentiate lead times: 180m for Long Haul/CBP, 120m for Short Haul/Domestic
            # Proxy: ICAO Cat D, E, F or CBP required = 180 mins. Cat A, B, C = 120 mins.
            
            # Check for CBP requirement: explicit flag or destination in US list
            is_cbp_explicit = str(row.get('CBP_Required', row.get('cbp_flag', 'False'))).upper() == 'TRUE'
            is_us_dest = str(row.get('destination_code', '')).upper() in US_DESTS
            is_cbp = is_cbp_explicit or is_us_dest
            
            is_intl = (row.get('ICAO_Cat', row.get('icao_cat', 'C')) in ['D', 'E', 'F']) or is_cbp
            lead_time = 180 if is_intl else 120
            
            t_chk, y_chk = generate_checkin_wave(pax, lead_time=lead_time)
            t_sec, y_sec = generate_security_wave(t_chk, y_chk)
            t_brd, y_brd = generate_boarding_wave(pax)
            
            for t_offset, val in zip(t_chk, y_chk):
                checkin_dict[dt + pd.Timedelta(minutes=int(t_offset))] += val
                
            for t_offset, val in zip(t_sec, y_sec):
                t_abs = dt + pd.Timedelta(minutes=int(t_offset))
                security_dict[t_abs] += val
                lounge_inflow[t_abs] += val
                
            if is_cbp:
                t_cbp, y_cbp = generate_cbp_wave(t_sec, y_sec)
                for t_offset, val in zip(t_cbp, y_cbp):
                    t_abs = dt + pd.Timedelta(minutes=int(t_offset))
                    cbp_dict[t_abs] += val
                    
            for t_offset, val in zip(t_brd, y_brd):
                t_abs = dt + pd.Timedelta(minutes=int(t_offset))
                boarding_dict[t_abs] += val
                lounge_outflow[t_abs] += val
        else:
            is_intl = (row.get('ICAO_Cat', row.get('icao_cat', 'C')) in ['D', 'E', 'F'])
            if is_intl:
                t_imm, y_imm = generate_immigration_wave(pax)
            else:
                t_imm, y_imm = [], []
                
            t_bag, y_bag = generate_baggage_wave(pax)
            
            for t_offset, val in zip(t_imm, y_imm):
                immigration_dict[dt + pd.Timedelta(minutes=int(t_offset))] += val
                
            for t_offset, val in zip(t_bag, y_bag):
                baggage_dict[dt + pd.Timedelta(minutes=int(t_offset))] += val

    # Convert to DataFrame
    time_idx = pd.date_range(start=start_time, end=end_time, freq='1min')
    master_df = pd.DataFrame(index=time_idx)
    master_df['checkin'] = pd.Series(checkin_dict)
    master_df['security'] = pd.Series(security_dict)
    master_df['cbp'] = pd.Series(cbp_dict)
    master_df['boarding'] = pd.Series(boarding_dict)
    master_df['immigration'] = pd.Series(immigration_dict)
    master_df['baggage'] = pd.Series(baggage_dict)
    master_df['lounge_inflow'] = pd.Series(lounge_inflow)
    master_df['lounge_outflow'] = pd.Series(lounge_outflow)
    master_df.fillna(0, inplace=True)

    master_df['lounge_occupancy'] = (master_df['lounge_inflow'] - master_df['lounge_outflow']).cumsum()
    master_df.loc[master_df['lounge_occupancy'] < 0, 'lounge_occupancy'] = 0
    
    # Resample to desired frequency
    resampled = master_df.resample(f'{freq_min}min').sum()
    resampled['lounge_occupancy'] = master_df['lounge_occupancy'].resample(f'{freq_min}min').mean()
    
    return resampled

def process_3day_plan():
    cache_path = "data/outputs/pax_3day_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    # Load 3-day arrivals and departures
    arr_path = "data/inputs/flights_arrivals_next3days_match.csv"
    dep_path = "data/inputs/flights_departures_next3days.csv"
    
    arr_df = pd.read_csv(arr_path, encoding='latin1')
    dep_df = pd.read_csv(dep_path, encoding='latin1')
    
    arr_df['datetime'] = pd.to_datetime(arr_df['date'] + ' ' + arr_df['sta'], format='%d-%b-%y %H:%M')
    dep_df['datetime'] = pd.to_datetime(dep_df['date'] + ' ' + dep_df['std'], format='%d-%b-%y %H:%M')
    
    # Start from 05:00 on first day (airport operations begin)
    first_day = arr_df['datetime'].min().replace(hour=0, minute=0, second=0)
    start_dt  = first_day + pd.Timedelta(hours=5)   # 05:00 day 1
    end_dt    = first_day + pd.Timedelta(days=3)     # midnight end of day 3
    
    arr_df = arr_df[(arr_df['datetime'] >= start_dt) & (arr_df['datetime'] < end_dt)]
    dep_df = dep_df[(dep_df['datetime'] >= start_dt) & (dep_df['datetime'] < end_dt)]
    
    # Aggregate (15-min bins) — compute from full day so pre-flight waves are included
    compute_start = first_day                        # waves can start before 05:00
    arr_waves = aggregate_flows(arr_df, is_arrival=True,  start_time=compute_start, end_time=end_dt, freq_min=15)
    dep_waves = aggregate_flows(dep_df, is_arrival=False, start_time=compute_start, end_time=end_dt, freq_min=15)
    
    combined = pd.DataFrame(index=arr_waves.index)
    combined['checkin']    = dep_waves.get('checkin',          0)
    combined['security']   = dep_waves.get('security',         0)
    combined['cbp']        = dep_waves.get('cbp',              0)
    combined['lounge']     = dep_waves.get('lounge_occupancy', 0)
    combined['boarding']   = dep_waves.get('boarding',         0)
    combined['immigration']= arr_waves.get('immigration',      0)
    combined['baggage']    = arr_waves.get('baggage',          0)
    combined.fillna(0, inplace=True)
    
    # Filter VIEW to start from 05:00 (waves computed from midnight, viewed from 05:00)
    combined = combined[combined.index >= start_dt]
    
    # Chart labels — HH:MM for ticks, full day string for annotation
    times      = [t.strftime("%H:%M")      for t in combined.index]
    day_labels = [t.strftime("%a %d %b")   for t in combined.index]
    
    # Table data — 30-min bins, split per day into 3 separate dicts
    table_df = combined.resample('30min').sum()
    table_df['lounge'] = combined['lounge'].resample('30min').mean()
    table_df.fillna(0, inplace=True)
    
    cols = ['checkin','security','cbp','lounge','boarding','immigration','baggage']
    days_data = []
    for d in range(3):
        day_start = first_day + pd.Timedelta(days=d, hours=5)
        day_end   = first_day + pd.Timedelta(days=d+1)
        mask = (table_df.index >= day_start) & (table_df.index < day_end)
        df_day = table_df[mask]
        day_dict = {
            "date":   (first_day + pd.Timedelta(days=d)).strftime("%a %d %b %Y"),
            "labels": [t.strftime("%H:%M") for t in df_day.index],
        }
        for c in cols:
            day_dict[c] = df_day[c].round(0).tolist() if c in df_day.columns else []
        days_data.append(day_dict)
    
    result = {
        "labels":    times,
        "day_labels": day_labels,
        "checkin":    combined['checkin'].round(1).tolist(),
        "security":   combined['security'].round(1).tolist(),
        "cbp":        combined['cbp'].round(1).tolist(),
        "lounge":     combined['lounge'].round(1).tolist(),
        "boarding":   combined['boarding'].round(1).tolist(),
        "immigration":combined['immigration'].round(1).tolist(),
        "baggage":    combined['baggage'].round(1).tolist(),
        "days_data":  days_data          # 3-element array, one per day
    }

    # Save to cache for next time
    with open(cache_path, 'w') as f:
        json.dump(result, f)

    return result

def process_intraday_pulse():
    # Load intraday flights
    path = "data/inputs/intraday_flights.csv"
    df = pd.read_csv(path, encoding='latin1')
    
    # Provide default date for intraday if Date is missing
    sim_date = '31-Mar-26'
    
    arrivals = df[df['Type'].isin(['ARR', 'TURN'])].copy()
    arrivals['datetime'] = pd.to_datetime(sim_date + ' ' + arrivals['ETA'], format='%d-%b-%y %H:%M', errors='coerce')
    arrivals.dropna(subset=['datetime'], inplace=True)
    
    departures = df[df['Type'].isin(['DEP', 'TURN'])].copy()
    departures['datetime'] = pd.to_datetime(sim_date + ' ' + departures['ETD'], format='%d-%b-%y %H:%M', errors='coerce')
    departures.dropna(subset=['datetime'], inplace=True)
    
    # Determine bounds (07:00 to 08:30)
    if not arrivals.empty:
        base_date = arrivals['datetime'].min().replace(hour=7, minute=0, second=0)
    else:
        base_date = pd.to_datetime("2026-01-01 07:00")
        
    start_dt = base_date - pd.Timedelta(hours=4) # start earlier to capture pre-flows
    end_dt = base_date + pd.Timedelta(hours=3)
    
    # Aggregate (1-min bins for pulse)
    arr_waves = aggregate_flows(arrivals, is_arrival=True, start_time=start_dt, end_time=end_dt, freq_min=1)
    dep_waves = aggregate_flows(departures, is_arrival=False, start_time=start_dt, end_time=end_dt, freq_min=1)
    
    # Combine
    combined = pd.DataFrame(index=arr_waves.index)
    combined['checkin'] = dep_waves.get('checkin', 0)
    combined['security'] = dep_waves.get('security', 0)
    combined['cbp'] = dep_waves.get('cbp', 0)
    combined['lounge'] = dep_waves.get('lounge_occupancy', 0)
    combined['boarding'] = dep_waves.get('boarding', 0)
    combined['immigration'] = arr_waves.get('immigration', 0)
    combined['baggage'] = arr_waves.get('baggage', 0)
    
    # Filter to 04:00 to 09:30 to show the full lifecycle of the morning wave check-ins
    view_start = base_date - pd.Timedelta(hours=3)
    view_end = base_date + pd.Timedelta(hours=2, minutes=30)
    combined = combined[(combined.index >= view_start) & (combined.index <= view_end)]
    
    combined.fillna(0, inplace=True)
    
    # Format for JSON
    times = [t.strftime("%H:%M") for t in combined.index]
    
    # Generate 15-min table data starting from 04:30
    table_df = combined.resample('15min').sum()
    table_df['lounge'] = combined['lounge'].resample('15min').mean()
    
    # Filter table to start from 04:30 (base_date - 2.5 hours)
    table_view_start = base_date - pd.Timedelta(hours=2, minutes=30)
    table_df = table_df[table_df.index >= table_view_start]
    
    table_labels = [t.strftime("%H:%M") for t in table_df.index]
    
    result = {
        "labels": times,
        "checkin": combined['checkin'].round(1).tolist(),
        "security": combined['security'].round(1).tolist(),
        "cbp": combined['cbp'].round(1).tolist(),
        "lounge": combined['lounge'].round(1).tolist(),
        "boarding": combined['boarding'].round(1).tolist(),
        "immigration": combined['immigration'].round(1).tolist(),
        "baggage": combined['baggage'].round(1).tolist(),
        "table_labels": table_labels,
        "table_data": {
            "checkin": table_df['checkin'].round(0).tolist(),
            "security": table_df['security'].round(0).tolist(),
            "cbp": table_df['cbp'].round(0).tolist(),
            "lounge": table_df['lounge'].round(0).tolist(),
            "boarding": table_df['boarding'].round(0).tolist(),
            "immigration": table_df['immigration'].round(0).tolist(),
            "baggage": table_df['baggage'].round(0).tolist()
        }
    }
    return result


# ─────────────────────────────────────────────────────────────────
# SIMULATION ENGINE  (Intra-Day What-If scenarios)
# ─────────────────────────────────────────────────────────────────

def _load_intraday_flights():
    """Load and parse raw intraday flights, returning (arrivals_df, departures_df, base_date)."""
    sim_date = '31-Mar-26'
    df = pd.read_csv("data/inputs/intraday_flights.csv", encoding='latin1')
    arrivals = df[df['Type'].isin(['ARR', 'TURN'])].copy()
    arrivals['datetime'] = pd.to_datetime(sim_date + ' ' + arrivals['ETA'], format='%d-%b-%y %H:%M', errors='coerce')
    arrivals.dropna(subset=['datetime'], inplace=True)
    departures = df[df['Type'].isin(['DEP', 'TURN'])].copy()
    departures['datetime'] = pd.to_datetime(sim_date + ' ' + departures['ETD'], format='%d-%b-%y %H:%M', errors='coerce')
    departures.dropna(subset=['datetime'], inplace=True)
    base_date = arrivals['datetime'].min().replace(hour=7, minute=0, second=0) if not arrivals.empty else pd.to_datetime("2026-01-01 07:00")
    return arrivals, departures, base_date


def _combine_waves(arr_waves, dep_waves):
    combined = pd.DataFrame(index=arr_waves.index)
    combined['checkin']    = dep_waves.get('checkin',          0)
    combined['security']   = dep_waves.get('security',         0)
    combined['cbp']        = dep_waves.get('cbp',              0)
    combined['lounge']     = dep_waves.get('lounge_occupancy', 0)
    combined['boarding']   = dep_waves.get('boarding',         0)
    combined['immigration']= arr_waves.get('immigration',      0)
    combined['baggage']    = arr_waves.get('baggage',          0)
    combined.fillna(0, inplace=True)
    return combined


def _apply_throughput_cap(series: pd.Series, cap_fraction: float, start_t, end_t) -> tuple:
    """Cap a time-series during [start_t, end_t] and compute queue overflow using absolute ceiling."""
    s = series.copy().astype(float)
    
    # 1. Determine "Nominal Capacity" (the peak demand the system is designed to handle)
    # We take the max demand in the window as 100% capacity.
    nominal_capacity = s.loc[start_t:end_t].max() if not s.loc[start_t:end_t].empty else s.max()
    if nominal_capacity <= 0: nominal_capacity = 1.0
    
    # 2. Set the Hard Ceiling (Absolute Throughput Limit)
    throughput_limit = nominal_capacity * cap_fraction
    
    queue_count = 0.0
    queue_series = np.zeros(len(s))
    
    for i, t in enumerate(s.index):
        demand = s.iloc[i]
        
        if start_t <= t <= end_t:
            # How many can physically pass this minute?
            can_pass = min(demand + queue_count, throughput_limit)
            
            # The rest stays in queue
            queue_count = (queue_count + demand) - can_pass
            s.iloc[i] = can_pass
        else:
            # Drain queue into remaining spare capacity
            # In off-peak, we assume system can handle 1.2x its previous demand to clear backlog
            spare_capacity = max(0, (throughput_limit / cap_fraction) - demand)
            drained = min(queue_count, spare_capacity)
            s.iloc[i] += drained
            queue_count -= drained
            
        queue_series[i] = queue_count
        
    return s, queue_series


def _calculate_table_data(combined_df, base_date):
    """Resample 1-min pulse data to 15-min table bins starting from 04:30."""
    table_df = combined_df.resample('15min').sum()
    table_df['lounge'] = combined_df['lounge'].resample('15min').mean()
    
    # Filter table to start from 04:30 (base_date - 2.5 hours)
    table_view_start = base_date - pd.Timedelta(hours=2, minutes=30)
    table_df = table_df[table_df.index >= table_view_start]
    
    table_labels = [t.strftime("%H:%M") for t in table_df.index]
    table_data = {
        "checkin": table_df['checkin'].round(0).tolist(),
        "security": table_df['security'].round(0).tolist(),
        "cbp": table_df['cbp'].round(0).tolist(),
        "lounge": table_df['lounge'].round(0).tolist(),
        "boarding": table_df['boarding'].round(0).tolist(),
        "immigration": table_df['immigration'].round(0).tolist(),
        "baggage": table_df['baggage'].round(0).tolist()
    }
    return table_labels, table_data


def _build_sim_result(labels, base_fmt, sim_fmt, cascade, at_risk, extra_notes, table_labels=None, table_data=None):
    """Assemble final simulation result dict including delta impact."""
    impact = {}
    for k in base_fmt:
        b_max = max(base_fmt[k]) if base_fmt[k] else 0
        s_max = max(sim_fmt[k])  if sim_fmt[k]  else 0
        impact[k] = {
            'baseline_peak': round(b_max, 1),
            'simulated_peak': round(s_max, 1),
            'delta': round(s_max - b_max, 1),
            'delta_pct': round((s_max - b_max) / b_max * 100, 1) if b_max > 0 else 0
        }
    return {
        'labels':        labels,
        'baseline':      base_fmt,
        'simulated':     sim_fmt,
        'impact':        impact,
        'cascade':       cascade,
        'at_risk_flights': at_risk,
        'extra':         extra_notes,
        'table_labels':  table_labels,
        'table_data':    table_data
    }


def simulate_intraday(scenario: str, params: dict) -> dict:
    """
    Run a what-if simulation over the intraday pulse.
    Returns: { baseline: {...}, simulated: {...}, impact: {...},
               cascade: [...], at_risk_flights: [...], table_labels: [...], table_data: {...} }
    """
    arrivals, departures, base_date = _load_intraday_flights()
    start_dt   = base_date - pd.Timedelta(hours=4)
    end_dt     = base_date + pd.Timedelta(hours=3)
    view_start = base_date - pd.Timedelta(hours=3)
    view_end   = base_date + pd.Timedelta(hours=2, minutes=30)

    # ── BASELINE ──────────────────────────────────────────────
    arr_b = aggregate_flows(arrivals,   is_arrival=True,  start_time=start_dt, end_time=end_dt, freq_min=1)
    dep_b = aggregate_flows(departures, is_arrival=False, start_time=start_dt, end_time=end_dt, freq_min=1)
    base  = _combine_waves(arr_b, dep_b)
    base  = base[(base.index >= view_start) & (base.index <= view_end)]

    # ── SCENARIO MUTATIONS ────────────────────────────────────
    sim_arrivals   = arrivals.copy()
    sim_departures = departures.copy()
    at_risk        = []
    cascade        = []
    extra_notes    = {}

    if scenario == 'flight_delay':
        flight_no  = params.get('flight_no', '')
        delay_min  = int(params.get('delay_min', 30))
        mask = sim_departures['Flight_No'].astype(str).str.upper() == str(flight_no).upper()
        if mask.any():
            sim_departures.loc[mask, 'datetime'] += pd.Timedelta(minutes=delay_min)
            row = departures[mask].iloc[0]
            at_risk.append({
                'flight': flight_no,
                'type': 'delay',
                'detail': f'+{delay_min} min delay applied'
            })
            cascade = [
                f"🛡 Security queuing for {flight_no} pax shifts {delay_min} min later — may overlap next departure wave.",
                f"🚪 Boarding gate opens {delay_min} min later — gate agent resource gap likely.",
                f"🛋 Lounge occupancy rises as {flight_no} pax wait longer airside.",
            ]
            if str(row.get('CBP_Required', 'False')).upper() == 'TRUE':
                cascade.append(f"🇺🇸 CBP preclearance for {flight_no} delayed — US slot risk if delay > 45 min.")

    elif scenario == 'ground_stop':
        stop_hhmm   = params.get('start', '07:30')
        duration    = int(params.get('duration_min', 45))
        sim_date_dt = pd.to_datetime('2026-03-31')
        sh, sm      = map(int, stop_hhmm.split(':'))
        stop_start  = sim_date_dt.replace(hour=sh, minute=sm)
        stop_end    = stop_start + pd.Timedelta(minutes=duration)

        # Freeze flights during stop → compress to stop_end
        dep_mask = (sim_departures['datetime'] >= stop_start) & (sim_departures['datetime'] < stop_end)
        arr_mask = (sim_arrivals['datetime']   >= stop_start) & (sim_arrivals['datetime'] < stop_end)
        frozen_dep_count = dep_mask.sum()
        frozen_arr_count = arr_mask.sum()
        sim_departures.loc[dep_mask, 'datetime'] = stop_end
        sim_arrivals.loc[arr_mask, 'datetime']   = stop_end

        at_risk = [{'flight': r['Flight_No'], 'type': 'ground_stop', 'detail': f'Held until {stop_end.strftime("%H:%M")}'} for _, r in departures[dep_mask].iterrows()]
        cascade = [
            f"✈ {frozen_dep_count} departures frozen — compressed burst on lift creates check-in/security/boarding spike.",
            f"🛬 {frozen_arr_count} arrivals held — immigration and baggage surge on lift.",
            f"🛋 Lounge occupancy peaks sharply as pax build up airside during stop.",
        ]
        extra_notes['stop_start'] = stop_start.strftime('%H:%M')
        extra_notes['stop_end']   = stop_end.strftime('%H:%M')

    elif scenario == 'security_reduction':
        reduction_pct = float(params.get('reduction_pct', 40))
        start_hhmm    = params.get('start', '07:00')
        end_hhmm      = params.get('end', '08:30')
        sim_date_dt   = pd.to_datetime('2026-03-31')
        sh, sm        = map(int, start_hhmm.split(':'))
        eh, em        = map(int, end_hhmm.split(':'))
        cap_start     = sim_date_dt.replace(hour=sh, minute=sm)
        cap_end       = sim_date_dt.replace(hour=eh, minute=em)
        cap_fraction  = 1.0 - (reduction_pct / 100.0)

        arr_s = aggregate_flows(sim_arrivals,   is_arrival=True,  start_time=start_dt, end_time=end_dt, freq_min=1)
        dep_s = aggregate_flows(sim_departures, is_arrival=False, start_time=start_dt, end_time=end_dt, freq_min=1)
        sim   = _combine_waves(arr_s, dep_s)
        sim   = sim[(sim.index >= view_start) & (sim.index <= view_end)]

        sec_capped, sec_queue = _apply_throughput_cap(sim['security'], cap_fraction, cap_start, cap_end)
        sim['security'] = sec_capped
        # Security backup delays boarding
        max_queue       = sec_queue.max()
        wait_min        = round(max_queue / max(sim['security'].mean(), 1))
        extra_notes['max_queue_pax'] = int(max_queue)
        extra_notes['est_wait_min']  = wait_min
        cascade = [
            f"📋 Queue builds to ~{int(max_queue)} pax at peak; estimated wait ~{wait_min} min.",
            f"🚪 Boarding delays cascade: flights scheduled to depart during this window may slip.",
            f"✈ Check-in not directly affected — but pax arriving at security face extended queues.",
        ]
        at_risk = [{'flight': r['Flight_No'], 'type': 'sec_queue', 'detail': f'Est. {wait_min} min extra queue'} for _, r in departures.iterrows() if cap_start <= r['datetime'] <= cap_end + pd.Timedelta(minutes=30)]

        # Reconstruct result early for security_reduction since we already have sim df
        base_fmt = {k: base[k].round(1).tolist() for k in ['checkin','security','cbp','lounge','boarding','immigration','baggage']}
        sim_fmt  = {k: sim[k].round(1).tolist()  for k in ['checkin','security','cbp','lounge','boarding','immigration','baggage']}
        labels   = [t.strftime('%H:%M') for t in base.index]
        t_labels, t_data = _calculate_table_data(sim, base_date)
        return _build_sim_result(labels, base_fmt, sim_fmt, cascade, at_risk, extra_notes, t_labels, t_data)

    elif scenario == 'checkin_reduction':
        reduction_pct = float(params.get('reduction_pct', 40))
        start_hhmm    = params.get('start', '06:00')
        end_hhmm      = params.get('end', '08:00')
        sim_date_dt   = pd.to_datetime('2026-03-31')
        sh, sm        = map(int, start_hhmm.split(':'))
        eh, em        = map(int, end_hhmm.split(':'))
        cap_start     = sim_date_dt.replace(hour=sh, minute=sm)
        cap_end       = sim_date_dt.replace(hour=eh, minute=em)
        cap_fraction  = 1.0 - (reduction_pct / 100.0)

        arr_s = aggregate_flows(sim_arrivals,   is_arrival=True,  start_time=start_dt, end_time=end_dt, freq_min=1)
        dep_s = aggregate_flows(sim_departures, is_arrival=False, start_time=start_dt, end_time=end_dt, freq_min=1)
        sim   = _combine_waves(arr_s, dep_s)
        sim   = sim[(sim.index >= view_start) & (sim.index <= view_end)]

        chk_capped, chk_queue = _apply_throughput_cap(sim['checkin'], cap_fraction, cap_start, cap_end)
        sim['checkin'] = chk_capped
        max_queue      = chk_queue.max()
        extra_notes['max_queue_pax'] = int(max_queue)
        cascade = [
            f"📋 Queue builds to ~{int(max_queue)} pax — pax arrive at security later than baseline.",
            f"🛡 Security wave shifts right and extends — downstream boarding windows compress.",
            f"🛋 Longer check-in queue increases landside dwell time; airside occupancy initially lower.",
        ]
        at_risk = []

        base_fmt = {k: base[k].round(1).tolist() for k in ['checkin','security','cbp','lounge','boarding','immigration','baggage']}
        sim_fmt  = {k: sim[k].round(1).tolist()  for k in ['checkin','security','cbp','lounge','boarding','immigration','baggage']}
        labels   = [t.strftime('%H:%M') for t in base.index]
        t_labels, t_data = _calculate_table_data(sim, base_date)
        return _build_sim_result(labels, base_fmt, sim_fmt, cascade, at_risk, extra_notes, t_labels, t_data)

    # ── RE-RUN WAVES (scenarios 1 and 2 mutate datetimes) ────
    arr_s = aggregate_flows(sim_arrivals,   is_arrival=True,  start_time=start_dt, end_time=end_dt, freq_min=1)
    dep_s = aggregate_flows(sim_departures, is_arrival=False, start_time=start_dt, end_time=end_dt, freq_min=1)
    sim   = _combine_waves(arr_s, dep_s)
    sim   = sim[(sim.index >= view_start) & (sim.index <= view_end)]

    base_fmt = {k: base[k].round(1).tolist() for k in ['checkin','security','cbp','lounge','boarding','immigration','baggage']}
    sim_fmt  = {k: sim[k].round(1).tolist()  for k in ['checkin','security','cbp','lounge','boarding','immigration','baggage']}
    labels   = [t.strftime('%H:%M') for t in base.index]
    t_labels, t_data = _calculate_table_data(sim, base_date)
    return _build_sim_result(labels, base_fmt, sim_fmt, cascade, at_risk, extra_notes, t_labels, t_data)
