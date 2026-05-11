import csv
from datetime import datetime, timedelta
from typing import List, Dict
from engine.models import MovementBlock, Flight, Stand
from engine.allocator import OperatorPreferences, generate_allocation, get_eligible_stands
from engine.ingestion import load_stands, load_airline_terminals

def safe_parse_time(time_str: str, base_date: datetime) -> datetime:
    if not time_str or time_str == 'N/A':
        return base_date
    try:
        if ':' in time_str:
            h, m = map(int, time_str.split(':')[:2])
            return base_date.replace(hour=h, minute=m, second=0, microsecond=0)
    except:
        pass
    return base_date

def build_predictive_rotations(
    arrivals_path: str,
    departures_path: str,
    airline_mapping: dict,
    config: dict
) -> List[MovementBlock]:
    """
    Build MovementBlocks straight from the matched next3days CSV.
    """
    blocks = []
    
    # 1. Load all departures for destination lookup
    dep_lookup = {}
    try:
        with open(departures_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_date = row.get('date', '').strip()
                d_fno = row.get('flight_no', '').strip()
                d_dest = row.get('destination_code', '').strip()
                if d_date and d_fno:
                    dep_lookup[(d_fno, d_date)] = d_dest
    except Exception as e:
        print(f"Warning: could not load departures lookup: {e}")

    # 2. Read the matched arrivals CSV
    try:
        with open(arrivals_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row.get('date', '').strip()
                if not date_str:
                    continue
                
                # Parse date
                try:
                    base_dt = datetime.strptime(date_str, "%d-%b-%y")
                except:
                    base_dt = datetime(2026, 4, 1)

                sta_str = row.get('sta', '').strip()
                sta_dt = safe_parse_time(sta_str, base_dt)
                
                airline = row.get('airline_name', '').strip()
                airline_code = row.get('flight_no', '')[:2] if row.get('flight_no') else 'DEFAULT'
                
                mapping_data = airline_mapping.get(airline_code)
                term = mapping_data['terminal'] if mapping_data else None
                pier_pref = mapping_data['pier_preference'] if mapping_data else None
                
                if not term:
                    if 'Ryanair' in airline:
                        term = 'T1'
                        pier_pref = 'Pier 1'
                    elif 'Aer Lingus' in airline or 'Delta' in airline or 'American' in airline:
                        term = 'T2'
                        pier_pref = 'Pier 4'
                    else:
                        default_p = airline_mapping.get('DEFAULT_T1', {'terminal': 'T1', 'pier_preference': 'Pier 1'})
                        term = default_p['terminal']
                        pier_pref = default_p['pier_preference']

                icao_cat = row.get('icao_cat', 'C').strip().upper()
                raw_cbp = str(row.get('cbp_flag', 'False')).strip().lower()
                cbp_flag = raw_cbp in ('true', '1', 'yes', 't')
                tail = row.get('tail_reg_updated', '').strip() or row.get('tail_reg', '').strip()
                gate = row.get('gate', '').strip()

                # Build Arrival Flight
                arr_flight = Flight(
                    date_str=date_str,
                    time_str=sta_str,
                    flight_no=row.get('flight_no', '').strip(),
                    direction='ARR',
                    airport_code=row.get('origin_code', '').strip(),
                    airline_name=airline,
                    aircraft_type=row.get('aircraft_type', '').strip(),
                    tail_reg=tail,
                    icao_cat=icao_cat,
                    cbp_flag=cbp_flag,
                    actual_time_str='',
                    assigned_gate=gate,
                    terminal_assignment=term,
                    pier_preference=pier_pref
                )

                # Matched params
                match_strategy = row.get('match_strategy', '').strip()
                dep_flight_no = row.get('matched_dep_flight', '').strip()
                turnaround_min_str = row.get('matched_turnaround_min', '').strip()
                
                # Determine Occupancy
                arr_buf = config.get("arrival_offset_standard_min", 8)
                occ_start = sta_dt - timedelta(minutes=arr_buf)
                
                dep_flight = None
                if dep_flight_no and turnaround_min_str and turnaround_min_str.isdigit():
                    turn_mins = int(turnaround_min_str)
                    occ_end = sta_dt + timedelta(minutes=turn_mins)
                    
                    # Destination Lookup
                    dest_code = dep_lookup.get((dep_flight_no, date_str), 'N/A')
                    if dest_code == 'N/A':
                        # Fallback for short-haul same origin
                        dest_code = row.get('origin_code', 'N/A')

                    dep_flight = Flight(
                        date_str=date_str,
                        time_str=(sta_dt + timedelta(minutes=turn_mins - 24)).strftime("%H:%M"), # approximate std
                        flight_no=dep_flight_no,
                        direction='DEP',
                        airport_code=dest_code,
                        airline_name=airline,
                        aircraft_type=row.get('aircraft_type', '').strip(),
                        tail_reg=tail,
                        icao_cat=icao_cat,
                        cbp_flag=False,
                        actual_time_str='',
                        assigned_gate='N/A',
                        terminal_assignment=term,
                        pier_preference=pier_pref
                    )
                else:
                    # Singular Arrival (Night Stopper)
                    turn_mins = config.get("turnaround_by_cat", {}).get(icao_cat, 45)
                    occ_end = sta_dt + timedelta(minutes=turn_mins)
                    
                blocks.append(MovementBlock(
                    tail_reg=tail,
                    icao_cat=icao_cat,
                    terminal=term,
                    pier_preference=pier_pref,
                    arrival_flight=arr_flight,
                    departure_flight=dep_flight,
                    occupancy_start=occ_start,
                    occupancy_end=occ_end,
                    cbp_required=cbp_flag,
                    assigned_stand=None
                ))

    except Exception as e:
        print(f"Error loading predictive matched flights: {e}")
        
    # We should also load singular departures here, but since the user has mapped next3days, 
    # we can trust the linked blocks for the majority. For MVP, we will run the allocator on these.

    return blocks

def run_predictive_engine(
    config_path,
    stands_path,
    airline_path,
    arrivals_csv_path,
    departures_csv_path,
    prefs=None,
    config_overrides: dict = None,
    closed_stands: list = None
):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Apply any runtime overrides from the UI simulation panel
    if config_overrides:
        config.update(config_overrides)

    stands = load_stands(stands_path)
    mapping = load_airline_terminals(airline_path)
    if prefs is None:
        prefs = OperatorPreferences()

    blocks = build_predictive_rotations(arrivals_csv_path, departures_csv_path, mapping, config)

    decisions = generate_allocation(blocks, stands, prefs, config=config, closed_stands=closed_stands or [])
    return blocks, decisions
