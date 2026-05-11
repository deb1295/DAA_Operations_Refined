import csv
from pathlib import Path
from typing import List, Dict
from engine.models import Flight, Stand

def load_stands(filepath: str) -> Dict[str, Stand]:
    """Load the stand matrix from stands_final.csv."""
    stands = {}
    with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_id = row['stand_id'].strip()
            yes_no = lambda v: v.strip().upper() == 'YES'
            stands[s_id] = Stand(
                stand_id=s_id,
                pier=row['pier'].strip(),
                terminal=row['terminal'].strip(),
                icao_cat_max=row['icao_cat_max'].strip().upper(),
                has_airbridge=yes_no(row.get('has_airbridge', 'NO')),
                cbp_eligible=yes_no(row.get('cbp_eligible', 'NO')),
                mars_condition=row.get('mars_condition', '').strip(),
                mars_family_id=row.get('mars_family_id', '').strip(),
                stand_type=row['stand_type'].strip().upper()   # CONTACT / REMOTE / SATELLITE
            )
    return stands

def load_airline_terminals(filepath: str) -> Dict[str, Dict[str, str]]:
    """Load terminal mapping (AirlineCode -> {terminal, pier_pref}). Handles DEFAULT routing."""
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['airline_code'].strip()
            mapping[code] = {
                'terminal': row['terminal'].strip(),
                'pier_preference': row.get('pier_preference', '').strip()
            }
    return mapping

def load_flights(filepath: str, direction: str, airline_mapping: Dict[str, Dict[str, str]]) -> List[Flight]:
    """
    Load arrivals or departures.
    direction: 'ARR' or 'DEP'
    """
    flights = []
    
    # Identify dynamic columns based on direction
    time_col = 'sta' if direction == 'ARR' else 'std'
    port_col = 'origin_code' if direction == 'ARR' else 'destination_code'
    actual_col = 'actual_landing' if direction == 'ARR' else 'actual_departure'
    
    try:
        with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Basic cleaning
                date_str = row['date'].strip()
                time_str = row[time_col].strip()
                tail = row['tail_reg'].strip()
                
                # Skip invalid rows or footer artifacts
                if not date_str or not time_str or tail == 'None':
                    continue
                
                airline_code = row['flight_no'][:2] if row['flight_no'] else 'DEFAULT'
                
                # Determine terminal and pier preference
                mapping_data = airline_mapping.get(airline_code)
                term = mapping_data['terminal'] if mapping_data else None
                pier_pref = mapping_data['pier_preference'] if mapping_data else None
                
                if not term:
                    # Fallback default matching
                    if 'Ryanair' in row['airline_name']:
                        term = 'T1'
                        pier_pref = 'Pier 1'
                    elif 'Aer Lingus' in row['airline_name'] or 'Delta' in row['airline_name'] or 'American' in row['airline_name']:
                        term = 'T2'
                        pier_pref = 'Pier 4'
                    else:
                        default_p = airline_mapping.get('DEFAULT_T1', {'terminal': 'T1', 'pier_preference': 'Pier 2'})
                        term = default_p['terminal']
                        pier_pref = default_p['pier_preference']
                        
                # CBP Flag (arrivals only logic explicitly declared in csv)
                # Departures do not have cbp_flag column, so default to False
                cbp_flag_raw = str(row.get('cbp_flag', 'False'))
                is_cbp = cbp_flag_raw.strip().lower() == 'true'

                flights.append(Flight(
                    date_str=date_str,
                    time_str=time_str,
                    flight_no=row['flight_no'].strip(),
                    direction=direction,
                    airport_code=row[port_col].strip(),
                    airline_name=row['airline_name'].strip(),
                    aircraft_type=row['aircraft_type'].strip(),
                    tail_reg=tail,
                    icao_cat=row['icao_cat'].strip().upper(),
                    cbp_flag=is_cbp,
                    actual_time_str=row[actual_col].strip(),
                    assigned_gate=row['gate'].strip(),
                    terminal_assignment=term,
                    pier_preference=pier_pref
                ))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        
    return flights
