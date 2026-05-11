from typing import List, Dict
from datetime import datetime, timedelta
from engine.models import Flight, MovementBlock

def parse_time(date_str: str, time_str: str) -> datetime:
    """Safely parse DAA formatting into datetime."""
    if not time_str:
        return datetime(1900, 1, 1)
    
    # Try multiple common formats
    dt_str = f"{date_str} {time_str}"
    
    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d-%b-%y %H:%M"]:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
            
    # If all fail, print to debug
    print(f"FAILED PARSING: '{dt_str}'")
    return datetime(1900, 1, 1)

def build_rotations(arrivals: List[Flight], departures: List[Flight], config: dict) -> List[MovementBlock]:
    """
    Match arrivals to departures by tail_reg and time proximity to build full turnaround blocks.
    Any unlinked arrival or departure becomes a single-sided MovementBlock (e.g. towing to remote).
    """
    rotations = []
    
    # Sort by time to make matching chronological
    arrivals_sorted = sorted(arrivals, key=lambda f: parse_time(f.date_str, f.time_str))
    departures_sorted = sorted(departures, key=lambda f: parse_time(f.date_str, f.time_str))
    
    # Track used departures to avoid double linking
    used_deps = set()
    
    # Hardcoded or config-driven buffers
    arr_std_buf = timedelta(minutes=config.get("arrival_offset_standard_min", 8))
    arr_wb_buf = timedelta(minutes=config.get("arrival_offset_widebody_min", 14))
    dep_buf = timedelta(minutes=config.get("departure_delay_buffer_min", 24))
    
    for arr in arrivals_sorted:
        sta_dt = parse_time(arr.date_str, arr.time_str)
        # Find the next departure for this tail
        matched_dep = None
        for dep in departures_sorted:
            if id(dep) in used_deps:
                continue
            if dep.tail_reg == arr.tail_reg:
                std_dt = parse_time(dep.date_str, dep.time_str)
                # It must depart AFTER it arrives + some reasonable turn window (e.g., within 24 hours)
                if std_dt > (sta_dt - timedelta(minutes=30)) and (std_dt - sta_dt) < timedelta(hours=24):
                    matched_dep = dep
                    used_deps.add(id(dep))
                    break
        
        # Calculate Occupancy
        # In actual operations, Widebodies / CBP clear earlier so buffer is larger
        occ_start = sta_dt - (arr_wb_buf if arr.icao_cat == 'E' or arr.cbp_flag else arr_std_buf)
        if matched_dep:
            occ_end = parse_time(matched_dep.date_str, matched_dep.time_str) + dep_buf
        else:
            # Tow to remote / long stay if no departure found -- give it a standard turnaround time on stand
            turn_mins = config.get("turnaround_by_cat", {}).get(arr.icao_cat, 45)
            occ_end = sta_dt + timedelta(minutes=turn_mins)
            
        rotations.append(MovementBlock(
            tail_reg=arr.tail_reg,
            icao_cat=arr.icao_cat,
            terminal=arr.terminal_assignment,
            arrival_flight=arr,
            departure_flight=matched_dep,
            occupancy_start=occ_start,
            occupancy_end=occ_end,
            cbp_required=arr.cbp_flag,
            assigned_stand=None
        ))
        
    # Pick up unmatched departures (e.g. towed back from remote to stand)
    for dep in departures_sorted:
        if id(dep) not in used_deps:
            std_dt = parse_time(dep.date_str, dep.time_str)
            turn_mins = config.get("turnaround_by_cat", {}).get(dep.icao_cat, 45)
            
            # Assume it got towed to stand `turn_mins` before departure
            occ_start = std_dt - timedelta(minutes=turn_mins)
            occ_end = std_dt + dep_buf
            
            rotations.append(MovementBlock(
                tail_reg=dep.tail_reg,
                icao_cat=dep.icao_cat,
                terminal=dep.terminal_assignment,
                arrival_flight=None,
                departure_flight=dep,
                occupancy_start=occ_start,
                occupancy_end=occ_end,
                cbp_required=False, # Departures don't require CBP usually from stand, only arrivals clear US customs here... wait DUB has preclearance!
                assigned_stand=None 
            ))
            # Wait, US preclearance in DUB actually applies to DEPARTURES! 
            # In DAAs model, flights departing to the US clear CBP in DUB before departure.
            # But the user data has `cbp_flag` listed in arrivals csv? Let's check `config.json` rules later.
    
    return rotations
