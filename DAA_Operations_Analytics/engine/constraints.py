from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from engine.models import MovementBlock, Stand

# ICAO character hierarchy
ICAO_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

def _terminal_ok(block_term: str, stand: Stand) -> bool:
    """
    True if this stand legally serves the flight's terminal.
    - REMOTE/SATELLITE => serves any terminal (bus/remote procedure)
    - 'T1/T2 Shared'   => serves T1 or T2
    - 'Cargo'          => Cargo or unassigned only
    - Otherwise exact match
    """
    if stand.stand_type in ('REMOTE', 'SATELLITE'):
        return True
    st = stand.terminal
    if st == 'T1/T2 Shared':
        return block_term in ('T1', 'T2')
    if st == 'Cargo':
        return block_term in ('Cargo', '')
    return st == block_term


def passes_static_constraints(
    block: MovementBlock,
    stand: Stand,
    config: dict = None,
    closed_stands: list = None
) -> Tuple[bool, str]:
    """Check if the stand physically and legally supports the flight."""
    if config is None:
        config = {}

    # 0. Closed stand check (operator override)
    if closed_stands and stand.stand_id in closed_stands:
        return False, f"Stand {stand.stand_id} is closed (operator override)"

    # 1. ICAO Size Check
    b_cat = ICAO_MAP.get(block.icao_cat, 3)
    s_cat = ICAO_MAP.get(stand.icao_cat_max, 3)
    if b_cat > s_cat:
        return False, "ICAO Category Exceeds Stand Limit"

    # 2. CBP Check — only CBP eligible stands can accept US-preclearance flights
    if block.cbp_required and not stand.cbp_eligible:
        return False, "CBP Required but stand ineligible"

    # 3. Terminal Match
    if not _terminal_ok(block.terminal, stand):
        return False, f"Terminal Mismatch ({block.terminal} vs {stand.terminal})"

    return True, "Valid"


def is_stand_available(
    target_stand: str,
    start: datetime,
    end: datetime,
    allocations: List[MovementBlock],
    stands_dict: Dict[str, Stand],
    config: dict = None
) -> bool:
    """Check if a stand is free for [start, end] including MARS blocking."""
    if config is None:
        config = {}

    stand_obj = stands_dict.get(target_stand)
    if not stand_obj:
        return False

    # Build the conflict family using MARS family ID and inferred role
    conflict_stands = {target_stand}
    if stand_obj.mars_family_id:
        fam = stand_obj.mars_family_id
        role = stand_obj.mars_type   # 'Centre', 'Left', 'Right', 'Tail', 'None'
        for sid, s in stands_dict.items():
            if s.mars_family_id == fam and s.stand_id != target_stand:
                if role == 'Centre':
                    # Centre is occupied => all siblings blocked
                    conflict_stands.add(sid)
                elif s.mars_type == 'Centre':
                    # Left/Right/Tail being used => Centre blocked
                    conflict_stands.add(sid)

    buffer_min = config.get("stand_buffer_min", 10)

    for alloc in allocations:
        if alloc.assigned_stand in conflict_stands:
            occ_start = alloc.occupancy_start - timedelta(minutes=buffer_min)
            occ_end = alloc.occupancy_end + timedelta(minutes=buffer_min)
            if occ_start < end and occ_end > start:
                return False

    return True


def get_eligible_stands(
    block: MovementBlock,
    all_stands: Dict[str, Stand],
    current_allocations: List[MovementBlock],
    config: dict = None,
    closed_stands: list = None
) -> List[Stand]:
    """Return all stands that are legally capable and point-in-time available."""
    if config is None:
        config = {}

    eligible = []
    for sid, stand in all_stands.items():
        ok, _ = passes_static_constraints(block, stand, config, closed_stands)
        if ok:
            if is_stand_available(sid, block.occupancy_start, block.occupancy_end,
                                  current_allocations, all_stands, config):
                eligible.append(stand)
    return eligible
