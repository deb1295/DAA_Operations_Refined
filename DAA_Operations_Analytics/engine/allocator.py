from typing import List, Dict, Optional
from dataclasses import dataclass
from engine.models import MovementBlock, Stand, DecisionRecord
from engine.constraints import get_eligible_stands
import json

@dataclass
class OperatorPreferences:
    w_contact: float = 8.0
    w_pier_match: float = 3.0
    w_seasonal_preference: float = 10.0
    w_airbridge: float = 2.0
    w_long_stay_penalty: float = 5.0
    w_south_apron_cbp_penalty: float = -8.0  # Last-resort overflow: penalise bus stands
    w_sizing_efficiency: float = 2.0
    w_regional_satellite_boost: float = 20.0

def score_stand(block: MovementBlock, stand: Stand, prefs: OperatorPreferences) -> Dict[str, float]:
    """Score a single stand based on soft operator preferences."""
    scores = {}
    
    # 1. Contact / Satellite preference (prefer over remote)
    if stand.stand_type in ('CONTACT', 'SATELLITE'):
        scores['contact'] = prefs.w_contact
    else:
        scores['contact'] = 0.0
        
    # 2. Terminal match (exact or Shared)
    if stand.terminal in (block.terminal, 'T1/T2 Shared'):
        scores['pier_match'] = prefs.w_pier_match
    else:
        scores['pier_match'] = 0.0

    # 3. Seasonal / Pier Preference (+10 points)
    # Give boost if stand matches the original historical gate OR the airline's preferred pier.
    # This keeps Ryanair in Pier 1/2 and Aer Lingus in Pier 4 as they are in real life.
    preference_match = False
    if block.original_gate and block.original_gate == stand.stand_id:
        preference_match = True
    elif block.pier_preference and block.pier_preference == stand.pier:
        preference_match = True
        
    if preference_match:
        scores['seasonal_preference'] = prefs.w_seasonal_preference
    else:
        scores['seasonal_preference'] = 0.0
        
    # 4. Airbridge bonus
    if stand.has_airbridge:
        scores['airbridge'] = prefs.w_airbridge
    else:
        scores['airbridge'] = 0.0

    # 5. South Apron CBP overflow penalty — only given to stands 411-418 (bus stands)
    SOUTH_APRON_CBP_OVERFLOW = {'411C','411L','411R','411T','412','413','414','415','416','417','418'}
    if block.cbp_required and stand.stand_id in SOUTH_APRON_CBP_OVERFLOW:
        scores['south_apron_cbp_penalty'] = prefs.w_south_apron_cbp_penalty
    else:
        scores['south_apron_cbp_penalty'] = 0.0

    import hashlib
    tie_str = f"1.{int(hashlib.sha256(stand.stand_id.encode()).hexdigest()[:8], 16)}"
    scores['tie_breaker'] = float(tie_str) * 0.000000001
    
    # 7. Sizing Efficiency (penalise over-sizing)
    ICAO_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    b_idx = ICAO_MAP.get(block.icao_cat, 3)
    s_idx = ICAO_MAP.get(stand.icao_cat_max, 3)
    if s_idx > b_idx:
        scores['sizing_efficiency'] = -prefs.w_sizing_efficiency * (s_idx - b_idx)
    else:
        scores['sizing_efficiency'] = 0.0

    # 8. Regional/Small Plane -> Satellite boost (Specifically for Aer Lingus Satellite 411-418)
    is_regional = False
    arr_f = block.arrival_flight
    dep_f = block.departure_flight
    if (arr_f and 'Aer Lingus' in arr_f.airline_name) or (dep_f and 'Aer Lingus' in dep_f.airline_name):
        is_regional = True
    
    if is_regional and block.icao_cat in ('A', 'B'):
        # Target South Apron (411-418)
        if stand.stand_type == 'SATELLITE' and stand.pier == 'South Apron':
            scores['regional_satellite_boost'] = prefs.w_regional_satellite_boost
        elif stand.stand_type == 'CONTACT' and stand.pier == 'Pier 4':
            scores['regional_satellite_boost'] = -10.0  # Push away from main Pier 4 Contact gates
        else:
            scores['regional_satellite_boost'] = 0.0
    else:
        scores['regional_satellite_boost'] = 0.0

    return scores

def generate_allocation(
    blocks: List[MovementBlock],
    stands: Dict[str, Stand],
    prefs: OperatorPreferences,
    config: dict = None,
    closed_stands: list = None
) -> List[DecisionRecord]:
    """
    Run greedy allocation chronologically.
    Returns the list of decision records (audit trail).
    Modifies the blocks in-place with `assigned_stand`.
    """
    if config is None:
        config = {}
    if closed_stands is None:
        closed_stands = []

    decisions = []
    
    # Sort chronologically by arrival/occupancy start
    sorted_blocks = sorted(blocks, key=lambda b: b.occupancy_start)
    
    # Track allocations incrementally
    current_allocations = []
    
    for block in sorted_blocks:
        # Determine unique flight ID
        f_id = "UNKNOWN"
        dir_t = "N/A"
        orig_gate = "N/A"
        if block.arrival_flight:
            f_id = block.arrival_flight.flight_no
            dir_t = "ARR_TURN" if block.departure_flight else "ARR_ONLY"
            orig_gate = block.arrival_flight.assigned_gate
        elif block.departure_flight:
            f_id = block.departure_flight.flight_no
            dir_t = "DEP_ONLY"
            orig_gate = block.departure_flight.assigned_gate
            
        # Get purely eligible stands
        eligible = get_eligible_stands(block, stands, current_allocations, config, closed_stands)
        
        if not eligible:
            # Unresolved Conflict
            block.assigned_stand = None
            decisions.append(DecisionRecord(
                flight_id=f_id,
                direction=dir_t,
                original_gate=orig_gate,
                assigned_stand="UNRESOLVED",
                priority_tier=1,
                score_breakdown={},
                total_score=0.0,
                eligible_count=0,
                suggestions=[],
                explanation=f"Critical: No eligible stands available for {block.icao_cat} at {block.occupancy_start.strftime('%H:%M')}."
            ))
            continue
            
        # Score eligible stands
        scored = []
        for s in eligible:
            breakdown = score_stand(block, s, prefs)
            total = sum(breakdown.values())
            scored.append({'stand': s, 'total': total, 'breakdown': breakdown})
            
        # Sort by total score descending
        scored.sort(key=lambda x: x['total'], reverse=True)
        
        best = scored[0]
        assigned = best['stand']
        block.assigned_stand = assigned.stand_id
        
        # Track for future time conflicts
        current_allocations.append(block)
        
        # Capture Top 3 suggestions (excluding the assigned one)
        top_suggestions = [s['stand'].stand_id for s in scored[1:4]]
        
        # Formulate explanation
        exp = f"Allocated to {assigned.stand_id}. Matched {len(best['breakdown'])-1} priority criteria."
        if assigned.stand_id != orig_gate and orig_gate:
            exp = f"Rerouted from {orig_gate} to {assigned.stand_id} due to constraints or better score."
            
        decisions.append(DecisionRecord(
            flight_id=f_id,
            direction=dir_t,
            original_gate=orig_gate,
            assigned_stand=assigned.stand_id,
            priority_tier=1,
            score_breakdown=best['breakdown'],
            total_score=best['total'],
            eligible_count=len(eligible),
            suggestions=top_suggestions,
            explanation=exp
        ))
        
    return decisions
