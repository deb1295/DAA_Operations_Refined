"""
engine/capacity_study.py
DAA Stand Analysis — Capacity Intelligence Engine (v2)
All metrics normalised to daily averages.
Gate-to-pier resolution uses numeric prefix matching for Pier 1/2/3/4.
"""
import csv
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List

ICAO_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
NARROWBODY_THRESHOLD = 3   # Cat <= C is narrowbody
WIDEBODY_THRESHOLD = 4     # Cat >= D is widebody

TIME_BANDS = [
    ('Dawn',     4,  7),
    ('AM Peak',  7, 11),
    ('Midday',  11, 15),
    ('PM Peak', 15, 19),
    ('Night',   19, 28),
]

# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────

def _load_csv(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
        return list(csv.DictReader(f))

def _parse_hour(t: str) -> int:
    try:
        return int(t.split(':')[0])
    except:
        return -1

def _time_band(hour: int) -> str:
    for label, start, end in TIME_BANDS:
        if start <= hour < end:
            return label
    return 'Overnight'

def _icao_rank(cat: str) -> int:
    return ICAO_MAP.get(cat.upper().strip(), 0)

def _is_narrowbody(cat: str) -> bool:
    return _icao_rank(cat) <= NARROWBODY_THRESHOLD

def _is_widebody(cat: str) -> bool:
    return _icao_rank(cat) >= WIDEBODY_THRESHOLD

def _is_mars_center(stand: Dict) -> bool:
    sid = stand['stand_id'].upper()
    fid = stand.get('mars_family_id', '').strip()
    return bool(fid) and (sid.endswith('C') or sid.endswith('T'))

# Fallback: Dublin pier range by gate number prefix
_PIER_RANGE_FALLBACK = {
    range(100, 200): 'Pier 1',
    range(200, 300): 'Pier 2',
    range(300, 400): 'Pier 3',
    range(400, 500): 'Pier 4',
}

def _build_gate_pier_map(stands: List[Dict]) -> Dict[str, str]:
    """
    Build a gate -> pier lookup with:
    - Direct match: '401L' -> 'Pier 4'
    - Numeric prefix: '401' -> 'Pier 4'
    - Range fallback: 115 -> Pier 1, 218 -> Pier 2 etc.
    """
    gate_pier = {}
    for s in stands:
        sid = s['stand_id']
        pier = s['pier']
        gate_pier[sid] = pier
        prefix = ''.join(c for c in sid if c.isdigit())
        if prefix and prefix not in gate_pier:
            gate_pier[prefix] = pier
    return gate_pier

def _resolve_pier(gate: str, gate_pier: Dict) -> str:
    gate = gate.strip()
    if gate in gate_pier:
        return gate_pier[gate]
    stripped = gate.rstrip('ABCLRT')
    if stripped in gate_pier:
        return gate_pier[stripped]
    # Range fallback for unmapped numeric gates
    try:
        n = int(stripped or gate)
        for rng, pier in _PIER_RANGE_FALLBACK.items():
            if n in rng:
                return pier
    except ValueError:
        pass
    return ''

def _resolve_gate_to_stand(gate: str, stands_index: Dict, icao_cat: str) -> str:
    gate = gate.strip()
    if gate in stands_index:
        return gate
    if gate + 'C' in stands_index:
        return gate + 'L' if _is_narrowbody(icao_cat) else gate + 'C'
    if gate + 'L' in stands_index:
        return gate + 'L'
    return gate

def _count_days(rows: List[Dict], date_col: str = 'date') -> int:
    dates = set(r.get(date_col, '').strip() for r in rows if r.get(date_col, '').strip())
    return max(len(dates), 1)


# ─────────────────────────────────────────────
# KPI SUMMARY — Linked Insight Cards
# ─────────────────────────────────────────────

def compute_kpis(stands: List[Dict], arrivals: List[Dict], departures: List[Dict]) -> Dict:
    n_days = _count_days(arrivals)
    total_arr = len(arrivals)
    total_dep = len(departures)
    total_flights = total_arr + total_dep

    daily_flights  = round(total_flights / n_days)
    daily_arr      = round(total_arr / n_days)
    daily_dep      = round(total_dep / n_days)

    narrow_arr = sum(1 for r in arrivals if _is_narrowbody(r.get('icao_cat', '')))
    wide_arr   = sum(1 for r in arrivals if _is_widebody(r.get('icao_cat', '')))
    cbp_arr    = sum(1 for r in arrivals if str(r.get('cbp_flag', '')).strip().lower() in ('true', '1', 'yes'))

    daily_narrow = round(narrow_arr / n_days)
    daily_wide   = round(wide_arr / n_days)
    daily_cbp    = round(cbp_arr / n_days)

    contact_stands  = [s for s in stands if s.get('stand_type') == 'CONTACT']
    remote_stands   = [s for s in stands if s.get('stand_type') in ('REMOTE', 'SATELLITE')]
    # "Widebody-capable" = Cat D and above
    wide_capable    = [s for s in stands if _icao_rank(s.get('icao_cat_max','C')) >= WIDEBODY_THRESHOLD]
    # "Narrowbody-only" = max Cat C or below
    narrow_only     = [s for s in stands if _icao_rank(s.get('icao_cat_max','C')) <= NARROWBODY_THRESHOLD]
    cbp_stands      = [s for s in stands if s.get('cbp_eligible','').upper() == 'YES']
    mars_centers    = [s for s in stands if _is_mars_center(s)]

    total_contact = max(len(contact_stands), 1)

    # Ratio insights
    flights_per_stand      = round(daily_flights  / total_contact, 1)
    hrs_per_flight_contact = round(24 / max(flights_per_stand, 0.1), 1)

    narrow_per_stand  = round(daily_narrow / max(len(wide_capable), 1), 1)  # vs widebody capable (most hold narrow too)
    hrs_per_narrow    = round(24 / max(narrow_per_stand, 0.1), 1)

    wide_per_stand    = round(daily_wide / max(len(wide_capable), 1), 1)
    hrs_per_wide      = round(24 / max(wide_per_stand, 0.1), 1)

    cbp_per_stand     = round(daily_cbp / max(len(cbp_stands), 1), 1)
    hrs_per_cbp       = round(24 / max(cbp_per_stand, 0.1), 1)

    return {
        'n_days_analysed': n_days,
        'daily_flights': daily_flights,
        'daily_arrivals': daily_arr,
        'daily_departures': daily_dep,
        'daily_narrowbody': daily_narrow,
        'daily_widebody': daily_wide,
        'daily_cbp': daily_cbp,
        'total_stands': len(stands),
        'contact_stands': len(contact_stands),
        'remote_stands': len(remote_stands),
        'widebody_capable_stands': len(wide_capable),
        'narrowbody_only_stands': len(narrow_only),
        'cbp_eligible_stands': len(cbp_stands),
        'mars_center_stands': len(mars_centers),
        # Linked ratio insights
        'flights_per_contact_stand': flights_per_stand,
        'avg_hrs_per_flight': hrs_per_flight_contact,
        'narrow_per_widecapable_stand': narrow_per_stand,
        'avg_hrs_per_narrow': hrs_per_narrow,
        'wide_per_widecapable_stand': wide_per_stand,
        'avg_hrs_per_wide': hrs_per_wide,
        'cbp_per_cbp_stand': cbp_per_stand,
        'avg_hrs_per_cbp': hrs_per_cbp,
    }


# ─────────────────────────────────────────────
# STUDY 1 — Supply vs. Demand (daily)
# ─────────────────────────────────────────────

def study_supply_demand(stands: List[Dict], arrivals: List[Dict]) -> Dict:
    n_days = _count_days(arrivals)
    demand = Counter()
    for r in arrivals:
        cat = r.get('icao_cat', 'C').strip().upper() or 'C'
        demand[cat] += 1

    supply = Counter()
    for s in stands:
        cat = s.get('icao_cat_max', 'C').strip().upper()
        supply[cat] += 1

    result = []
    for cat in sorted(ICAO_MAP.keys()):
        d_total = demand.get(cat, 0)
        d_daily = round(d_total / n_days, 1)
        # Cumulative eligible supply: stands that CAN handle this cat (their max >= this cat)
        eligible = sum(supply.get(c, 0) for c in ICAO_MAP if ICAO_MAP[c] >= ICAO_MAP[cat])
        result.append({
            'icao_cat': cat,
            'demand_daily': d_daily,
            'demand_total': d_total,
            'eligible_stands': eligible,
            'status': 'Very High' if d_daily > eligible * 2 else ('Balanced' if d_daily <= eligible else 'Low')
        })
    return {'data': result, 'n_days': n_days}


# ─────────────────────────────────────────────
# STUDY 2 — CBP Calibration (daily)
# ─────────────────────────────────────────────

def study_cbp_calibration(stands: List[Dict], arrivals: List[Dict]) -> Dict:
    n_days = _count_days(arrivals)
    cbp_stands_count = sum(1 for s in stands if s.get('cbp_eligible', '').upper() == 'YES')
    cbp_flights = [r for r in arrivals if str(r.get('cbp_flag', '')).strip().lower() in ('true', '1', 'yes')]

    # Calculate peak concurrent by 30-min slot (across all days, then divide for daily)
    slot_counts = defaultdict(int)
    for r in cbp_flights:
        h = _parse_hour(r.get('sta', '00:00'))
        turn = int(r.get('matched_turnaround_min', 60) or 60)
        for t in range(h * 60, h * 60 + turn, 30):
            slot_counts[t % (24 * 60) // 30] += 1  # normalise to slot-of-day

    # Average concurrent per slot
    daily_peak = round(max(slot_counts.values(), default=0) / n_days)
    avg_concurrent = round(sum(slot_counts.values()) / max(len(slot_counts), 1) / n_days, 1)
    idle_at_peak = max(cbp_stands_count - daily_peak, 0)

    band_counts = Counter()
    for r in cbp_flights:
        h = _parse_hour(r.get('sta', '00:00'))
        band_counts[_time_band(h)] += 1

    # Daily band counts
    band_daily = {k: round(v / n_days, 1) for k, v in band_counts.items()}

    return {
        'cbp_eligible_stands': cbp_stands_count,
        'daily_cbp_flights': round(len(cbp_flights) / n_days, 1),
        'daily_peak_concurrent': daily_peak,
        'daily_avg_concurrent': avg_concurrent,
        'idle_cbp_stands_at_peak': idle_at_peak,
        'by_time_band_daily': band_daily,
        'suggestion': f"Daily peak CBP demand is ~{daily_peak} simultaneous stands vs {cbp_stands_count} eligible. "
                      + (f"{idle_at_peak} CBP stands idle at peak — consider off-peak reclassification." if idle_at_peak > 2 else "CBP stands are near-fully utilised at peak.")
    }


# ─────────────────────────────────────────────
# STUDY 3 — MARS Efficiency (daily)
# ─────────────────────────────────────────────

def study_mars_efficiency(stands: List[Dict], arrivals: List[Dict]) -> Dict:
    n_days = _count_days(arrivals)
    stands_idx = {s['stand_id']: s for s in stands}
    mars_centers = {s['stand_id'] for s in stands if _is_mars_center(s)}

    narrow_on_center = 0
    wide_on_center   = 0
    for r in arrivals:
        gate     = r.get('gate', '').strip()
        cat      = r.get('icao_cat', 'C').strip().upper()
        resolved = _resolve_gate_to_stand(gate, stands_idx, cat)
        if resolved in mars_centers:
            if _is_narrowbody(cat):
                narrow_on_center += 1
            else:
                wide_on_center += 1

    total_center = narrow_on_center + wide_on_center
    waste_pct = round(narrow_on_center / max(total_center, 1) * 100, 1)

    return {
        'mars_center_stands': len(mars_centers),
        'daily_narrowbody_on_center': round(narrow_on_center / n_days, 1),
        'daily_widebody_on_center':   round(wide_on_center / n_days, 1),
        'narrowbody_waste_pct': waste_pct,
        'suggestion': f"{waste_pct}% of MARS Center occupancy historically was narrowbody. "
                      f"HC-6 constraint would free ~{round(narrow_on_center/n_days,1)} center-slots/day."
    }


# ─────────────────────────────────────────────
# STUDY 4 — Time-of-Day Heatmap (daily avg)
# ─────────────────────────────────────────────

def study_time_bands(stands: List[Dict], arrivals: List[Dict], departures: List[Dict]) -> Dict:
    n_days = _count_days(arrivals)
    gate_pier = _build_gate_pier_map(stands)
    piers = sorted(set(s['pier'] for s in stands))
    pier_contact = Counter(s['pier'] for s in stands if s.get('stand_type') == 'CONTACT')

    heatmap = defaultdict(lambda: defaultdict(int))  # pier -> band -> count (total)

    for r in list(arrivals) + list(departures):
        gate = r.get('gate', '').strip()
        pier = _resolve_pier(gate, gate_pier)
        h = _parse_hour(r.get('sta', r.get('std', '00:00')))
        band = _time_band(h)
        if pier:
            heatmap[pier][band] += 1

    result = {}
    for pier in piers:
        total_stands = max(pier_contact.get(pier, 1), 1)
        result[pier] = {}
        for label, _, _ in TIME_BANDS:
            total_mvmts = heatmap[pier].get(label, 0)
            daily_mvmts = round(total_mvmts / n_days, 1)
            # util: daily movements vs stands (crude throughput proxy)
            util = min(round(daily_mvmts / total_stands * 100), 100)
            result[pier][label] = {
                'daily_movements': daily_mvmts,
                'util_pct': util
            }

    return {'piers': piers, 'heatmap': result, 'n_days': n_days}


# ─────────────────────────────────────────────
# STUDY 5 — Pier Throughput (daily)
# ─────────────────────────────────────────────

def study_pier_throughput(stands: List[Dict], arrivals: List[Dict], departures: List[Dict]) -> Dict:
    n_days = _count_days(arrivals)
    gate_pier = _build_gate_pier_map(stands)
    pier_contacts = Counter(s['pier'] for s in stands if s.get('stand_type') == 'CONTACT')

    pier_movements = Counter()
    for r in list(arrivals) + list(departures):
        gate = r.get('gate', '').strip()
        pier = _resolve_pier(gate, gate_pier)
        if pier:
            pier_movements[pier] += 1

    result = []
    for pier, total in sorted(pier_movements.items(), key=lambda x: -x[1]):
        daily = round(total / n_days, 1)
        n_stands = max(pier_contacts.get(pier, 1), 1)
        mps = round(daily / n_stands, 1)
        result.append({
            'pier': pier,
            'daily_movements': daily,
            'contact_stands': pier_contacts.get(pier, 0),
            'movements_per_stand_per_day': mps,
            'avg_hrs_per_stand_per_turn': round(24 / max(mps, 0.1), 1),
            'status': 'Bottleneck' if mps > 6 else ('Efficient' if mps >= 2 else 'Underutilised')
        })
    return {'piers': result, 'n_days': n_days}


# ─────────────────────────────────────────────
# STUDY 6 — Historical Gate Affinity
# ─────────────────────────────────────────────

def study_gate_affinity(stands: List[Dict], arrivals: List[Dict]) -> Dict:
    n_days = _count_days(arrivals)
    stands_idx = {s['stand_id']: s for s in stands}
    affinity = defaultdict(Counter)

    for r in arrivals:
        code = r.get('flight_no', '')[:2]
        gate = r.get('gate', '').strip()
        cat  = r.get('icao_cat', 'C').strip().upper()
        resolved = _resolve_gate_to_stand(gate, stands_idx, cat)
        if code and resolved:
            affinity[code][resolved] += 1

    result = []
    for code, gate_counts in affinity.items():
        total = sum(gate_counts.values())
        top = [
            {'stand': g, 'daily_count': round(c / n_days, 1), 'confidence_pct': round(c / total * 100, 1)}
            for g, c in gate_counts.most_common(3)
        ]
        result.append({'airline_code': code, 'daily_movements': round(total / n_days, 1), 'top_stands': top})
    result.sort(key=lambda x: -x['daily_movements'])
    return {'affinity': result, 'n_days': n_days}


# ─────────────────────────────────────────────
# POLICY SUGGESTIONS — bottleneck / gap based only
# ─────────────────────────────────────────────

def generate_policy_suggestions(kpis: Dict, supply_demand, cbp, mars, pier_tp, stands: List[Dict] = None) -> List[Dict]:
    suggestions = []

    narrow_daily = kpis['daily_narrowbody']
    wide_cap     = kpis['widebody_capable_stands']
    narrow_only  = kpis['narrowbody_only_stands']
    wide_daily   = kpis['daily_widebody']
    cbp_peak     = cbp['daily_peak_concurrent']
    cbp_total    = kpis['cbp_eligible_stands']

    # 1. Cat C demand >> Cat C supply → reclassify nearby Cat D/E stands
    cat_c_demand = next((d['demand_daily'] for d in supply_demand['data'] if d['icao_cat'] == 'C'), 0)
    cat_c_stands = sum(1 for s in (stands or []) if s.get('icao_cat_max','').upper() == 'C')
    cat_d_stands = sum(1 for s in (stands or []) if s.get('icao_cat_max','').upper() in ('D','E','F'))
    if cat_c_demand > cat_c_stands * 5 and cat_d_stands > 10:
        suggestions.append({
            'type': 'warn', 'icon': '📐',
            'title': f'Cat C Gate Shortage — {cat_c_demand:.0f} flights/day, only {cat_c_stands} dedicated Cat C stands',
            'action': f'Reclassify {min(cat_d_stands // 3, 15)} proximate Cat D/E stands to Cat C — prioritise Pier 1 & Pier 3 contact gates.'
        })

    # 2. Widebody over-provisioned → activate MARS twin mode
    if wide_daily < wide_cap * 0.2:
        suggestions.append({
            'type': 'info', 'icon': '🛫',
            'title': f'Widebody Stands Over-Provisioned — {wide_daily} flights/day across {wide_cap} capable stands',
            'action': 'Activate MARS L/R twin mode off-peak to convert widebody centres into paired narrowbody slots.'
        })

    # 3. CBP idle gates
    if cbp_peak < cbp_total * 0.6:
        idle = cbp_total - cbp_peak
        suggestions.append({
            'type': 'info', 'icon': '🇺🇸',
            'title': f'{idle} CBP stands idle at daily peak — peak demand is only {cbp_peak} simultaneous',
            'action': 'Release idle CBP stands to general widebody use outside 07:00–11:00 AM Peak window.'
        })

    # 4. MARS narrowbody waste
    if mars['narrowbody_waste_pct'] > 10:
        suggestions.append({
            'type': 'warn', 'icon': '🔀',
            'title': f"{mars['narrowbody_waste_pct']}% of MARS Center usage was narrowbody — widebody slots wasted",
            'action': 'Enforce HC-6: hard ban narrowbodies from all MARS Center/Tail stands in the allocator.'
        })

    # 5. Bottleneck piers → move load to underutilised
    bottlenecks = [p for p in pier_tp['piers'] if p['status'] == 'Bottleneck']
    underused   = [p for p in pier_tp['piers'] if p['status'] == 'Underutilised']
    for p in bottlenecks:
        target = underused[0]['pier'] if underused else 'a less loaded pier'
        suggestions.append({
            'type': 'danger', 'icon': '🔥',
            'title': f"{p['pier']} overloaded — {p['movements_per_stand_per_day']} movements/stand/day across {p['contact_stands']} stands",
            'action': f'Redistribute narrowbody overflow from {p["pier"]} to {target} — {underused[0]["contact_stands"] if underused else ""} contact stands available.' if underused else f'Implement staggered scheduling at {p["pier"]} to reduce peak pressure.'
        })
    for p in underused:
        if p['daily_movements'] < 10:
            btarget = bottlenecks[0]['pier'] if bottlenecks else 'high-demand piers'
            suggestions.append({
                'type': 'info', 'icon': '📉',
                'title': f"{p['pier']} underutilised — {p['movements_per_stand_per_day']} movements/stand/day, {p['contact_stands']} stands available",
                'action': f'Route overflowing flights from {btarget} here — {p["contact_stands"]} contact stands with spare capacity.'
            })

    return suggestions


# ─────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────

def run_full_study(
    stands_path, arrivals_hist_path, departures_hist_path,
    arrivals_fwd_path, airline_terminal_path
) -> Dict:
    stands   = _load_csv(stands_path)
    arrivals = _load_csv(arrivals_hist_path)
    deps     = _load_csv(departures_hist_path)
    fwd      = _load_csv(arrivals_fwd_path)

    kpis     = compute_kpis(stands, arrivals, deps)
    sd       = study_supply_demand(stands, arrivals)
    cbp      = study_cbp_calibration(stands, fwd)
    mars     = study_mars_efficiency(stands, arrivals)
    tb       = study_time_bands(stands, arrivals, deps)
    pt       = study_pier_throughput(stands, arrivals, deps)
    aff      = study_gate_affinity(stands, arrivals)
    sug      = generate_policy_suggestions(kpis, sd, cbp, mars, pt, stands)

    return {
        'kpis': kpis,
        'supply_demand': sd,
        'cbp': cbp,
        'mars': mars,
        'time_bands': tb,
        'pier_throughput': pt,
        'gate_affinity': aff,
        'suggestions': sug,
    }
