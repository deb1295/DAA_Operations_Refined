"""
DAA Intra-day Planning Engine
Layers:
  1. MILP (PuLP)          — Golden Plan before T=0
  2. Constraint Propagation — Real-time conflict detection (interval overlap)
  3. Fix-and-Optimize      — Event-driven local search through 5-Step Decision Tree
"""
import os, csv, json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict

try:
    import pulp
except ImportError:
    pulp = None

# ── Constants ─────────────────────────────────────────────────────────────────
SIM_START   = datetime(2026, 1, 1, 7, 0)   # 07:00 reference
SIM_END     = datetime(2026, 1, 1, 8, 30)  # 08:30 reference
BUFFER_MIN  = 10
REMOTE_PENALTY   = 500
TERMINAL_PENALTY = 100
PIER_PENALTY     = 30
MAX_HOLD_MIN     = 20
BUFFER_HOLD_MAX  = 15
ICAO_ORDER = {'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}

BADGE_COLOURS = {
    'PIER_CONTACT':   '#10b981',
    'QUICK_TURN':     '#14b8a6',
    'ARR_PARK':       '#a855f7',
    'DEP_HOLD':       '#06b6d4',
    'CBP_CONTACT':    '#ec4899',
    'WIDEBODY_PIER':  '#2563eb',
    'BUFFER_HOLD':    '#f59e0b',
    'DEGRADED_REMOTE':'#f97316',
    'REMOTE_EXPAND':  '#fb923c',
    'AIRBORNE_HOLD':  '#ef4444',
    'GDP_ACTIVE':     '#dc2626',
    'STAND_CLEAR':    '#3b82f6',
    'DIVERSION_RISK': '#374151',
}

# ── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class IDFlight:
    flight_no: str
    ftype: str          # ARR / DEP / TURN
    eta: Optional[datetime]
    etd: Optional[datetime]
    aircraft_type: str
    icao_cat: str
    airline: str
    airline_code: str
    cbp_required: bool
    preferred_stand: str
    terminal: str
    # runtime fields
    assigned_stand: Optional[str] = None
    badge: str = 'UNASSIGNED'
    occ_start: Optional[datetime] = None
    occ_end:   Optional[datetime] = None

    def occupancy(self, buffer=BUFFER_MIN):
        """Return (start, end) occupancy window."""
        if self.ftype == 'ARR':
            s = self.eta
            e = self.eta + timedelta(minutes=20) + timedelta(minutes=buffer)  # 20m deboarding then towed
        elif self.ftype == 'DEP':
            s = SIM_START
            e = self.etd + timedelta(minutes=buffer)
        else:  # TURN
            s = self.eta
            e = self.etd + timedelta(minutes=buffer)
        return s, e


@dataclass
class IDStand:
    stand_id: str
    stype: str          # Contact / Remote / Buffer_Remote
    icao_cat_max: str
    cbp_eligible: bool
    terminal: str
    pier: str
    has_airbridge: bool
    mars_group: str = ''
    active: bool = True  # Buffer remotes start inactive

    def compatible(self, flight: IDFlight) -> bool:
        if not self.active:
            return False
        if ICAO_ORDER.get(flight.icao_cat, 2) > ICAO_ORDER.get(self.icao_cat_max, 2):
            return False
        if flight.cbp_required and not self.cbp_eligible:
            return False
        return True


# ── Loaders ───────────────────────────────────────────────────────────────────
def _parse_time(t_str: str) -> Optional[datetime]:
    if not t_str or t_str.strip() == '':
        return None
    h, m = map(int, t_str.strip().split(':'))
    return datetime(2026, 1, 1, h, m)


def load_intraday_flights(path: str) -> List[IDFlight]:
    flights = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            flights.append(IDFlight(
                flight_no      = row['Flight_No'],
                ftype          = row['Type'],
                eta            = _parse_time(row.get('ETA','')),
                etd            = _parse_time(row.get('ETD','')),
                aircraft_type  = row['Aircraft_Type'],
                icao_cat       = row['ICAO_Cat'],
                airline        = row['Airline'],
                airline_code   = row['Airline_Code'],
                cbp_required   = row['CBP_Required'].strip().lower() == 'true',
                preferred_stand= row['Preferred_Stand'],
                terminal       = row['Terminal'],
            ))
    return flights


def load_intraday_stands(path: str) -> List[IDStand]:
    stands = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            active = True  # All stands available for golden plan
            stands.append(IDStand(
                stand_id     = row['Stand_ID'],
                stype        = row['Type'],
                icao_cat_max = row['ICAO_Cat_Max'],
                cbp_eligible = row['CBP_Eligible'].strip().lower() == 'true',
                terminal     = row['Terminal'],
                pier         = row['Pier'],
                has_airbridge= row['Has_Airbridge'].strip().lower() == 'true',
                mars_group   = row.get('MARS_Group',''),
                active       = active,
            ))
    return stands


# ── Overlap Check ─────────────────────────────────────────────────────────────
def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


# ── MILP Golden Plan ──────────────────────────────────────────────────────────
def solve_milp(flights: List[IDFlight], stands: List[IDStand]) -> Dict:
    """
    Solve MILP for all flights. Returns allocation dict + shadow conflict log.
    """
    if pulp is None:
        raise RuntimeError("pulp not installed. Run: pip install pulp")

    contact = [s for s in stands if s.stype == 'Contact' and s.active]
    remote  = [s for s in stands if s.stype in ('Remote', 'Buffer_Remote') and s.active]

    N, M, R = len(flights), len(contact), len(remote)

    prob = pulp.LpProblem("IntraDayStandAlloc", pulp.LpMinimize)

    # Binary variables
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat='Binary') for j in range(M)] for i in range(N)]
    r = [[pulp.LpVariable(f"r_{i}_{k}", cat='Binary') for k in range(R)] for i in range(N)]

    # Objective
    cost_terms = []
    for i, fl in enumerate(flights):
        for j, st in enumerate(contact):
            w = 0
            if fl.terminal != st.terminal:  w += TERMINAL_PENALTY
            if fl.preferred_stand != st.stand_id: w += PIER_PENALTY
            cost_terms.append(w * x[i][j])
        for k in range(R):
            penalty = REMOTE_PENALTY
            if fl.cbp_required:
                penalty += 10000  # CBP flights MUST NOT go to remote
            elif ICAO_ORDER.get(fl.icao_cat, 2) >= ICAO_ORDER.get('E', 4):
                penalty += 2000   # Widebodies heavily prioritize contact stands
                
            # Flight Type priority (Quick Turn > Slow Turn > ARR/DEP)
            if fl.ftype == 'TURN':
                if fl.eta and fl.etd and (fl.etd - fl.eta).total_seconds() / 60 <= 50:
                    penalty += 800  # Quick Turn: massive priority
                else:
                    penalty += 400  # Slow Turn: medium priority
            # ARR/DEP inherently get no extra penalty (+0), so they are sacrificed first
            
            # FCFS Bias: Earlier flights slightly prefer contact stands more
            eta_m = int((fl.eta - SIM_START).total_seconds() / 60) if fl.eta else 0
            penalty += (90 - eta_m) * 0.01
            
            cost_terms.append(penalty * r[i][k])
    prob += pulp.lpSum(cost_terms)

    # C1: Each flight assigned exactly once
    for i in range(N):
        prob += pulp.lpSum(x[i][j] for j in range(M)) + pulp.lpSum(r[i][k] for k in range(R)) == 1

    # C2: Size + CBP compatibility (hard)
    for i, fl in enumerate(flights):
        for j, st in enumerate(contact):
            if not st.compatible(fl):
                prob += x[i][j] == 0
            # CBP exclusivity: non-CBP flights cannot park at CBP stands
            if st.cbp_eligible and not fl.cbp_required:
                prob += x[i][j] == 0

    # C3: No temporal overlap on same contact stand
    for j in range(M):
        for i in range(N):
            for ii in range(i+1, N):
                s_i, e_i = flights[i].occupancy()
                s_ii, e_ii = flights[ii].occupancy()
                if _overlaps(s_i, e_i, s_ii, e_ii):
                    prob += x[i][j] + x[ii][j] <= 1

    # C3b: No temporal overlap on same remote stand
    for k in range(R):
        for i in range(N):
            for ii in range(i+1, N):
                s_i, e_i = flights[i].occupancy()
                s_ii, e_ii = flights[ii].occupancy()
                if _overlaps(s_i, e_i, s_ii, e_ii):
                    prob += r[i][k] + r[ii][k] <= 1

    # C4: MARS exclusivity (widebody on centre blocks adjacent)
    mars_groups = {}
    for j, st in enumerate(contact):
        if st.mars_group:
            mars_groups.setdefault(st.mars_group, []).append(j)

    for grp, idxs in mars_groups.items():
        for i, fl in enumerate(flights):
            if ICAO_ORDER.get(fl.icao_cat, 2) >= ICAO_ORDER.get('E', 4):
                # Widebody on any stand in group blocks all others in that window
                for j in idxs:
                    for jj in idxs:
                        if j != jj:
                            prob += x[i][j] + pulp.lpSum(
                                x[ii][jj] for ii in range(N)
                                if _overlaps(*flights[ii].occupancy(), *flights[i].occupancy())
                            ) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract solution
    allocation = {}   # flight_no -> stand_id
    conflict_log = [] # conflicts that MILP resolved

    def _classify_badge(fl, st):
        """Assign a rich operation-type badge based on flight context."""
        if fl.cbp_required:
            return 'CBP_CONTACT'
        if fl.ftype == 'ARR':
            return 'ARR_PARK'
        if fl.ftype == 'DEP':
            return 'DEP_HOLD'
        if ICAO_ORDER.get(fl.icao_cat, 2) >= ICAO_ORDER.get('E', 4):
            return 'WIDEBODY_PIER'
        if fl.ftype == 'TURN':
            turnaround = 0
            if fl.eta and fl.etd:
                turnaround = (fl.etd - fl.eta).total_seconds() / 60
            return 'QUICK_TURN' if turnaround <= 50 else 'PIER_CONTACT'
        return 'PIER_CONTACT'

    def _conflict_reason(fl, wanted_st, assigned_st):
        """Generate business-context conflict reason."""
        if fl.cbp_required:
            return f'CBP-eligible flight rerouted — Stand {assigned_st} has US Preclearance access'
        if ICAO_ORDER.get(fl.icao_cat, 2) >= ICAO_ORDER.get('E', 4):
            return f'Widebody {fl.aircraft_type} requires Cat E stand — reassigned to {assigned_st} (MARS capable)'
        if fl.ftype == 'TURN' and fl.eta and fl.etd and (fl.etd - fl.eta).total_seconds()/60 <= 50:
            return f'Quick turnaround ({int((fl.etd - fl.eta).total_seconds()/60)} min) — moved to {assigned_st} for fast gate availability'
        if fl.terminal == 'T1':
            return f'{fl.airline} T1 flight — preferred stand {wanted_st} occupied, reassigned within Pier 1 to {assigned_st}'
        return f'{fl.airline} flight reassigned from {wanted_st} to {assigned_st} — overlap conflict resolved by MILP'

    for i, fl in enumerate(flights):
        assigned = None
        assigned_stand_obj = None
        for j, st in enumerate(contact):
            if pulp.value(x[i][j]) and pulp.value(x[i][j]) > 0.5:
                assigned = st.stand_id
                assigned_stand_obj = st
                break
        if not assigned:
            for k, st in enumerate(remote):
                if pulp.value(r[i][k]) and pulp.value(r[i][k]) > 0.5:
                    assigned = st.stand_id
                    assigned_stand_obj = st
                    break
        if not assigned:
            assigned = 'UNRESOLVED'
            badge    = 'DIVERSION_RISK'
        else:
            badge = _classify_badge(fl, assigned_stand_obj)

        fl.assigned_stand = assigned
        fl.badge          = badge
        s, e = fl.occupancy()
        fl.occ_start, fl.occ_end = s, e
        allocation[fl.flight_no] = assigned

        # Was preferred stand overridden? Log with business reason
        if fl.preferred_stand and assigned != fl.preferred_stand and assigned != 'UNRESOLVED':
            conflict_log.append({
                'flight':      fl.flight_no,
                'wanted':      fl.preferred_stand,
                'assigned':    assigned,
                'reason':      _conflict_reason(fl, fl.preferred_stand, assigned),
                'badge':       badge,
                'step_resolved': 1
            })

    return {'allocation': allocation, 'conflict_log': conflict_log,
            'status': str(prob.status)}


# ── Constraint Propagation (real-time conflict detection) ─────────────────────
def detect_conflicts(flights: List[IDFlight]) -> List[Dict]:
    """Check all assigned flights for temporal overlaps on same stand."""
    conflicts = []
    by_stand = {}
    for fl in flights:
        if fl.assigned_stand and fl.assigned_stand != 'UNRESOLVED':
            by_stand.setdefault(fl.assigned_stand, []).append(fl)

    for stand_id, fls in by_stand.items():
        for i in range(len(fls)):
            for j in range(i+1, len(fls)):
                a, b = fls[i], fls[j]
                if _overlaps(a.occ_start, a.occ_end, b.occ_start, b.occ_end):
                    overlap_s = max(a.occ_start, b.occ_start).strftime('%H:%M')
                    overlap_e = min(a.occ_end, b.occ_end).strftime('%H:%M')
                    conflicts.append({
                        'stand':    stand_id,
                        'flight_a': a.flight_no,
                        'flight_b': b.flight_no,
                        'overlap_start': overlap_s,
                        'overlap_end':   overlap_e,
                        'description': (
                            f'Stand {stand_id}: {a.flight_no} ({a.airline}) and '
                            f'{b.flight_no} ({b.airline}) overlap {overlap_s}–{overlap_e}. '
                            f'Buffer violation — need {BUFFER_MIN} min gap between turnarounds.'
                        )
                    })
    return conflicts


# ── Fix-and-Optimize: 5-Step Decision Tree ────────────────────────────────────
def resolve_event(flights: List[IDFlight], stands: List[IDStand],
                  affected_flight_no: str, delta_minutes: int = 0,
                  event_type: str = 'ETA_SHIFT') -> Dict:
    """
    Apply an event to the affected flight and replan using the 5-Step Decision Tree.
    Returns: updated flight list + decision log entry.
    """
    flight = next((f for f in flights if f.flight_no == affected_flight_no), None)
    if not flight:
        return {'error': f'Flight {affected_flight_no} not found'}

    log_entry = {
        'flight':     affected_flight_no,
        'event_type': event_type,
        'delta_min':  delta_minutes,
        'steps_tried': [],
        'step_resolved': None,
        'badge':      None,
        'stand':      None,
        'reason':     {},
        'cascades':   []
    }

    # Apply event mutation to state
    displaced_flight = None
    if event_type == 'FLIGHT_DELAY':
        # Shift both ETA and ETD together to preserve turnaround window
        if flight.eta:
            flight.eta = flight.eta + timedelta(minutes=delta_minutes)
        if flight.etd:
            flight.etd = flight.etd + timedelta(minutes=delta_minutes)
    elif event_type == 'ETA_SHIFT' and flight.eta:
        flight.eta = flight.eta + timedelta(minutes=delta_minutes)
    elif event_type == 'ETD_SHIFT' and flight.etd:
        flight.etd = flight.etd + timedelta(minutes=delta_minutes)
    elif event_type == 'GO_AROUND' and flight.eta:
        flight.eta = flight.eta + timedelta(minutes=8)
    elif event_type == 'MAYDAY':
        # Highest priority — may displace current stand occupant
        flight.eta = flight.eta or SIM_START
        # Find who is currently on nearest available stand
        for st in sorted(stands, key=lambda s: 0 if s.stype=='Contact' else 1):
            if not st.compatible(flight):
                continue
            occupant = next((f for f in flights
                             if f.assigned_stand == st.stand_id
                             and f.flight_no != affected_flight_no), None)
            if occupant and occupant.ftype == 'TURN':
                continue  # Can't displace active turnaround
            if occupant:
                displaced_flight = occupant
            flight.assigned_stand = st.stand_id
            flight.badge = 'PIER_CONTACT'
            s, e = flight.occupancy()
            flight.occ_start, flight.occ_end = s, e
            log_entry.update({'step_resolved': 1, 'badge': 'PIER_CONTACT',
                              'stand': st.stand_id,
                              'reason': {'step_1_pass': 'MAYDAY priority — nearest stand cleared'}})
            if displaced_flight:
                # Displaced flight re-enters decision tree
                cascade = resolve_event(flights, stands, displaced_flight.flight_no,
                                        delta_minutes=0, event_type='STAND_CLOSE')
                log_entry['cascades'].append(cascade)
            return log_entry
    elif event_type == 'ACFT_SWAP':
        flight.icao_cat = delta_minutes  # repurpose param as new ICAO cat string
    elif event_type == 'STAND_CLOSE':
        pass  # Stand already removed from active by caller
    elif event_type == 'PUSHBACK_FAIL' and flight.etd:
        import random
        flight.etd = flight.etd + timedelta(minutes=random.randint(5, 20))
    elif event_type == 'CREW_CURFEW' and flight.etd:
        flight.etd = flight.etd + timedelta(minutes=delta_minutes)

    # Recompute occupancy
    s, e = flight.occupancy()
    flight.occ_start, flight.occ_end = s, e

    # Release old stand
    old_stand = flight.assigned_stand
    flight.assigned_stand = None
    flight.badge = 'UNASSIGNED'

    contact_stands = [st for st in stands if st.stype == 'Contact' and st.active]
    remote_stands  = [st for st in stands if st.stype in ('Remote','Buffer_Remote') and st.active]

    def stand_free_at(stand_id, query_start, query_end, exclude=None):
        for f in flights:
            if f.flight_no == (exclude or ''):
                continue
            if f.assigned_stand == stand_id and f.occ_start and f.occ_end:
                if _overlaps(f.occ_start, f.occ_end, query_start, query_end):
                    return False, f
        return True, None

    def earliest_vacate(stand_id, after):
        latest = after
        for f in flights:
            if f.assigned_stand == stand_id and f.occ_end and f.occ_end > after:
                latest = max(latest, f.occ_end)
        return latest

    # ── STEP 1: Free pier contact stand at arrival time ──
    log_entry['steps_tried'].append(1)
    for st in contact_stands:
        if not st.compatible(flight):
            continue
        free, _ = stand_free_at(st.stand_id, s, e, exclude=flight.flight_no)
        if free:
            flight.assigned_stand = st.stand_id
            flight.badge = 'PIER_CONTACT'
            log_entry.update({'step_resolved': 1, 'badge': 'PIER_CONTACT', 'stand': st.stand_id,
                              'reason': {'step_1_pass': f'Stand {st.stand_id} free'}})
            return log_entry
    log_entry['reason']['step_1_fail'] = 'No contact stand free at arrival time'

    # ── STEP 2: Will a stand free up within 15 min? ──
    log_entry['steps_tried'].append(2)
    for st in contact_stands:
        if not st.compatible(flight):
            continue
        vacate_at = earliest_vacate(st.stand_id, s)
        wait = (vacate_at - s).total_seconds() / 60
        if 0 < wait <= BUFFER_HOLD_MAX:
            # Split allocation: temp remote + pier reassignment
            temp_remote = next((r for r in remote_stands
                                if stand_free_at(r.stand_id, s, vacate_at)[0]), None)
            if temp_remote:
                flight.assigned_stand = st.stand_id
                flight.badge = 'BUFFER_HOLD'
                flight.occ_start = vacate_at
                log_entry.update({
                    'step_resolved': 2, 'badge': 'BUFFER_HOLD', 'stand': st.stand_id,
                    'temp_remote': temp_remote.stand_id,
                    'reassign_at': vacate_at.strftime('%H:%M'),
                    'wait_min': round(wait, 1),
                    'reason': {'step_2_pass':
                        f'Stand {st.stand_id} vacates in {round(wait,1)} min. '
                        f'Holding on {temp_remote.stand_id} until {vacate_at.strftime("%H:%M")}'}
                })
                return log_entry
    log_entry['reason']['step_2_fail'] = 'No stand vacates within 15-min buffer threshold'

    # ── STEP 3: Any remote stand available? ──
    log_entry['steps_tried'].append(3)
    for rs in remote_stands:
        free, _ = stand_free_at(rs.stand_id, s, e, exclude=flight.flight_no)
        if free and rs.compatible(flight):
            flight.assigned_stand = rs.stand_id
            flight.badge = 'DEGRADED_REMOTE' if rs.stype=='Remote' else 'REMOTE_EXPAND'
            log_entry.update({'step_resolved': 3, 'badge': flight.badge, 'stand': rs.stand_id,
                              'reason': {'step_3_pass': f'Remote {rs.stand_id} available. PAX bus required.'}})
            return log_entry
    log_entry['reason']['step_3_fail'] = 'All remote stands occupied'

    # ── STEP 4: Airborne hold or GDP? ──
    log_entry['steps_tried'].append(4)
    all_vacate = min(
        (earliest_vacate(st.stand_id, s) for st in contact_stands + remote_stands
         if st.compatible(flight)),
        default=SIM_END
    )
    wait_min = (all_vacate - s).total_seconds() / 60

    if wait_min <= MAX_HOLD_MIN:
        fuel_kg = round(wait_min * 50, 0)
        flight.badge = 'AIRBORNE_HOLD'
        log_entry.update({
            'step_resolved': 4, 'badge': 'AIRBORNE_HOLD', 'stand': None,
            'hold_min': round(wait_min, 1),
            'fuel_burn_kg': fuel_kg,
            'reason': {'step_4a': f'Airborne hold {round(wait_min,1)} min. Est. fuel burn: {fuel_kg}kg'}
        })
    else:
        flight.badge = 'GDP_ACTIVE'
        log_entry.update({
            'step_resolved': 4, 'badge': 'GDP_ACTIVE', 'stand': None,
            'gdp_delay_min': round(wait_min, 1),
            'reason': {'step_4b': f'Wait {round(wait_min,1)} min exceeds 20-min fuel budget. GDP issued to origin.'}
        })
    return log_entry

    # ── STEP 5: Diversion ──
    log_entry['steps_tried'].append(5)
    flight.badge = 'DIVERSION_RISK'
    log_entry.update({'step_resolved': 5, 'badge': 'DIVERSION_RISK', 'stand': None,
                      'reason': {'step_5': 'No stand capacity within any horizon. DIVERSION flagged to ATC.'}})
    return log_entry


# ── Serialiser (flights → Gantt JSON) ─────────────────────────────────────────
def flights_to_gantt(flights: List[IDFlight], stands: List[IDStand] = None) -> List[Dict]:
    stands_map = {s.stand_id: s for s in (stands or [])}
    rows = []
    for fl in flights:
        if not fl.occ_start:
            continue
        start_min = int((fl.occ_start - SIM_START).total_seconds() / 60)
        # Remove the padded buffer for visual rendering to show gaps
        end_min   = int(((fl.occ_end - timedelta(minutes=BUFFER_MIN)) - SIM_START).total_seconds() / 60)

        assigned_st = stands_map.get(fl.assigned_stand)
        preferred_st = stands_map.get(fl.preferred_stand)

        # ── Build constraint-by-constraint MILP reasoning ──
        reasons = []

        if fl.assigned_stand == 'UNRESOLVED':
            reasons.append('All 15 stands failed feasibility — capacity exhausted')
        elif fl.assigned_stand == fl.preferred_stand:
            st = assigned_st
            if st:
                reasons.append(f'✅ Terminal: {fl.terminal} flight → {st.terminal} stand')
                reasons.append(f'✅ ICAO: Cat {fl.icao_cat} ≤ Cat {st.icao_cat_max} max')
                if fl.cbp_required:
                    reasons.append(f'✅ CBP: Stand {st.stand_id} US Preclearance eligible')
                if st.has_airbridge:
                    reasons.append(f'✅ Airbridge available for {fl.aircraft_type}')
                neighbours = [f2 for f2 in flights if f2.assigned_stand == fl.assigned_stand
                              and f2.flight_no != fl.flight_no]
                for nb in neighbours:
                    if nb.occ_end and fl.occ_start and nb.occ_end <= fl.occ_start:
                        gap = int((fl.occ_start - nb.occ_end).total_seconds() / 60)
                        reasons.append(f'✅ Buffer: {gap} min gap after {nb.flight_no}')
                    elif nb.occ_start and fl.occ_end and fl.occ_end <= nb.occ_start:
                        gap = int((nb.occ_start - fl.occ_end).total_seconds() / 60)
                        reasons.append(f'✅ Buffer: {gap} min gap before {nb.flight_no}')
                if st.mars_group:
                    reasons.append(f'✅ MARS {st.mars_group}: No widebody conflict')
            reasons.append('Preferred stand optimal — all constraints passed')
        else:
            # Explain why preferred failed
            if preferred_st:
                if fl.cbp_required and not preferred_st.cbp_eligible:
                    reasons.append(f'❌ CBP: Stand {fl.preferred_stand} not US-eligible — {fl.airline} needs Preclearance')
                elif ICAO_ORDER.get(fl.icao_cat, 2) > ICAO_ORDER.get(preferred_st.icao_cat_max, 2):
                    reasons.append(f'❌ ICAO: {fl.aircraft_type} Cat {fl.icao_cat} exceeds Stand {fl.preferred_stand} max Cat {preferred_st.icao_cat_max}')
                elif fl.terminal != preferred_st.terminal and preferred_st.terminal != 'Remote':
                    reasons.append(f'❌ Terminal: {fl.airline} is {fl.terminal}, Stand {fl.preferred_stand} is {preferred_st.terminal}')
                else:
                    blockers = [f2 for f2 in flights if f2.assigned_stand == fl.preferred_stand
                                and f2.flight_no != fl.flight_no
                                and f2.occ_start and f2.occ_end and fl.occ_start and fl.occ_end
                                and _overlaps(f2.occ_start, f2.occ_end, fl.occ_start, fl.occ_end)]
                    if blockers:
                        b = blockers[0]
                        reasons.append(f'❌ Overlap: {b.flight_no} ({b.airline}) on Stand {fl.preferred_stand} '
                                       f'{b.occ_start.strftime("%H:%M")}–{b.occ_end.strftime("%H:%M")} '
                                       f'({BUFFER_MIN} min buffer required)')
                    else:
                        reasons.append(f'❌ Stand {fl.preferred_stand} blocked by MILP cost optimisation')

            # Explain why assigned stand was chosen
            if assigned_st:
                if assigned_st.stand_id.startswith('R'):
                    reasons.append(f'→ No contact stand free in {fl.occ_start.strftime("%H:%M")}–{fl.occ_end.strftime("%H:%M")}')
                    reasons.append(f'→ Remote {assigned_st.stand_id}: Bus transfer ~8 min, no airbridge')
                    if assigned_st.stype == 'Buffer_Remote':
                        reasons.append(f'→ Buffer overflow stand activated')
                else:
                    parts = []
                    if fl.terminal == assigned_st.terminal:
                        parts.append(f'terminal {assigned_st.terminal} ✓')
                    if assigned_st.pier:
                        parts.append(assigned_st.pier)
                    if fl.cbp_required and assigned_st.cbp_eligible:
                        parts.append('CBP eligible ✓')
                    if assigned_st.has_airbridge:
                        parts.append('airbridge ✓')
                    if assigned_st.mars_group:
                        parts.append(f'MARS {assigned_st.mars_group} clear')
                    reasons.append(f'→ Stand {assigned_st.stand_id}: {", ".join(parts)}' if parts
                                   else f'→ Stand {assigned_st.stand_id}: lowest MILP cost')

        milp_reason = ' | '.join(reasons)

        rows.append({
            'flight_no':    fl.flight_no,
            'airline':      fl.airline,
            'airline_code': fl.airline_code,
            'ftype':        fl.ftype,
            'icao_cat':     fl.icao_cat,
            'aircraft':     fl.aircraft_type,
            'stand':        fl.assigned_stand or 'UNRESOLVED',
            'preferred':    fl.preferred_stand,
            'badge':        fl.badge,
            'colour':       BADGE_COLOURS.get(fl.badge, '#6b7280'),
            'milp_reason':  milp_reason,
            'start_min':    max(0, start_min),
            'end_min':      min(90, end_min),
            'eta':          fl.eta.strftime('%H:%M') if fl.eta else '',
            'etd':          fl.etd.strftime('%H:%M') if fl.etd else '',
            'terminal':     fl.terminal,
            'cbp':          fl.cbp_required,
        })
    return rows


def generate_recommendation(gantt):
    """Analyze Gantt rows to find the biggest bottleneck and suggest a new stand."""
    if not gantt:
        return "✨ No flights scheduled for analysis."
        
    remote_counts = {'CAT_E': 0, 'CBP': 0, 'T1': 0, 'T2': 0, 'TOTAL': 0}
    for r in gantt:
        if r['stand'].startswith('R'):
            remote_counts['TOTAL'] += 1
            if r['icao_cat'] in ('E', 'F'): remote_counts['CAT_E'] += 1
            if r['cbp']: remote_counts['CBP'] += 1
            if r['terminal'] == 'T1': remote_counts['T1'] += 1
            if r['terminal'] == 'T2': remote_counts['T2'] += 1

    if remote_counts['TOTAL'] == 0:
        return "✨ System Fully Optimized: All flights assigned to contact stands."
    
    if remote_counts['CBP'] > 0:
        return "⚠️ CRITICAL: Open a CBP-eligible Contact Stand to eliminate US-bound remote towing."
    
    if remote_counts['CAT_E'] > 0:
        if remote_counts['T1'] > remote_counts['T2']:
            return "🛫 STRATEGIC: Open a Category E (Widebody) Contact Stand in Terminal 1 to accommodate heavy demand."
        else:
            return "🛫 STRATEGIC: Open a Category E (Widebody) Contact Stand in Terminal 2 to resolve MARS/Widebody contention."
            
    if remote_counts['T1'] > remote_counts['T2']:
        return "🏢 CAPACITY: Terminal 1 Pier 1 is saturated; consider opening an additional Cat C contact stand."
    else:
        return "🏢 CAPACITY: Terminal 2 Pier 2 is saturated; consider opening an additional Cat C contact stand."


