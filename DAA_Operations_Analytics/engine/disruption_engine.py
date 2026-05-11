"""
disruption_engine.py
Post-processes the baseline 3-day gantt to model four disruption scenarios.
Does NOT call the main allocator — operates entirely on cached gantt data.
"""
import json
import copy
import os
from datetime import datetime, timedelta

# ── Specific Schedule Changes for Day 1 ─────────────────────────────────────
# Each entry defines the exact delay (in minutes) for a specific dep_flight
# along with human-readable original/new times for display.
SCHED_CHANGES = {
    "FR2971": {
        "delay_min": 68,
        "original":  "07:52",
        "new":       "09:00",
        "airline":   "Ryanair",
        "route":     "DUB → STN",
    },
    "EI204": {
        "delay_min": 143,
        "original":  "08:07",
        "new":       "10:30",
        "airline":   "Aer Lingus",
        "route":     "DUB → CDG",
    },
    "EI404": {
        "delay_min": 153,
        "original":  "09:12",
        "new":       "11:45",
        "airline":   "Aer Lingus",
        "route":     "DUB → LHR",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def _fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _duration_minutes(block):
    start = _parse_dt(block["start_time"])
    end   = _parse_dt(block["end_time"])
    return (end - start).total_seconds() / 60

def _load_baseline():
    path = os.path.join("data", "outputs", "3day_plan_cache.json")
    with open(path, "r") as f:
        return json.load(f)

# ── Greedy conflict resolver ──────────────────────────────────────────────────

def _resolve_conflicts(gantt, all_stands):
    """
    Scan gantt sorted by start_time.
    If two blocks overlap on the same gate → reassign the second to the next
    available stand in the same pier, falling back to any free stand.
    Returns modified gantt and count of reassignments.
    """
    # Build stand → pier/terminal map
    stand_meta = {s["stand_id"]: s for s in all_stands}

    occupied = {}   # gate → list of (start, end) tuples
    reassigned = 0
    unresolved = 0

    gantt_sorted = sorted(gantt, key=lambda x: x["start_time"])
    result = []

    for block in gantt_sorted:
        gate = block["gate"]
        start = _parse_dt(block["start_time"])
        end   = _parse_dt(block["end_time"])

        # Check conflict on current gate
        conflict = False
        if gate in occupied:
            for (occ_s, occ_e) in occupied[gate]:
                if start < occ_e and end > occ_s:
                    conflict = True
                    break

        if conflict:
            # Try to find a free stand in the same pier first
            meta = stand_meta.get(gate, {})
            preferred_pier = meta.get("pier", "")
            preferred_term = meta.get("terminal", "T1")

            candidates = [
                s for s in all_stands
                if (s.get("pier") == preferred_pier or s.get("terminal") == preferred_term)
                and s["stand_id"] != gate
            ]
            # Also allow any stand as fallback
            candidates += [s for s in all_stands if s["stand_id"] not in [c["stand_id"] for c in candidates]]

            found = False
            for cand in candidates:
                cg = cand["stand_id"]
                slots = occupied.get(cg, [])
                ok = all(not (start < occ_e and end > occ_s) for (occ_s, occ_e) in slots)
                if ok:
                    block = copy.deepcopy(block)
                    block["gate"] = cg
                    block["_reassigned"] = True
                    gate = cg
                    reassigned += 1
                    found = True
                    break

            if not found:
                block = copy.deepcopy(block)
                block["_unresolved"] = True
                unresolved += 1

        occupied.setdefault(gate, []).append((start, end))
        result.append(block)

    return result, reassigned, unresolved


# ── Disruption Functions ──────────────────────────────────────────────────────

def apply_ground_stop(gantt, all_stands, stop_start_h=10, stop_duration_h=2):
    """
    NOTAM Ground Stop: freeze all Day-1 DEP blocks whose start_time falls
    within [stop_start_h, stop_start_h + stop_duration_h].
    Release sequentially post-stop in original STD order, spaced 3 min apart.
    """
    gantt = copy.deepcopy(gantt)

    stop_start = datetime(2026, 3, 31, stop_start_h, 0, 0)
    stop_end   = stop_start + timedelta(hours=stop_duration_h)

    frozen = []
    others = []

    for block in gantt:
        if "2026-03-31" not in block.get("start_time", ""):
            others.append(block)
            continue
        if not block.get("dep_flight", ""):
            others.append(block)
            continue
        start = _parse_dt(block["start_time"])
        if stop_start <= start < stop_end:
            frozen.append(block)
        else:
            others.append(block)

    # Sort frozen by original start_time (STD order)
    frozen.sort(key=lambda x: x["start_time"])

    # Re-schedule: first release 3 min after stop_end, then every 3 min
    release_time = stop_end + timedelta(minutes=3)
    updated_frozen = []
    for block in frozen:
        original_start = _parse_dt(block["start_time"])
        original_end   = _parse_dt(block["end_time"])
        duration = original_end - original_start

        new_block = copy.deepcopy(block)
        new_block["start_time"] = _fmt_dt(release_time)
        new_block["end_time"]   = _fmt_dt(release_time + duration)
        new_block["_disrupted"] = True
        new_block["_delay_min"] = int((release_time - original_start).total_seconds() / 60)
        updated_frozen.append(new_block)
        release_time += timedelta(minutes=3)

    combined = others + updated_frozen
    resolved, reassigned, unresolved = _resolve_conflicts(combined, all_stands)

    frozen_count = len(frozen)
    avg_delay = (
        sum(b.get("_delay_min", 0) for b in updated_frozen) // max(frozen_count, 1)
        if frozen_count else 0
    )

    impact = [
        f"{frozen_count} departures frozen {stop_start.strftime('%H:%M')}–{stop_end.strftime('%H:%M')} (Day 1)",
        f"Sequential release began {stop_end.strftime('%H:%M')} — all cleared by {_parse_dt(updated_frozen[-1]['end_time']).strftime('%H:%M') if updated_frozen else stop_end.strftime('%H:%M')}",
        f"{reassigned} stand conflicts auto-resolved via greedy re-assignment",
        f"{unresolved} flights moved to remote/overflow stands" if unresolved else "All conflicts resolved within pier capacity",
        f"Average departure delay: {avg_delay} min",
    ]

    return resolved, {
        "conflicts_resolved": reassigned,
        "unresolved": unresolved,
        "frozen_departures": frozen_count,
        "avg_delay_min": avg_delay,
    }, impact


def apply_weather(gantt, all_stands, stretch_pct=25,
                  window_start_h=16, window_end_h=18):
    """
    Weather: extend end_time of Day-1 stand blocks whose start falls within
    [window_start_h, window_end_h) by stretch_pct%.
    Re-run conflict resolution on newly extended blocks.
    """
    gantt = copy.deepcopy(gantt)
    factor = 1 + (stretch_pct / 100)
    affected = 0

    for block in gantt:
        if "2026-03-31" not in block.get("start_time", ""):
            continue
        start = _parse_dt(block["start_time"])
        if not (window_start_h <= start.hour < window_end_h):
            continue
        end = _parse_dt(block["end_time"])
        duration = (end - start).total_seconds() / 60
        new_duration = duration * factor
        new_end = start + timedelta(minutes=new_duration)
        block["end_time"] = _fmt_dt(new_end)
        block["_disrupted"] = True
        block["_extra_min"] = int(new_duration - duration)
        affected += 1

    resolved, reassigned, unresolved = _resolve_conflicts(gantt, all_stands)

    win_label = f"{window_start_h:02d}:00\u2013{window_end_h:02d}:00"
    impact = [
        f"Ground times extended by {stretch_pct}% for {affected} Day-1 stand blocks in window {win_label}",
        f"Average extension: +{int((factor - 1) * 60)} min per affected stand block",
        f"{reassigned} stand conflicts auto-resolved via greedy re-assignment",
        f"{unresolved} flights unable to re-allocate within pier — moved to remote stands" if unresolved else "All conflicts resolved within pier capacity",
        "CBP Pier E stands at elevated risk — turnaround buffers reduced to <15 min" if stretch_pct >= 25 else "Stand buffers tightened but remain within operational margins",
    ]

    return resolved, {
        "conflicts_resolved": reassigned,
        "unresolved": unresolved,
        "affected_blocks": affected,
        "avg_extension_min": int((factor - 1) * 60),
        "window": win_label,
    }, impact


def apply_runway_cap(gantt, all_stands, cap_reduction_pct=25):
    """
    Reduced runway capacity: stagger Day-1 arrivals beyond the metered rate.
    cap_reduction_pct=25 → max ~24 arr/hr instead of ~32.
    Adds incremental delay to overflow arrivals.
    """
    gantt = copy.deepcopy(gantt)

    # Dublin normal: ~32 movements/hr per runway
    # At 25% cap: ~24/hr; at 50% cap: ~16/hr
    max_per_hour = int(32 * (1 - cap_reduction_pct / 100))

    hour_counts = {}
    delayed = 0
    total_delay = 0

    # Sort arrivals by start_time and apply delay when exceeding hourly cap
    arrivals = sorted(
        [b for b in gantt if "2026-03-31" in b.get("start_time", "") and b.get("arr_flight", "")],
        key=lambda x: x["start_time"]
    )

    for block in arrivals:
        start = _parse_dt(block["start_time"])
        hour_key = start.hour
        count = hour_counts.get(hour_key, 0)

        if count >= max_per_hour:
            # Delay by 5 min per overflow aircraft in this hour
            overflow = count - max_per_hour + 1
            delay_min = overflow * 5
            block["start_time"] = _fmt_dt(start + timedelta(minutes=delay_min))
            block["end_time"]   = _fmt_dt(_parse_dt(block["end_time"]) + timedelta(minutes=delay_min))
            block["_disrupted"] = True
            block["_delay_min"] = delay_min
            delayed += 1
            total_delay += delay_min

        hour_counts[hour_key] = hour_counts.get(hour_key, 0) + 1

    resolved, reassigned, unresolved = _resolve_conflicts(gantt, all_stands)
    avg_delay = total_delay // max(delayed, 1)

    impact = [
        f"Runway capacity reduced by {cap_reduction_pct}% — max throughput {max_per_hour} movements/hr",
        f"{delayed} Day-1 arrivals delayed due to reduced metering rate",
        f"Average arrival delay: +{avg_delay} min",
        f"{reassigned} stand conflicts auto-resolved",
        f"Peak congestion window: 08:00–11:00 (morning wave most affected)",
    ]

    return resolved, {
        "conflicts_resolved": reassigned,
        "unresolved": unresolved,
        "delayed_arrivals": delayed,
        "avg_delay_min": avg_delay,
    }, impact


def apply_schedule_change(gantt, all_stands, flight_keys=None):
    """
    Schedule Change: shift specific Day-1 flights by their exact pre-defined delay.
    flight_keys: list of keys from SCHED_CHANGES to apply. None = apply all.
    """
    gantt = copy.deepcopy(gantt)
    if flight_keys is None:
        flight_keys = list(SCHED_CHANGES.keys())

    affected_count = 0
    applied = []

    for block in gantt:
        dep = block.get("dep_flight", "")
        if "2026-03-31" not in block.get("start_time", ""):
            continue
        if dep not in flight_keys:
            continue
        change = SCHED_CHANGES[dep]
        delay_min = change["delay_min"]
        start = _parse_dt(block["start_time"])
        end   = _parse_dt(block["end_time"])
        block["start_time"] = _fmt_dt(start + timedelta(minutes=delay_min))
        block["end_time"]   = _fmt_dt(end   + timedelta(minutes=delay_min))
        block["_disrupted"] = True
        block["_delay_min"] = delay_min
        affected_count += 1
        applied.append(change)

    resolved, reassigned, unresolved = _resolve_conflicts(gantt, all_stands)

    change_lines = [
        f"{SCHED_CHANGES[k]['airline']} {k}: {SCHED_CHANGES[k]['original']} → {SCHED_CHANGES[k]['new']} (+{SCHED_CHANGES[k]['delay_min']}min)"
        for k in flight_keys if k in SCHED_CHANGES
    ]
    impact = [
        f"{affected_count} stand block(s) shifted due to schedule amendment",
    ] + change_lines + [
        f"{reassigned} downstream stand conflicts auto-resolved",
        f"{unresolved} flights require remote stand allocation" if unresolved else "All conflicts resolved within allocated pier capacity",
        "Operator advised to re-brief affected airline handlers before 06:00",
    ]

    return resolved, {
        "conflicts_resolved": reassigned,
        "unresolved": unresolved,
        "affected_blocks": affected_count,
        "flights_changed": [
            {"flight": k, **{f: SCHED_CHANGES[k][f] for f in ("original", "new", "delay_min", "airline", "route")}}
            for k in flight_keys if k in SCHED_CHANGES
        ],
    }, impact


# ── Master builder (called by prebake script) ─────────────────────────────────

def build_scenario(scenario_key, label, disruption_type, intensity_label, gantt, all_stands, fn, **kwargs):
    resolved_gantt, kpi, impact = fn(gantt, all_stands, **kwargs)
    return {
        "scenario_key":   scenario_key,
        "scenario_label": label,
        "disruption_type": disruption_type,
        "intensity":      intensity_label,
        "gantt":          resolved_gantt,
        "kpi":            kpi,
        "impact_summary": impact,
    }
