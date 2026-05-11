from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict

@dataclass
class Stand:
    stand_id: str
    pier: str
    terminal: str
    icao_cat_max: str
    has_airbridge: bool
    cbp_eligible: bool
    mars_condition: str   # Raw condition text e.g. "STANDS 107L, 107R VACANT"
    mars_family_id: str
    stand_type: str       # 'CONTACT', 'REMOTE', 'SATELLITE'

    @property
    def mars_type(self) -> str:
        """Infer MARS role from stand_id suffix: C=Centre, L=Left, R=Right, else None."""
        if not self.mars_family_id:
            return 'None'
        sid = self.stand_id.upper()
        if sid.endswith('C'):
            return 'Centre'
        if sid.endswith('L'):
            return 'Left'
        if sid.endswith('R'):
            return 'Right'
        if sid.endswith('T'):
            return 'Tail'
        return 'None'

@dataclass
class Flight:
    date_str: str                # YYYY-MM-DD
    time_str: str                # HH:MM (STA or STD)
    flight_no: str
    direction: str               # 'ARR' or 'DEP'
    airport_code: str            # origin or destination code
    airline_name: str
    aircraft_type: str
    tail_reg: str
    icao_cat: str
    cbp_flag: bool
    actual_time_str: str         # HH:MM or ''
    assigned_gate: str           # Original gate from schedule (if any)
    terminal_assignment: str = 'T1' # Populated by cross-referencing airline_terminal.csv
    pier_preference: Optional[str] = None

@dataclass
class MovementBlock:
    """Represents a full turnaround combining an Arrival and a Departure for the same tail_reg."""
    tail_reg: str
    icao_cat: str
    terminal: str
    pier_preference: Optional[str] = None
    arrival_flight: Optional[Flight] = None
    departure_flight: Optional[Flight] = None
    
    # Computed physics
    occupancy_start: Optional[datetime] = None
    occupancy_end: Optional[datetime] = None
    cbp_required: bool = False
    
    # Outcome
    assigned_stand: Optional[str] = None
    priority_score: float = 0.0

    @property
    def original_gate(self) -> Optional[str]:
        if self.arrival_flight and self.arrival_flight.assigned_gate:
            return self.arrival_flight.assigned_gate
        if self.departure_flight and self.departure_flight.assigned_gate:
            return self.departure_flight.assigned_gate
        return None

@dataclass
class DecisionRecord:
    flight_id: str
    direction: str
    original_gate: str
    assigned_stand: str
    priority_tier: int
    score_breakdown: Dict[str, float]
    total_score: float
    eligible_count: int
    suggestions: List[str]
    explanation: str

@dataclass
class DailyPlan:
    date: str
    movements: List[MovementBlock] = field(default_factory=list)
    decisions: List[DecisionRecord] = field(default_factory=list)
    unresolved_conflicts: List[MovementBlock] = field(default_factory=list)
