"""The mobility warehouse, declared once and used everywhere.

This module is the single source of truth for the warehouse. Three
consumers read it, and they must never disagree:

* the loader, which turns these declarations into ``CREATE TABLE``
  statements for whichever dialect is in play;
* the retriever, which embeds the table and column *descriptions* to
  decide which tables a question is about;
* the prompt builder, which renders the retrieved subset for the model.

Keeping them on one declaration is what makes retrieval trustworthy. If
the descriptions lived in a separate hand-maintained catalogue, they
would drift from the real columns, and retrieval would confidently pull
a table whose description no longer matches what it holds -- a failure
that looks like a model problem and is actually a data problem.

The descriptions are written for a *reader*, not as column-name
restatements. "trip_legs.delay_min: minutes later than the scheduled
arrival; 0 for on-time, negative when early" is retrievable by a
question about lateness. "delay_min: the delay in minutes" is not much
better than the column name itself, and embedding it adds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Column:
    """One column: its physical type and what it actually means."""

    name: str
    type: str
    description: str
    #: Set for the table's primary key.
    primary_key: bool = False
    #: ``"table.column"`` this column points at, when it is a foreign key.
    references: str | None = None
    #: For low-cardinality columns, the exact set of values that exist.
    #: These are surfaced to the model verbatim so it writes
    #: ``mode = 'e_scooter'`` and not ``'scooter'``.
    enum_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class Table:
    """One table, with the prose that makes it findable by retrieval."""

    name: str
    #: One or two sentences a person would recognise. This is the primary
    #: text embedded for retrieval, so it should describe the table's
    #: *purpose and grain*, not repeat its name.
    description: str
    #: Coarse grouping. Used to expand a retrieval hit to its neighbours
    #: and to keep the schema browsable in the UI.
    subject_area: str
    columns: tuple[Column, ...]
    #: The grain, stated plainly: what one row represents.
    grain: str = ""
    #: Alternative words a person might use for this table. Embedded
    #: alongside the description, because riders say "journey" and the
    #: table is called "trips".
    synonyms: tuple[str, ...] = field(default_factory=tuple)

    @property
    def primary_key(self) -> str | None:
        for column in self.columns:
            if column.primary_key:
                return column.name
        return None

    @property
    def foreign_keys(self) -> tuple[tuple[str, str], ...]:
        """``((local_column, "other_table.other_column"), ...)``."""
        return tuple(
            (c.name, c.references) for c in self.columns if c.references
        )

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name} has no column {name!r}")


# ---------------------------------------------------------------------------
# Subject areas. Retrieval expands a hit to same-area neighbours, so these
# groupings are load-bearing, not documentation.
# ---------------------------------------------------------------------------

# Prefixed AREA_* so a subject-area constant can never collide with the
# Table object of the same name -- which it did, silently, on the first
# pass: `TRIPS = "trips"` was rebound to the trips Table before trip_legs
# was declared, so trip_legs got a Table as its subject_area.
AREA_GEOGRAPHY = "geography"
AREA_NETWORK = "network"
AREA_FLEET = "fleet"
AREA_RIDERSHIP = "ridership"
AREA_TRIPS = "trips"
AREA_REVENUE = "revenue"
AREA_OPERATIONS = "operations"
AREA_ENVIRONMENT = "environment"

SUBJECT_AREAS: tuple[str, ...] = (
    AREA_GEOGRAPHY, AREA_NETWORK, AREA_FLEET, AREA_RIDERSHIP,
    AREA_TRIPS, AREA_REVENUE, AREA_OPERATIONS, AREA_ENVIRONMENT,
)

MODES = ("bike", "e_scooter", "bus", "metro", "rideshare")


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

ZONES = Table(
    name="zones",
    description=(
        "Administrative districts the city is divided into. Every trip "
        "starts and ends in a zone, and weather, air quality and fares are "
        "all measured or priced per zone."
    ),
    subject_area=AREA_GEOGRAPHY,
    grain="one row per city zone",
    synonyms=("district", "neighbourhood", "area", "borough", "region"),
    columns=(
        Column("zone_id", "INTEGER", "Unique zone identifier.", primary_key=True),
        Column("zone_name", "VARCHAR", "Human-readable neighbourhood name."),
        Column(
            "borough",
            "VARCHAR",
            "The larger administrative unit the zone belongs to.",
            enum_values=("Northside", "Harbour", "Old Town", "Westgate", "Industrial Park"),
        ),
        Column("area_sq_km", "DOUBLE", "Land area of the zone in square kilometres."),
        Column("population", "INTEGER", "Resident population of the zone."),
        Column(
            "zone_type",
            "VARCHAR",
            "Dominant land use, which drives commuting patterns: residential "
            "zones generate morning outbound trips, commercial ones absorb them.",
            enum_values=("residential", "commercial", "mixed", "industrial", "parkland"),
        ),
        Column("centroid_lat", "DOUBLE", "Latitude of the zone centroid."),
        Column("centroid_lon", "DOUBLE", "Longitude of the zone centroid."),
    ),
)

STATIONS = Table(
    name="stations",
    description=(
        "Physical pick-up and drop-off points: bike docks, scooter "
        "corrals, bus stops and metro platforms. Trips legs begin and end "
        "at stations, and shared vehicles are charged at them."
    ),
    subject_area=AREA_GEOGRAPHY,
    grain="one row per physical station",
    synonyms=("dock", "stop", "platform", "corral", "terminal"),
    columns=(
        Column("station_id", "INTEGER", "Unique station identifier.", primary_key=True),
        Column("station_name", "VARCHAR", "Public-facing name of the station."),
        Column("zone_id", "INTEGER", "Zone the station sits in.", references="zones.zone_id"),
        Column(
            "mode",
            "VARCHAR",
            "Transport mode this station serves.",
            enum_values=MODES,
        ),
        Column("dock_capacity", "INTEGER", "Number of docks or bays; 0 for metro and bus stops."),
        Column("has_charging", "BOOLEAN", "Whether the station can charge electric vehicles."),
        Column("latitude", "DOUBLE", "Station latitude."),
        Column("longitude", "DOUBLE", "Station longitude."),
        Column("opened_date", "DATE", "Date the station entered service."),
    ),
)

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

OPERATORS = Table(
    name="operators",
    description=(
        "Companies contracted to run part of the network. Each operator "
        "runs one mode; comparing operators is how service quality and "
        "cost per contract get assessed."
    ),
    subject_area=AREA_NETWORK,
    grain="one row per contracted operator",
    synonyms=("company", "provider", "contractor", "vendor"),
    columns=(
        Column("operator_id", "INTEGER", "Unique operator identifier.", primary_key=True),
        Column("operator_name", "VARCHAR", "Trading name of the operator."),
        Column("mode", "VARCHAR", "The single mode this operator runs.", enum_values=MODES),
        Column("contract_start", "DATE", "Date the current contract began."),
        Column("contract_end", "DATE", "Date the current contract expires."),
        Column("annual_subsidy_eur", "DOUBLE", "Public subsidy paid to the operator per year, in euros."),
    ),
)

ROUTES = Table(
    name="routes",
    description=(
        "Fixed scheduled services: bus and metro lines that follow an "
        "ordered sequence of stops. Bike, scooter and rideshare trips are "
        "free-floating and have no route."
    ),
    subject_area=AREA_NETWORK,
    grain="one row per scheduled route or line",
    synonyms=("line", "service", "corridor"),
    columns=(
        Column("route_id", "INTEGER", "Unique route identifier.", primary_key=True),
        Column("route_name", "VARCHAR", "Public route designation, such as 'M4' or 'Harbour Loop'."),
        Column("mode", "VARCHAR", "Mode of the route; only bus and metro have routes.", enum_values=("bus", "metro")),
        Column("operator_id", "INTEGER", "Operator running the route.", references="operators.operator_id"),
        Column("length_km", "DOUBLE", "End-to-end route length in kilometres."),
        Column("is_express", "BOOLEAN", "Whether the route skips intermediate stops."),
        Column("headway_peak_min", "INTEGER", "Scheduled minutes between services at peak."),
        Column("headway_offpeak_min", "INTEGER", "Scheduled minutes between services off peak."),
    ),
)

ROUTE_STOPS = Table(
    name="route_stops",
    description=(
        "The ordered sequence of stations each route calls at, with the "
        "scheduled running time from the start of the route. Joining this "
        "to trip legs is how actual arrival is compared against schedule."
    ),
    subject_area=AREA_NETWORK,
    grain="one row per station per route, in call order",
    synonyms=("timetable", "schedule", "stop sequence", "calling points"),
    columns=(
        Column("route_stop_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("route_id", "INTEGER", "Route this stop belongs to.", references="routes.route_id"),
        Column("station_id", "INTEGER", "Station called at.", references="stations.station_id"),
        Column("stop_sequence", "INTEGER", "Position in the route, starting at 1."),
        Column("scheduled_offset_min", "INTEGER", "Scheduled minutes from route start to this stop."),
    ),
)

# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------

VEHICLE_MODELS = Table(
    name="vehicle_models",
    description=(
        "The catalogue of vehicle types in service, with purchase cost, "
        "battery size and range. Comparing models is how procurement "
        "decisions and cost-per-kilometre analyses are made."
    ),
    subject_area=AREA_FLEET,
    grain="one row per vehicle model",
    synonyms=("vehicle type", "make", "model", "equipment"),
    columns=(
        Column("model_id", "INTEGER", "Unique model identifier.", primary_key=True),
        Column("manufacturer", "VARCHAR", "Company that builds the vehicle."),
        Column("model_name", "VARCHAR", "Manufacturer's model designation."),
        Column("mode", "VARCHAR", "Mode this model serves.", enum_values=MODES),
        Column("is_electric", "BOOLEAN", "Whether the vehicle is battery-powered."),
        Column("battery_kwh", "DOUBLE", "Usable battery capacity in kilowatt-hours; 0 for non-electric."),
        Column("range_km", "DOUBLE", "Manufacturer-rated range on a full charge."),
        Column("passenger_capacity", "INTEGER", "Seats plus standing room; 1 for bikes and scooters."),
        Column("purchase_cost_eur", "DOUBLE", "Unit purchase price in euros."),
        Column("year_introduced", "INTEGER", "Year the model first entered the fleet."),
    ),
)

VEHICLES = Table(
    name="vehicles",
    description=(
        "Individual physical vehicles in the fleet, each an instance of a "
        "model. Odometer and status here drive utilisation and "
        "availability questions."
    ),
    subject_area=AREA_FLEET,
    grain="one row per physical vehicle",
    synonyms=("bike", "scooter", "bus", "train", "car", "asset", "unit"),
    columns=(
        Column("vehicle_id", "INTEGER", "Unique vehicle identifier.", primary_key=True),
        Column("model_id", "INTEGER", "Model this vehicle is.", references="vehicle_models.model_id"),
        Column("operator_id", "INTEGER", "Operator responsible for it.", references="operators.operator_id"),
        Column("home_station_id", "INTEGER", "Station the vehicle is based at.", references="stations.station_id"),
        Column(
            "status",
            "VARCHAR",
            "Current availability. Only 'active' vehicles can be dispatched.",
            enum_values=("active", "maintenance", "retired", "stolen"),
        ),
        Column("in_service_date", "DATE", "Date the vehicle entered service."),
        Column("odometer_km", "DOUBLE", "Lifetime distance travelled in kilometres."),
    ),
)

MAINTENANCE_EVENTS = Table(
    name="maintenance_events",
    description=(
        "Every time a vehicle went into the workshop: what was wrong, what "
        "it cost, and how long it was out of service. This is the table "
        "for reliability, downtime and repair-cost questions."
    ),
    subject_area=AREA_FLEET,
    grain="one row per maintenance visit",
    synonyms=("repair", "service", "workshop", "breakdown", "downtime"),
    columns=(
        Column("maintenance_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("vehicle_id", "INTEGER", "Vehicle serviced.", references="vehicles.vehicle_id"),
        Column("opened_at", "TIMESTAMP", "When the vehicle went out of service."),
        Column("closed_at", "TIMESTAMP", "When it returned to service; null if still in the workshop."),
        Column(
            "maintenance_type",
            "VARCHAR",
            "Why it was in the workshop. Scheduled work is planned; "
            "corrective and accident work is not.",
            enum_values=("scheduled", "corrective", "accident", "vandalism", "battery"),
        ),
        Column("cost_eur", "DOUBLE", "Parts and labour, in euros."),
        Column("downtime_hours", "DOUBLE", "Hours the vehicle was unavailable."),
        Column("technician_notes", "VARCHAR", "Free-text note from the technician."),
    ),
)

CHARGING_SESSIONS = Table(
    name="charging_sessions",
    description=(
        "Each time an electric vehicle was plugged in: energy delivered, "
        "duration and cost. Used for energy-consumption and charging-"
        "infrastructure utilisation questions."
    ),
    subject_area=AREA_FLEET,
    grain="one row per charging session",
    synonyms=("charge", "recharge", "plug in", "energy", "kwh"),
    columns=(
        Column("session_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("vehicle_id", "INTEGER", "Vehicle charged.", references="vehicles.vehicle_id"),
        Column("station_id", "INTEGER", "Station where charging happened.", references="stations.station_id"),
        Column("started_at", "TIMESTAMP", "When charging began."),
        Column("ended_at", "TIMESTAMP", "When charging finished."),
        Column("kwh_delivered", "DOUBLE", "Energy transferred in kilowatt-hours."),
        Column("cost_eur", "DOUBLE", "Cost of the energy, in euros."),
        Column("start_soc_pct", "DOUBLE", "Battery state of charge when plugged in, as a percentage."),
        Column("end_soc_pct", "DOUBLE", "Battery state of charge when unplugged, as a percentage."),
    ),
)

# ---------------------------------------------------------------------------
# Ridership
# ---------------------------------------------------------------------------

RIDERS = Table(
    name="riders",
    description=(
        "Registered users of the network. Contains personal attributes, so "
        "queries touching it are the ones output sanitisation cares about."
    ),
    subject_area=AREA_RIDERSHIP,
    grain="one row per registered rider",
    synonyms=("user", "customer", "passenger", "member", "commuter"),
    columns=(
        Column("rider_id", "INTEGER", "Unique rider identifier.", primary_key=True),
        Column("email", "VARCHAR", "Contact email address. Personally identifying."),
        Column("full_name", "VARCHAR", "Rider's full name. Personally identifying."),
        Column("signup_date", "DATE", "Date the rider registered."),
        Column("home_zone_id", "INTEGER", "Zone the rider lives in.", references="zones.zone_id"),
        Column("birth_year", "INTEGER", "Year of birth, used for age-band analysis."),
        Column(
            "rider_type",
            "VARCHAR",
            "Fare category the rider qualifies for, which determines discounts.",
            enum_values=("standard", "student", "senior", "employee", "tourist"),
        ),
        Column("is_active", "BOOLEAN", "Whether the account is currently open."),
    ),
)

SUBSCRIPTIONS = Table(
    name="subscriptions",
    description=(
        "Recurring travel passes held by riders, such as monthly or annual "
        "plans. Churn, retention and recurring-revenue questions live here."
    ),
    subject_area=AREA_RIDERSHIP,
    grain="one row per subscription period per rider",
    synonyms=("pass", "plan", "membership", "season ticket"),
    columns=(
        Column("subscription_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("rider_id", "INTEGER", "Rider holding the subscription.", references="riders.rider_id"),
        Column(
            "plan_name",
            "VARCHAR",
            "Commercial name of the plan.",
            enum_values=("Basic Monthly", "Commuter Plus", "Unlimited Annual", "Weekend Only", "Student Term"),
        ),
        Column("monthly_price_eur", "DOUBLE", "Recurring charge in euros."),
        Column("started_on", "DATE", "First day of cover."),
        Column("ended_on", "DATE", "Last day of cover; null while active."),
        Column(
            "status",
            "VARCHAR",
            "Lifecycle state. 'cancelled' means the rider churned.",
            enum_values=("active", "cancelled", "expired", "paused"),
        ),
        Column("auto_renew", "BOOLEAN", "Whether the plan renews automatically."),
    ),
)

FEEDBACK = Table(
    name="feedback",
    description=(
        "Star ratings and written comments riders left about a trip. The "
        "table to use for satisfaction, complaint-theme and service-quality "
        "questions."
    ),
    subject_area=AREA_RIDERSHIP,
    grain="one row per piece of feedback on a trip",
    synonyms=("review", "rating", "complaint", "survey", "satisfaction", "nps"),
    columns=(
        Column("feedback_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("trip_id", "INTEGER", "Trip being rated.", references="trips.trip_id"),
        Column("rider_id", "INTEGER", "Rider who left it.", references="riders.rider_id"),
        Column("submitted_at", "TIMESTAMP", "When the feedback was submitted."),
        Column("rating", "INTEGER", "Star rating from 1 (worst) to 5 (best)."),
        Column(
            "category",
            "VARCHAR",
            "What the feedback is about.",
            enum_values=("cleanliness", "punctuality", "safety", "pricing", "app_experience", "staff"),
        ),
        Column("comment_text", "VARCHAR", "Free-text comment left by the rider."),
    ),
)

# ---------------------------------------------------------------------------
# Trips -- the central fact
# ---------------------------------------------------------------------------

TRIPS = Table(
    name="trips",
    description=(
        "A complete door-to-door journey by one rider, which may combine "
        "several modes. This is the central fact table: most questions "
        "about demand, distance, duration or revenue start here and join "
        "outward."
    ),
    subject_area=AREA_TRIPS,
    grain="one row per completed or attempted journey",
    synonyms=("journey", "ride", "travel", "movement", "demand"),
    columns=(
        Column("trip_id", "INTEGER", "Unique trip identifier.", primary_key=True),
        Column("rider_id", "INTEGER", "Rider who travelled.", references="riders.rider_id"),
        Column("started_at", "TIMESTAMP", "When the first leg began."),
        Column("ended_at", "TIMESTAMP", "When the last leg finished."),
        Column("origin_zone_id", "INTEGER", "Zone the journey started in.", references="zones.zone_id"),
        Column("dest_zone_id", "INTEGER", "Zone the journey ended in.", references="zones.zone_id"),
        Column("leg_count", "INTEGER", "Number of legs; more than 1 means the rider changed mode."),
        Column("total_distance_km", "DOUBLE", "Sum of all leg distances."),
        Column("duration_min", "DOUBLE", "Wall-clock minutes from start to end, including waiting between legs."),
        Column("total_fare_eur", "DOUBLE", "Total charged for the journey, after any discount."),
        Column(
            "trip_status",
            "VARCHAR",
            "How the journey ended. Only 'completed' trips reached their "
            "destination; 'abandoned' means the rider gave up mid-journey.",
            enum_values=("completed", "cancelled", "abandoned"),
        ),
        Column("is_peak", "BOOLEAN", "Whether the trip started during a weekday peak window."),
    ),
)

TRIP_LEGS = Table(
    name="trip_legs",
    description=(
        "One mode-specific segment of a journey. A rider who takes a bike "
        "to the metro and then walks produces multiple legs under one trip. "
        "Delay, vehicle and route detail all live at this grain, not on the "
        "trip."
    ),
    subject_area=AREA_TRIPS,
    grain="one row per mode segment within a trip",
    synonyms=("segment", "stage", "hop", "transfer", "connection"),
    columns=(
        Column("leg_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("trip_id", "INTEGER", "Journey this leg belongs to.", references="trips.trip_id"),
        Column("leg_sequence", "INTEGER", "Order of this leg within the trip, starting at 1."),
        Column("mode", "VARCHAR", "How the rider travelled on this leg.", enum_values=MODES),
        Column("vehicle_id", "INTEGER", "Vehicle used; null for metro and bus where vehicles are pooled.", references="vehicles.vehicle_id"),
        Column("route_id", "INTEGER", "Route followed; null for free-floating modes.", references="routes.route_id"),
        Column("start_station_id", "INTEGER", "Where the leg began.", references="stations.station_id"),
        Column("end_station_id", "INTEGER", "Where the leg ended.", references="stations.station_id"),
        Column("started_at", "TIMESTAMP", "Leg start time."),
        Column("ended_at", "TIMESTAMP", "Leg end time."),
        Column("distance_km", "DOUBLE", "Distance covered on this leg."),
        Column(
            "delay_min",
            "DOUBLE",
            "Minutes later than the scheduled arrival. Zero is on time and "
            "negative means the leg arrived early. Only meaningful for bus "
            "and metro, which run to a timetable.",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Revenue
# ---------------------------------------------------------------------------

FARES = Table(
    name="fares",
    description=(
        "The published price list: what each mode costs between any two "
        "zones, and from when. Historic rows are kept so past trips can be "
        "priced with the tariff that applied at the time."
    ),
    subject_area=AREA_REVENUE,
    grain="one row per mode per zone pair per tariff period",
    synonyms=("tariff", "price", "pricing", "rate card", "cost"),
    columns=(
        Column("fare_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("mode", "VARCHAR", "Mode the price applies to.", enum_values=MODES),
        Column("origin_zone_id", "INTEGER", "Zone travelled from.", references="zones.zone_id"),
        Column("dest_zone_id", "INTEGER", "Zone travelled to.", references="zones.zone_id"),
        Column("base_fare_eur", "DOUBLE", "Flat charge for boarding, in euros."),
        Column("per_km_eur", "DOUBLE", "Additional charge per kilometre travelled."),
        Column("peak_multiplier", "DOUBLE", "Factor applied to the total during peak hours."),
        Column("effective_from", "DATE", "First date this tariff applied."),
        Column("effective_to", "DATE", "Last date it applied; null for the current tariff."),
    ),
)

PAYMENTS = Table(
    name="payments",
    description=(
        "Money actually collected for a trip, including the method used and "
        "whether it succeeded. Revenue questions should use this rather "
        "than trip fares, because failed and refunded payments differ from "
        "amounts charged."
    ),
    subject_area=AREA_REVENUE,
    grain="one row per payment attempt",
    synonyms=("transaction", "charge", "billing", "revenue", "settlement"),
    columns=(
        Column("payment_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("trip_id", "INTEGER", "Trip being paid for.", references="trips.trip_id"),
        Column("rider_id", "INTEGER", "Rider charged.", references="riders.rider_id"),
        Column("amount_eur", "DOUBLE", "Amount taken, in euros. Negative for refunds."),
        Column(
            "payment_method",
            "VARCHAR",
            "How the rider paid.",
            enum_values=("card", "wallet", "subscription", "cash", "employer_benefit"),
        ),
        Column("paid_at", "TIMESTAMP", "When the payment was processed."),
        Column("promotion_id", "INTEGER", "Promotion applied, if any.", references="promotions.promotion_id"),
        Column(
            "status",
            "VARCHAR",
            "Outcome. Only 'settled' payments represent collected revenue.",
            enum_values=("settled", "failed", "refunded", "pending", "disputed"),
        ),
    ),
)

PROMOTIONS = Table(
    name="promotions",
    description=(
        "Discount campaigns riders can redeem at payment. Used to measure "
        "campaign uptake and the revenue given away."
    ),
    subject_area=AREA_REVENUE,
    grain="one row per promotional campaign",
    synonyms=("discount", "promo code", "voucher", "campaign", "offer"),
    columns=(
        Column("promotion_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("promo_code", "VARCHAR", "Code riders enter to redeem."),
        Column("campaign_name", "VARCHAR", "Internal name of the campaign."),
        Column("discount_pct", "DOUBLE", "Percentage taken off the fare."),
        Column("valid_from", "DATE", "First day the code works."),
        Column("valid_to", "DATE", "Last day the code works."),
        Column("max_redemptions", "INTEGER", "Cap on total uses across all riders."),
        Column(
            "target_segment",
            "VARCHAR",
            "Who the campaign was aimed at.",
            enum_values=("new_riders", "lapsed", "students", "all", "commuters"),
        ),
    ),
)

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

INCIDENTS = Table(
    name="incidents",
    description=(
        "Disruptions on the network: collisions, breakdowns, obstructions "
        "and security events. The table for reliability, safety and "
        "disruption-impact questions."
    ),
    subject_area=AREA_OPERATIONS,
    grain="one row per reported incident",
    synonyms=("disruption", "accident", "outage", "event", "failure", "safety"),
    columns=(
        Column("incident_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("occurred_at", "TIMESTAMP", "When the incident began."),
        Column("resolved_at", "TIMESTAMP", "When normal service resumed; null if ongoing."),
        Column("zone_id", "INTEGER", "Zone where it happened.", references="zones.zone_id"),
        Column("route_id", "INTEGER", "Route affected, if any.", references="routes.route_id"),
        Column("vehicle_id", "INTEGER", "Vehicle involved, if any.", references="vehicles.vehicle_id"),
        Column(
            "incident_type",
            "VARCHAR",
            "What happened.",
            enum_values=("collision", "breakdown", "obstruction", "weather", "medical", "security"),
        ),
        Column(
            "severity",
            "VARCHAR",
            "How disruptive it was. 'critical' incidents suspend service.",
            enum_values=("low", "moderate", "high", "critical"),
        ),
        Column("riders_affected", "INTEGER", "Estimated number of riders delayed or stranded."),
    ),
)

SERVICE_ALERTS = Table(
    name="service_alerts",
    description=(
        "Public notices posted to riders about planned or unplanned service "
        "changes. Distinct from incidents: an alert is the communication, "
        "an incident is the event."
    ),
    subject_area=AREA_OPERATIONS,
    grain="one row per posted alert",
    synonyms=("notice", "announcement", "advisory", "warning", "notification"),
    columns=(
        Column("alert_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("route_id", "INTEGER", "Route the alert concerns.", references="routes.route_id"),
        Column("incident_id", "INTEGER", "Incident that prompted it, if any.", references="incidents.incident_id"),
        Column("posted_at", "TIMESTAMP", "When the alert went live."),
        Column("cleared_at", "TIMESTAMP", "When it was withdrawn; null while active."),
        Column(
            "alert_type",
            "VARCHAR",
            "Nature of the change.",
            enum_values=("delay", "detour", "suspension", "planned_works", "crowding"),
        ),
        Column("message", "VARCHAR", "Text shown to riders."),
    ),
)

STATION_SNAPSHOTS = Table(
    name="station_snapshots",
    description=(
        "Periodic readings of how many vehicles and free docks each station "
        "had. This is the table for availability, rebalancing and "
        "'was anything there when I needed it' questions."
    ),
    subject_area=AREA_OPERATIONS,
    grain="one row per station per snapshot timestamp",
    synonyms=("availability", "inventory", "occupancy", "rebalancing", "stock"),
    columns=(
        Column("snapshot_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("station_id", "INTEGER", "Station observed.", references="stations.station_id"),
        Column("captured_at", "TIMESTAMP", "Time of the reading."),
        Column("bikes_available", "INTEGER", "Bikes ready to hire."),
        Column("scooters_available", "INTEGER", "Scooters ready to hire."),
        Column("docks_available", "INTEGER", "Empty docks available to return to."),
        Column("is_empty", "BOOLEAN", "True when no vehicle was available to hire."),
        Column("is_full", "BOOLEAN", "True when no dock was free to return to."),
    ),
)

STAFF_SHIFTS = Table(
    name="staff_shifts",
    description=(
        "Worked shifts by operational staff: drivers, technicians and "
        "rebalancing crews. Used for labour-cost and coverage questions."
    ),
    subject_area=AREA_OPERATIONS,
    grain="one row per staff member per shift",
    synonyms=("roster", "rota", "labour", "crew", "staffing", "workforce"),
    columns=(
        Column("shift_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("staff_name", "VARCHAR", "Name of the staff member. Personally identifying."),
        Column("operator_id", "INTEGER", "Operator employing them.", references="operators.operator_id"),
        Column("zone_id", "INTEGER", "Zone the shift was worked in.", references="zones.zone_id"),
        Column(
            "role",
            "VARCHAR",
            "What the staff member does.",
            enum_values=("driver", "technician", "rebalancer", "supervisor", "customer_service"),
        ),
        Column("shift_start", "TIMESTAMP", "Clock-in time."),
        Column("shift_end", "TIMESTAMP", "Clock-out time."),
        Column("hourly_rate_eur", "DOUBLE", "Pay rate for the shift, in euros."),
        Column("overtime_hours", "DOUBLE", "Hours worked beyond the scheduled shift."),
    ),
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

WEATHER_HOURLY = Table(
    name="weather_hourly",
    description=(
        "Hourly weather readings per zone. Rain and cold suppress bike and "
        "scooter demand sharply, so this is the table to join when a "
        "question mentions weather, rain or temperature."
    ),
    subject_area=AREA_ENVIRONMENT,
    grain="one row per zone per hour",
    synonyms=("rain", "temperature", "precipitation", "conditions", "forecast", "wet"),
    columns=(
        Column("weather_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("zone_id", "INTEGER", "Zone observed.", references="zones.zone_id"),
        Column("observed_at", "TIMESTAMP", "Hour of the observation."),
        Column("temp_c", "DOUBLE", "Air temperature in degrees Celsius."),
        Column("precipitation_mm", "DOUBLE", "Rainfall in millimetres during the hour; 0 when dry."),
        Column("wind_kph", "DOUBLE", "Wind speed in kilometres per hour."),
        Column(
            "condition",
            "VARCHAR",
            "Summary of conditions. 'rain' and 'snow' are the wet cases.",
            enum_values=("clear", "cloudy", "rain", "snow", "fog"),
        ),
    ),
)

AIR_QUALITY_HOURLY = Table(
    name="air_quality_hourly",
    description=(
        "Hourly air pollution readings per zone. Used for environmental "
        "impact questions and to test whether mode shift improves air "
        "quality in a district."
    ),
    subject_area=AREA_ENVIRONMENT,
    grain="one row per zone per hour",
    synonyms=("pollution", "aqi", "pm2.5", "emissions", "smog", "air"),
    columns=(
        Column("aq_id", "INTEGER", "Unique identifier.", primary_key=True),
        Column("zone_id", "INTEGER", "Zone observed.", references="zones.zone_id"),
        Column("observed_at", "TIMESTAMP", "Hour of the observation."),
        Column("pm25", "DOUBLE", "Fine particulate concentration in micrograms per cubic metre."),
        Column("no2", "DOUBLE", "Nitrogen dioxide concentration in micrograms per cubic metre."),
        Column("aqi", "INTEGER", "Composite air quality index; higher is worse."),
        Column(
            "aqi_category",
            "VARCHAR",
            "Banded interpretation of the index.",
            enum_values=("good", "moderate", "unhealthy_sensitive", "unhealthy", "hazardous"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# The warehouse
# ---------------------------------------------------------------------------

#: Declared in dependency order, so a loader can create and populate them
#: front to back without deferring foreign keys.
TABLES: tuple[Table, ...] = (
    ZONES,
    STATIONS,
    OPERATORS,
    ROUTES,
    ROUTE_STOPS,
    VEHICLE_MODELS,
    VEHICLES,
    RIDERS,
    SUBSCRIPTIONS,
    PROMOTIONS,
    FARES,
    TRIPS,
    TRIP_LEGS,
    PAYMENTS,
    FEEDBACK,
    MAINTENANCE_EVENTS,
    CHARGING_SESSIONS,
    INCIDENTS,
    SERVICE_ALERTS,
    STATION_SNAPSHOTS,
    STAFF_SHIFTS,
    WEATHER_HOURLY,
    AIR_QUALITY_HOURLY,
)

TABLES_BY_NAME: dict[str, Table] = {t.name: t for t in TABLES}

#: Columns holding personally identifying data. The output sanitiser masks
#: these unless a request explicitly opts in, and the list lives here --
#: beside the column declarations -- so adding a PII column and forgetting
#: to protect it requires actively skipping this line.
PII_COLUMNS: frozenset[str] = frozenset(
    {
        "riders.email",
        "riders.full_name",
        "staff_shifts.staff_name",
        "feedback.comment_text",
        "maintenance_events.technician_notes",
    }
)


def table_names() -> tuple[str, ...]:
    return tuple(t.name for t in TABLES)


def foreign_key_graph() -> dict[str, set[str]]:
    """Undirected adjacency of tables joined by a declared foreign key.

    Retrieval uses this to expand a set of candidate tables to something
    actually joinable: retrieving ``trips`` without ``trip_legs`` gives
    the model no way to reach mode or delay, and it will either invent a
    column or answer the wrong question.
    """
    graph: dict[str, set[str]] = {t.name: set() for t in TABLES}
    for table in TABLES:
        for _, target in table.foreign_keys:
            other = target.split(".", 1)[0]
            if other in graph:
                graph[table.name].add(other)
                graph[other].add(table.name)
    return graph


def tables_in_subject_area(area: str) -> tuple[Table, ...]:
    return tuple(t for t in TABLES if t.subject_area == area)


def retrieval_document(table: Table) -> str:
    """The text embedded to make a table findable.

    Deliberately more than the description: the grain tells the model what
    one row means, the synonyms bridge the gap between how people speak
    and how columns are named, and the column descriptions carry the
    specific vocabulary ("delay", "state of charge", "precipitation") that
    a question will actually use. Column *names* alone retrieve badly --
    nobody asks about ``pm25``, they ask about pollution.
    """
    parts = [
        f"Table: {table.name}",
        f"Subject area: {table.subject_area}",
        table.description,
    ]
    if table.grain:
        parts.append(f"Grain: {table.grain}")
    if table.synonyms:
        parts.append("Also called: " + ", ".join(table.synonyms))
    parts.append(
        "Columns: "
        + "; ".join(f"{c.name} ({c.type}) - {c.description}" for c in table.columns)
    )
    return "\n".join(parts)
