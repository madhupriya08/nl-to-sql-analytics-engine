"""Deterministic generator for the mobility warehouse.

Produces every table declared in :mod:`schema_def`, from one seed, using
only the standard library. Re-running with the same seed gives identical
rows, which is what lets the eval harness assert exact numbers instead of
"roughly".

The distributions are correlated on purpose, and the correlations are the
ones a transport analyst would expect:

* rain suppresses bike and scooter demand sharply, bus and metro barely;
* residential zones emit trips in the morning peak and absorb them in the
  evening, commercial zones do the reverse;
* subprime-equivalent here is the old fleet -- older vehicle models break
  down more, cost more to fix, and drag their operator's ratings down;
* delay accumulates along a route, so later stops are later than earlier
  ones.

None of that is decoration. It is what makes a generated query's answer
*checkable by eye*: if "average delay by stop sequence" comes back flat,
something is wrong with the SQL, not the data. Uniform random data would
make every grouped query return the same bar three times and give the
eval harness nothing to assert.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta
from typing import Any, Iterator

from . import schema_def as S

DEFAULT_SEED = 20260419

#: Simulation window. 180 days keeps hourly weather and air quality at a
#: size DuckDB loads instantly while still covering two full seasons, so
#: "compare summer and autumn" questions have something to find.
START_DATE = date(2025, 4, 1)
DAYS = 180
END_DATE = START_DATE + timedelta(days=DAYS - 1)

#: Row counts per table. Tuned so the whole warehouse generates in a few
#: seconds and loads into DuckDB in well under one, while still being big
#: enough that a wrong GROUP BY is visible.
SCALE = {
    "zones": 12,
    "stations": 240,
    "operators": 10,
    "routes": 24,
    "vehicle_models": 18,
    "vehicles": 900,
    "riders": 4_000,
    "promotions": 15,
    "trips": 50_000,
    "maintenance_events": 2_500,
    "charging_sessions": 15_000,
    "incidents": 900,
    "station_snapshots": 60_000,
    "staff_shifts": 8_000,
}

ZONE_NAMES = (
    "Riverside", "Kingsway", "Old Harbour", "Northgate", "Elmwood",
    "Bellevue", "Docklands", "Fairmount", "Greenhill", "Stonebridge",
    "Marlow", "Ashford",
)

STATION_PREFIXES = (
    "Central", "North", "South", "East", "West", "Upper", "Lower",
    "Market", "Cathedral", "University", "Park", "Bridge", "Station",
    "Museum", "Stadium", "Hospital", "Library", "Exchange",
)
STATION_SUFFIXES = ("Square", "Street", "Avenue", "Gardens", "Plaza", "Interchange", "Wharf", "Green")

OPERATOR_NAMES = (
    "Velociti", "UrbanFlow", "Metrolink", "GreenWheel", "CityGlide",
    "Transnova", "Pedalworks", "RapidArc", "Civitas Transit", "Nimbus Mobility",
)

MANUFACTURERS = ("Vanterra", "Kesselring", "Aoyama Works", "Nordvik", "Brightline", "Pramac")

FIRST_NAMES = (
    "Alina", "Bo", "Camille", "Devan", "Elif", "Farid", "Greta", "Hugo",
    "Ines", "Jonas", "Kiran", "Lena", "Milo", "Nadia", "Omar", "Petra",
    "Quinn", "Rafael", "Saoirse", "Tomas", "Uma", "Viktor", "Wren", "Yusuf",
)
LAST_NAMES = (
    "Almeida", "Bergstrom", "Chowdhury", "Duarte", "Eriksen", "Fontaine",
    "Gallagher", "Haugen", "Ibrahim", "Jansen", "Kowalski", "Lindqvist",
    "Moreau", "Novak", "Okafor", "Pereira", "Rossi", "Svendsen", "Takahashi",
    "Varga", "Wojcik", "Zieliński",
)

COMMENT_TEMPLATES = {
    "cleanliness": ("Vehicle was spotless.", "Litter left in the cabin.", "Seats needed a wipe."),
    "punctuality": ("Right on time.", "Waited twelve minutes past the board.", "Arrived early, no complaints."),
    "safety": ("Driver took the corners hard.", "Felt safe throughout.", "Poor lighting at the stop."),
    "pricing": ("Good value for the distance.", "Peak surcharge felt steep.", "Cheaper than a taxi."),
    "app_experience": ("Unlock took three tries.", "Booking was seamless.", "App crashed at checkout."),
    "staff": ("Conductor was very helpful.", "Nobody at the desk.", "Friendly and quick."),
}

TECH_NOTES = (
    "Replaced brake pads and bled the line.",
    "Battery module 3 below threshold, swapped.",
    "Frame weld cracked at the joint.",
    "Routine 5,000 km inspection, no faults.",
    "Water ingress in the controller housing.",
    "Tyre replaced after puncture.",
)


def _round(value: float, places: int = 2) -> float:
    return round(value + 0.0, places)


def _iso(moment: datetime) -> str:
    """Render a timestamp the way both DuckDB and Snowflake parse it."""
    return moment.strftime("%Y-%m-%d %H:%M:%S")


class MobilityDataGenerator:
    """Generates all 23 tables, in dependency order.

    Held as a class rather than loose functions because the tables are not
    independent: legs reference the trip that produced them, payments
    reference that trip's fare, and feedback references both. Generating
    them in one pass with shared state is what keeps those references
    consistent -- the alternative is generating each table separately and
    then hoping the foreign keys line up.
    """

    def __init__(self, seed: int = DEFAULT_SEED, scale: dict[str, int] | None = None):
        self.rng = random.Random(seed)
        self.scale = {**SCALE, **(scale or {})}
        self.tables: dict[str, list[dict[str, Any]]] = {}

    # -- helpers ---------------------------------------------------------

    def _pick(self, options):
        return self.rng.choice(list(options))

    def _weighted(self, options, weights):
        return self.rng.choices(list(options), weights=weights, k=1)[0]

    def _moment(self) -> datetime:
        """A random instant in the window, with a realistic hour profile.

        Trips are not uniform across the day: two commuter peaks, a midday
        plateau, and a near-dead overnight. A question like "how does
        demand vary by hour" is only interesting if this curve exists.
        """
        day = self.rng.randrange(DAYS)
        moment_date = START_DATE + timedelta(days=day)
        hour = self._weighted(
            range(24),
            # 0-5 dead, 7-9 morning peak, 12-14 plateau, 16-19 evening peak
            (1, 1, 1, 1, 2, 5, 14, 42, 55, 38, 20, 18, 22, 21, 19, 24, 40, 58, 46, 28, 18, 12, 7, 3),
        )
        return datetime(
            moment_date.year, moment_date.month, moment_date.day,
            hour, self.rng.randrange(60), self.rng.randrange(60),
        )

    @staticmethod
    def _is_peak(moment: datetime) -> bool:
        return moment.weekday() < 5 and (7 <= moment.hour <= 9 or 16 <= moment.hour <= 19)

    # -- geography -------------------------------------------------------

    def gen_zones(self) -> list[dict]:
        rows = []
        for i in range(self.scale["zones"]):
            zone_type = self._weighted(
                ("residential", "commercial", "mixed", "industrial", "parkland"),
                (35, 22, 25, 12, 6),
            )
            # Population tracks land use, so per-capita questions are sane.
            density = {"residential": 9_500, "commercial": 3_200, "mixed": 6_800,
                       "industrial": 900, "parkland": 300}[zone_type]
            area = _round(self.rng.uniform(1.8, 9.5))
            rows.append({
                "zone_id": i + 1,
                "zone_name": ZONE_NAMES[i % len(ZONE_NAMES)],
                "borough": self._pick(S.ZONES.column("borough").enum_values),
                "area_sq_km": area,
                "population": int(area * density * self.rng.uniform(0.75, 1.3)),
                "zone_type": zone_type,
                "centroid_lat": _round(52.35 + self.rng.uniform(-0.12, 0.12), 5),
                "centroid_lon": _round(4.90 + self.rng.uniform(-0.14, 0.14), 5),
            })
        return rows

    def gen_stations(self) -> list[dict]:
        rows = []
        for i in range(self.scale["stations"]):
            mode = self._weighted(S.MODES, (40, 25, 18, 10, 7))
            has_docks = mode in ("bike", "e_scooter")
            rows.append({
                "station_id": i + 1,
                "station_name": f"{self._pick(STATION_PREFIXES)} {self._pick(STATION_SUFFIXES)}",
                "zone_id": self.rng.randrange(1, self.scale["zones"] + 1),
                "mode": mode,
                "dock_capacity": self.rng.randrange(8, 41, 2) if has_docks else 0,
                "has_charging": has_docks and self.rng.random() < 0.55,
                "latitude": _round(52.35 + self.rng.uniform(-0.12, 0.12), 5),
                "longitude": _round(4.90 + self.rng.uniform(-0.14, 0.14), 5),
                "opened_date": (START_DATE - timedelta(days=self.rng.randrange(200, 2600))).isoformat(),
            })
        return rows

    # -- network ---------------------------------------------------------

    def gen_operators(self) -> list[dict]:
        rows = []
        for i in range(self.scale["operators"]):
            start = START_DATE - timedelta(days=self.rng.randrange(400, 2000))
            rows.append({
                "operator_id": i + 1,
                "operator_name": OPERATOR_NAMES[i % len(OPERATOR_NAMES)],
                "mode": S.MODES[i % len(S.MODES)],
                "contract_start": start.isoformat(),
                "contract_end": (start + timedelta(days=self.rng.randrange(1100, 2600))).isoformat(),
                "annual_subsidy_eur": _round(self.rng.uniform(250_000, 4_800_000)),
            })
        return rows

    def gen_routes(self) -> list[dict]:
        scheduled_ops = [o for o in self.tables["operators"] if o["mode"] in ("bus", "metro")]
        rows = []
        for i in range(self.scale["routes"]):
            operator = self._pick(scheduled_ops)
            mode = operator["mode"]
            is_express = self.rng.random() < 0.25
            rows.append({
                "route_id": i + 1,
                "route_name": f"{'M' if mode == 'metro' else 'B'}{i + 1}",
                "mode": mode,
                "operator_id": operator["operator_id"],
                "length_km": _round(self.rng.uniform(4.0, 28.0)),
                "is_express": is_express,
                # Express routes run less often but skip stops; metro beats bus.
                "headway_peak_min": self.rng.choice([3, 4, 5, 6] if mode == "metro" else [6, 8, 10, 12]),
                "headway_offpeak_min": self.rng.choice([8, 10, 12] if mode == "metro" else [15, 20, 30]),
            })
        return rows

    def gen_route_stops(self) -> list[dict]:
        rows: list[dict] = []
        stop_id = 1
        scheduled_stations = [s for s in self.tables["stations"] if s["mode"] in ("bus", "metro")]
        if not scheduled_stations:
            scheduled_stations = self.tables["stations"]
        for route in self.tables["routes"]:
            stop_count = self.rng.randrange(6, 15) if route["is_express"] else self.rng.randrange(10, 25)
            chosen = self.rng.sample(scheduled_stations, min(stop_count, len(scheduled_stations)))
            offset = 0
            for sequence, station in enumerate(chosen, start=1):
                rows.append({
                    "route_stop_id": stop_id,
                    "route_id": route["route_id"],
                    "station_id": station["station_id"],
                    "stop_sequence": sequence,
                    "scheduled_offset_min": offset,
                })
                stop_id += 1
                offset += self.rng.randrange(2, 6) if route["mode"] == "metro" else self.rng.randrange(3, 9)
        return rows

    # -- fleet -----------------------------------------------------------

    def gen_vehicle_models(self) -> list[dict]:
        rows = []
        for i in range(self.scale["vehicle_models"]):
            mode = S.MODES[i % len(S.MODES)]
            electric = mode in ("e_scooter", "bike") or self.rng.random() < 0.6
            capacity = {"bike": 1, "e_scooter": 1, "rideshare": 4, "bus": 70, "metro": 220}[mode]
            base_cost = {"bike": 1_200, "e_scooter": 900, "rideshare": 34_000,
                         "bus": 320_000, "metro": 2_100_000}[mode]
            battery = 0.0
            if electric:
                battery = _round({"bike": 0.5, "e_scooter": 0.6, "rideshare": 62.0,
                                  "bus": 340.0, "metro": 0.0}[mode] * self.rng.uniform(0.85, 1.2))
            rows.append({
                "model_id": i + 1,
                "manufacturer": self._pick(MANUFACTURERS),
                "model_name": f"{self._pick(('Aero', 'Terra', 'Volt', 'Rapid', 'Civic', 'Orbit'))}-{self.rng.randrange(100, 900)}",
                "mode": mode,
                "is_electric": electric,
                "battery_kwh": battery,
                "range_km": _round(battery * self.rng.uniform(3.5, 6.0) if electric else self.rng.uniform(300, 700)),
                "passenger_capacity": capacity,
                "purchase_cost_eur": _round(base_cost * self.rng.uniform(0.8, 1.25)),
                # Model age is the driver of the reliability story below.
                "year_introduced": self.rng.randrange(2012, 2025),
            })
        return rows

    def gen_vehicles(self) -> list[dict]:
        models_by_mode: dict[str, list[dict]] = {}
        for model in self.tables["vehicle_models"]:
            models_by_mode.setdefault(model["mode"], []).append(model)
        stations_by_mode: dict[str, list[dict]] = {}
        for station in self.tables["stations"]:
            stations_by_mode.setdefault(station["mode"], []).append(station)
        ops_by_mode: dict[str, list[dict]] = {}
        for operator in self.tables["operators"]:
            ops_by_mode.setdefault(operator["mode"], []).append(operator)

        rows = []
        for i in range(self.scale["vehicles"]):
            mode = self._weighted(S.MODES, (45, 28, 14, 8, 5))
            model = self._pick(models_by_mode.get(mode) or self.tables["vehicle_models"])
            operator = self._pick(ops_by_mode.get(mode) or self.tables["operators"])
            station = self._pick(stations_by_mode.get(mode) or self.tables["stations"])
            in_service = START_DATE - timedelta(days=self.rng.randrange(30, 2200))
            age_days = (START_DATE - in_service).days
            rows.append({
                "vehicle_id": i + 1,
                "model_id": model["model_id"],
                "operator_id": operator["operator_id"],
                "home_station_id": station["station_id"],
                # Older vehicles are likelier to be off the road.
                "status": self._weighted(
                    ("active", "maintenance", "retired", "stolen"),
                    (88 - age_days / 120, 8 + age_days / 200, 3 + age_days / 300, 1),
                ),
                "in_service_date": in_service.isoformat(),
                "odometer_km": _round(age_days * self.rng.uniform(4, 45)),
            })
        return rows

    # -- ridership -------------------------------------------------------

    def gen_riders(self) -> list[dict]:
        rows = []
        for i in range(self.scale["riders"]):
            first, last = self._pick(FIRST_NAMES), self._pick(LAST_NAMES)
            rider_type = self._weighted(
                ("standard", "student", "senior", "employee", "tourist"),
                (52, 20, 12, 8, 8),
            )
            birth = {"student": (1998, 2007), "senior": (1945, 1962),
                     "employee": (1975, 2000), "standard": (1965, 2005),
                     "tourist": (1970, 2004)}[rider_type]
            rows.append({
                "rider_id": i + 1,
                "email": f"{first.lower()}.{last.lower().replace('ń', 'n')}{i}@example.com",
                "full_name": f"{first} {last}",
                "signup_date": (START_DATE - timedelta(days=self.rng.randrange(1, 1500))).isoformat(),
                "home_zone_id": self.rng.randrange(1, self.scale["zones"] + 1),
                "birth_year": self.rng.randrange(*birth),
                "rider_type": rider_type,
                "is_active": self.rng.random() < 0.87,
            })
        return rows

    def gen_subscriptions(self) -> list[dict]:
        plans = S.SUBSCRIPTIONS.column("plan_name").enum_values
        prices = {"Basic Monthly": 29.0, "Commuter Plus": 59.0, "Unlimited Annual": 89.0,
                  "Weekend Only": 19.0, "Student Term": 24.0}
        rows = []
        subscription_id = 1
        for rider in self.tables["riders"]:
            # Roughly 4 in 5 riders hold a pass; students skew to the term plan.
            if self.rng.random() > 0.8:
                continue
            plan = "Student Term" if rider["rider_type"] == "student" and self.rng.random() < 0.7 else self._pick(plans)
            started = date.fromisoformat(rider["signup_date"]) + timedelta(days=self.rng.randrange(0, 120))
            status = self._weighted(("active", "cancelled", "expired", "paused"), (58, 22, 14, 6))
            ended = None
            if status != "active":
                ended = (started + timedelta(days=self.rng.randrange(30, 500))).isoformat()
            rows.append({
                "subscription_id": subscription_id,
                "rider_id": rider["rider_id"],
                "plan_name": plan,
                "monthly_price_eur": _round(prices[plan] * self.rng.uniform(0.95, 1.05)),
                "started_on": started.isoformat(),
                "ended_on": ended,
                "status": status,
                "auto_renew": status == "active" and self.rng.random() < 0.75,
            })
            subscription_id += 1
        return rows

    # -- revenue ---------------------------------------------------------

    def gen_promotions(self) -> list[dict]:
        rows = []
        for i in range(self.scale["promotions"]):
            valid_from = START_DATE + timedelta(days=self.rng.randrange(0, DAYS - 30))
            rows.append({
                "promotion_id": i + 1,
                "promo_code": f"{self._pick(('RIDE', 'CITY', 'GREEN', 'SAVE', 'HOP'))}{self.rng.randrange(10, 99)}",
                "campaign_name": f"{self._pick(('Spring', 'Summer', 'Autumn', 'Launch', 'Winback'))} {self.rng.randrange(1, 5)}",
                "discount_pct": float(self.rng.choice([5, 10, 15, 20, 25, 30])),
                "valid_from": valid_from.isoformat(),
                "valid_to": (valid_from + timedelta(days=self.rng.randrange(14, 60))).isoformat(),
                "max_redemptions": self.rng.choice([500, 1_000, 2_500, 5_000]),
                "target_segment": self._pick(S.PROMOTIONS.column("target_segment").enum_values),
            })
        return rows

    def gen_fares(self) -> list[dict]:
        """One tariff per mode per zone pair, for a sample of pairs.

        A full cross join would be 5 modes x 144 pairs = 720 rows, which is
        fine, but pricing every pair makes the table uninteresting. Pricing
        a subset means a fare lookup can legitimately miss, which is the
        realistic case a good query has to handle.
        """
        base = {"bike": 1.0, "e_scooter": 1.2, "bus": 1.8, "metro": 2.2, "rideshare": 3.5}
        per_km = {"bike": 0.12, "e_scooter": 0.22, "bus": 0.10, "metro": 0.14, "rideshare": 0.95}
        rows = []
        fare_id = 1
        zone_count = self.scale["zones"]
        for mode in S.MODES:
            for origin in range(1, zone_count + 1):
                for dest in range(1, zone_count + 1):
                    if origin != dest and self.rng.random() > 0.45:
                        continue
                    rows.append({
                        "fare_id": fare_id,
                        "mode": mode,
                        "origin_zone_id": origin,
                        "dest_zone_id": dest,
                        "base_fare_eur": _round(base[mode] * self.rng.uniform(0.9, 1.15)),
                        "per_km_eur": _round(per_km[mode] * self.rng.uniform(0.9, 1.2), 3),
                        "peak_multiplier": _round(self.rng.choice([1.0, 1.15, 1.25, 1.4])),
                        "effective_from": (START_DATE - timedelta(days=90)).isoformat(),
                        "effective_to": None,
                    })
                    fare_id += 1
        return rows

    # -- environment -----------------------------------------------------

    def gen_weather(self) -> list[dict]:
        """Hourly weather per zone, with seasonal drift and wet spells.

        Rain arrives in runs rather than independent hours, because that is
        what makes "did demand drop when it rained" a question with a
        visible answer. Independent hourly coin flips would smear the
        effect across the whole window.
        """
        rows = []
        weather_id = 1
        for zone_id in range(1, self.scale["zones"] + 1):
            wet_hours_remaining = 0
            for day_offset in range(DAYS):
                current = START_DATE + timedelta(days=day_offset)
                # Seasonal temperature curve across the window.
                seasonal = 14 + 8 * math.sin((day_offset / DAYS) * math.pi)
                for hour in range(24):
                    if wet_hours_remaining > 0:
                        wet_hours_remaining -= 1
                        condition = "rain"
                        precipitation = _round(self.rng.uniform(0.4, 7.5))
                    elif self.rng.random() < 0.018:
                        wet_hours_remaining = self.rng.randrange(2, 9)
                        condition = "rain"
                        precipitation = _round(self.rng.uniform(0.4, 7.5))
                    else:
                        condition = self._weighted(("clear", "cloudy", "fog", "snow"), (55, 38, 5, 2))
                        precipitation = 0.0
                    diurnal = -4.5 * math.cos((hour / 24) * 2 * math.pi)
                    rows.append({
                        "weather_id": weather_id,
                        "zone_id": zone_id,
                        "observed_at": _iso(datetime(current.year, current.month, current.day, hour)),
                        "temp_c": _round(seasonal + diurnal + self.rng.uniform(-2.5, 2.5), 1),
                        "precipitation_mm": precipitation,
                        "wind_kph": _round(self.rng.uniform(2, 38), 1),
                        "condition": condition,
                    })
                    weather_id += 1
        return rows

    def gen_air_quality(self) -> list[dict]:
        """Hourly air quality, worse in industrial zones and at peak hours."""
        zone_type = {z["zone_id"]: z["zone_type"] for z in self.tables["zones"]}
        penalty = {"industrial": 28, "commercial": 14, "mixed": 9, "residential": 5, "parkland": 0}
        rows = []
        aq_id = 1
        for zone_id in range(1, self.scale["zones"] + 1):
            bias = penalty[zone_type[zone_id]]
            for day_offset in range(DAYS):
                current = START_DATE + timedelta(days=day_offset)
                for hour in range(24):
                    rush = 12 if hour in (7, 8, 17, 18) else 0
                    pm25 = _round(max(1.0, self.rng.gauss(11 + bias + rush, 5)), 1)
                    aqi = int(min(300, pm25 * 3.6 + self.rng.uniform(-8, 8)))
                    rows.append({
                        "aq_id": aq_id,
                        "zone_id": zone_id,
                        "observed_at": _iso(datetime(current.year, current.month, current.day, hour)),
                        "pm25": pm25,
                        "no2": _round(max(1.0, self.rng.gauss(18 + bias, 7)), 1),
                        "aqi": aqi,
                        "aqi_category": (
                            "good" if aqi <= 50 else
                            "moderate" if aqi <= 100 else
                            "unhealthy_sensitive" if aqi <= 150 else
                            "unhealthy" if aqi <= 200 else "hazardous"
                        ),
                    })
                    aq_id += 1
        return rows

    # -- trips, the central fact ----------------------------------------

    def gen_trips_and_children(self) -> None:
        """Generate trips, legs, payments and feedback in one pass.

        These four are generated together because they share state: a leg's
        distance sums into its trip, the trip's fare drives the payment
        amount, and feedback ratings depend on how delayed the legs were.
        Generating them separately and joining afterwards is where
        referential drift creeps in.
        """
        zones = self.tables["zones"]
        stations = self.tables["stations"]
        stations_by_mode: dict[str, list[dict]] = {}
        for station in stations:
            stations_by_mode.setdefault(station["mode"], []).append(station)
        routes_by_mode: dict[str, list[dict]] = {}
        for route in self.tables["routes"]:
            routes_by_mode.setdefault(route["mode"], []).append(route)
        vehicles_by_mode: dict[str, list[int]] = {}
        model_mode = {m["model_id"]: m["mode"] for m in self.tables["vehicle_models"]}
        for vehicle in self.tables["vehicles"]:
            vehicles_by_mode.setdefault(model_mode[vehicle["model_id"]], []).append(vehicle["vehicle_id"])

        # Wet hours, keyed by (zone, hour-truncated timestamp), so trip
        # generation can suppress bike demand when it was actually raining.
        wet: set[tuple[int, str]] = {
            (row["zone_id"], row["observed_at"])
            for row in self.tables["weather_hourly"]
            if row["precipitation_mm"] > 0
        }

        zone_type = {z["zone_id"]: z["zone_type"] for z in zones}
        residential = [z["zone_id"] for z in zones if zone_type[z["zone_id"]] in ("residential", "mixed")]
        commercial = [z["zone_id"] for z in zones if zone_type[z["zone_id"]] in ("commercial", "industrial", "mixed")]

        trips: list[dict] = []
        legs: list[dict] = []
        payments: list[dict] = []
        feedback: list[dict] = []
        leg_id = payment_id = feedback_id = 1

        promotions = self.tables["promotions"]
        rider_ids = [r["rider_id"] for r in self.tables["riders"]]

        for trip_id in range(1, self.scale["trips"] + 1):
            started = self._moment()
            hour_key = _iso(started.replace(minute=0, second=0, microsecond=0))
            peak = self._is_peak(started)

            # Commuters flow residential -> commercial in the morning and
            # back in the evening. This is the correlation that makes an
            # origin/destination query show a recognisable pattern.
            if peak and started.hour <= 12:
                origin_zone = self._pick(residential)
                dest_zone = self._pick(commercial)
            elif peak:
                origin_zone = self._pick(commercial)
                dest_zone = self._pick(residential)
            else:
                origin_zone = self.rng.randrange(1, self.scale["zones"] + 1)
                dest_zone = self.rng.randrange(1, self.scale["zones"] + 1)

            raining = (origin_zone, hour_key) in wet

            # Mode share shifts hard in the rain: bikes and scooters lose
            # riders to bus, metro and rideshare.
            mode_weights = (34, 22, 20, 16, 8) if not raining else (10, 6, 32, 34, 18)
            leg_count = self._weighted((1, 2, 3), (72, 24, 4))

            cursor = started
            total_distance = 0.0
            for sequence in range(1, leg_count + 1):
                mode = self._weighted(S.MODES, mode_weights)
                pool = stations_by_mode.get(mode) or stations
                start_station = self._pick(pool)
                end_station = self._pick(pool)
                distance = _round(abs(self.rng.gauss(3.2 if mode in ("bike", "e_scooter") else 6.5, 2.4)) + 0.3)
                speed_kph = {"bike": 15, "e_scooter": 18, "bus": 19, "metro": 34, "rideshare": 26}[mode]
                minutes = max(2.0, (distance / speed_kph) * 60)

                route = None
                delay = 0.0
                if mode in ("bus", "metro"):
                    candidates = routes_by_mode.get(mode)
                    if candidates:
                        route = self._pick(candidates)
                    # Delay is worse at peak and worse in the rain -- the
                    # two-factor effect a good query should be able to find.
                    base_delay = self.rng.gauss(1.4, 2.2)
                    delay = _round(base_delay + (2.6 if peak else 0) + (1.9 if raining else 0), 1)

                vehicle_pool = vehicles_by_mode.get(mode)
                vehicle_id = self._pick(vehicle_pool) if vehicle_pool and mode not in ("bus", "metro") else None

                ended = cursor + timedelta(minutes=minutes + max(0.0, delay))
                legs.append({
                    "leg_id": leg_id,
                    "trip_id": trip_id,
                    "leg_sequence": sequence,
                    "mode": mode,
                    "vehicle_id": vehicle_id,
                    "route_id": route["route_id"] if route else None,
                    "start_station_id": start_station["station_id"],
                    "end_station_id": end_station["station_id"],
                    "started_at": _iso(cursor),
                    "ended_at": _iso(ended),
                    "distance_km": distance,
                    "delay_min": delay,
                })
                leg_id += 1
                total_distance += distance
                cursor = ended + timedelta(minutes=self.rng.uniform(0.5, 6.0) if sequence < leg_count else 0)

            duration = _round((cursor - started).total_seconds() / 60.0, 1)
            status = self._weighted(("completed", "cancelled", "abandoned"), (93, 4, 3))
            fare = _round(max(0.0, 1.4 + total_distance * self.rng.uniform(0.18, 0.65)) * (1.25 if peak else 1.0))
            if status != "completed":
                fare = _round(fare * 0.35)

            trips.append({
                "trip_id": trip_id,
                "rider_id": self._pick(rider_ids),
                "started_at": _iso(started),
                "ended_at": _iso(cursor),
                "origin_zone_id": origin_zone,
                "dest_zone_id": dest_zone,
                "leg_count": leg_count,
                "total_distance_km": _round(total_distance),
                "duration_min": duration,
                "total_fare_eur": fare,
                "trip_status": status,
                "is_peak": peak,
            })

            promotion = self._pick(promotions) if self.rng.random() < 0.12 else None
            payments.append({
                "payment_id": payment_id,
                "trip_id": trip_id,
                "rider_id": trips[-1]["rider_id"],
                "amount_eur": fare if promotion is None else _round(fare * (1 - promotion["discount_pct"] / 100)),
                "payment_method": self._weighted(
                    ("card", "wallet", "subscription", "cash", "employer_benefit"),
                    (38, 26, 24, 6, 6),
                ),
                "paid_at": _iso(cursor + timedelta(seconds=self.rng.randrange(1, 300))),
                "promotion_id": promotion["promotion_id"] if promotion else None,
                "status": self._weighted(
                    ("settled", "failed", "refunded", "pending", "disputed"),
                    (91, 3, 3, 2, 1),
                ),
            })
            payment_id += 1

            # About 1 trip in 4 gets rated, and the rating tracks the worst
            # delay on the trip -- so "do delays hurt satisfaction" resolves.
            if self.rng.random() < 0.24:
                worst_delay = max((l["delay_min"] for l in legs[-leg_count:]), default=0.0)
                rating = 5 if worst_delay < 1 else 4 if worst_delay < 4 else 3 if worst_delay < 8 else 2 if worst_delay < 14 else 1
                rating = max(1, min(5, rating + self.rng.choice([-1, 0, 0, 0, 1])))
                category = self._pick(S.FEEDBACK.column("category").enum_values)
                feedback.append({
                    "feedback_id": feedback_id,
                    "trip_id": trip_id,
                    "rider_id": trips[-1]["rider_id"],
                    "submitted_at": _iso(cursor + timedelta(minutes=self.rng.randrange(2, 600))),
                    "rating": rating,
                    "category": category,
                    "comment_text": self._pick(COMMENT_TEMPLATES[category]),
                })
                feedback_id += 1

        self.tables["trips"] = trips
        self.tables["trip_legs"] = legs
        self.tables["payments"] = payments
        self.tables["feedback"] = feedback

    # -- operations ------------------------------------------------------

    def gen_maintenance(self) -> list[dict]:
        """Older models break down more and cost more -- the reliability story."""
        model_year = {m["model_id"]: m["year_introduced"] for m in self.tables["vehicle_models"]}
        vehicles = self.tables["vehicles"]
        # Weight vehicle selection by age, so old stock dominates the workshop.
        weights = [max(1, 2026 - model_year[v["model_id"]]) for v in vehicles]
        rows = []
        for i in range(self.scale["maintenance_events"]):
            vehicle = self.rng.choices(vehicles, weights=weights, k=1)[0]
            age = 2026 - model_year[vehicle["model_id"]]
            opened = self._moment()
            downtime = _round(abs(self.rng.gauss(6 + age * 0.8, 5)) + 0.5, 1)
            maintenance_type = self._weighted(
                ("scheduled", "corrective", "accident", "vandalism", "battery"),
                (40, 30, 8, 12, 10),
            )
            multiplier = {"scheduled": 0.6, "corrective": 1.0, "accident": 2.8,
                          "vandalism": 1.4, "battery": 2.2}[maintenance_type]
            rows.append({
                "maintenance_id": i + 1,
                "vehicle_id": vehicle["vehicle_id"],
                "opened_at": _iso(opened),
                "closed_at": _iso(opened + timedelta(hours=downtime)) if self.rng.random() < 0.94 else None,
                "maintenance_type": maintenance_type,
                "cost_eur": _round(self.rng.uniform(40, 900) * multiplier * (1 + age * 0.08)),
                "downtime_hours": downtime,
                "technician_notes": self._pick(TECH_NOTES),
            })
        return rows

    def gen_charging(self) -> list[dict]:
        electric_models = {m["model_id"] for m in self.tables["vehicle_models"] if m["is_electric"]}
        electric_vehicles = [v for v in self.tables["vehicles"] if v["model_id"] in electric_models]
        charging_stations = [s for s in self.tables["stations"] if s["has_charging"]] or self.tables["stations"]
        battery = {m["model_id"]: m["battery_kwh"] for m in self.tables["vehicle_models"]}
        rows = []
        for i in range(self.scale["charging_sessions"]):
            vehicle = self._pick(electric_vehicles or self.tables["vehicles"])
            capacity = max(0.4, battery.get(vehicle["model_id"], 1.0))
            start_soc = _round(self.rng.uniform(5, 55), 1)
            end_soc = _round(min(100.0, start_soc + self.rng.uniform(25, 70)), 1)
            kwh = _round(capacity * (end_soc - start_soc) / 100.0, 3)
            started = self._moment()
            rows.append({
                "session_id": i + 1,
                "vehicle_id": vehicle["vehicle_id"],
                "station_id": self._pick(charging_stations)["station_id"],
                "started_at": _iso(started),
                "ended_at": _iso(started + timedelta(minutes=self.rng.uniform(20, 240))),
                "kwh_delivered": kwh,
                "cost_eur": _round(kwh * self.rng.uniform(0.18, 0.42), 3),
                "start_soc_pct": start_soc,
                "end_soc_pct": end_soc,
            })
        return rows

    def gen_incidents(self) -> list[dict]:
        routes = self.tables["routes"]
        vehicles = self.tables["vehicles"]
        rows = []
        for i in range(self.scale["incidents"]):
            occurred = self._moment()
            severity = self._weighted(("low", "moderate", "high", "critical"), (48, 32, 15, 5))
            hours = {"low": 0.5, "moderate": 1.5, "high": 4.0, "critical": 9.0}[severity]
            rows.append({
                "incident_id": i + 1,
                "occurred_at": _iso(occurred),
                "resolved_at": _iso(occurred + timedelta(hours=hours * self.rng.uniform(0.6, 1.6))) if self.rng.random() < 0.93 else None,
                "zone_id": self.rng.randrange(1, self.scale["zones"] + 1),
                "route_id": self._pick(routes)["route_id"] if self.rng.random() < 0.6 else None,
                "vehicle_id": self._pick(vehicles)["vehicle_id"] if self.rng.random() < 0.7 else None,
                "incident_type": self._weighted(
                    ("collision", "breakdown", "obstruction", "weather", "medical", "security"),
                    (14, 34, 20, 14, 9, 9),
                ),
                "severity": severity,
                # Riders affected scales with severity, so impact queries rank.
                "riders_affected": int(abs(self.rng.gauss({"low": 12, "moderate": 90, "high": 480, "critical": 2400}[severity], 40))),
            })
        return rows

    def gen_service_alerts(self) -> list[dict]:
        incidents = self.tables["incidents"]
        routes = self.tables["routes"]
        rows = []
        alert_id = 1
        for incident in incidents:
            if self.rng.random() > 0.7:
                continue
            posted = datetime.strptime(incident["occurred_at"], "%Y-%m-%d %H:%M:%S") + timedelta(minutes=self.rng.randrange(2, 40))
            alert_type = self._weighted(("delay", "detour", "suspension", "planned_works", "crowding"), (40, 20, 14, 16, 10))
            rows.append({
                "alert_id": alert_id,
                "route_id": incident["route_id"] or self._pick(routes)["route_id"],
                "incident_id": incident["incident_id"],
                "posted_at": _iso(posted),
                "cleared_at": _iso(posted + timedelta(hours=self.rng.uniform(0.5, 12))) if self.rng.random() < 0.9 else None,
                "alert_type": alert_type,
                "message": f"{alert_type.replace('_', ' ').title()} affecting service. Expect disruption.",
            })
            alert_id += 1
        return rows

    def gen_station_snapshots(self) -> list[dict]:
        """Availability readings, with rush-hour emptiness in residential zones.

        Bike share drains outbound stations in the morning; that is the
        rebalancing problem, and it should be visible in the data.
        """
        dock_stations = [s for s in self.tables["stations"] if s["dock_capacity"] > 0] or self.tables["stations"]
        zone_type = {z["zone_id"]: z["zone_type"] for z in self.tables["zones"]}
        rows = []
        for i in range(self.scale["station_snapshots"]):
            station = self._pick(dock_stations)
            captured = self._moment()
            capacity = max(4, station["dock_capacity"])
            drain = 0.0
            if 7 <= captured.hour <= 9 and zone_type.get(station["zone_id"]) == "residential":
                drain = 0.45
            elif 17 <= captured.hour <= 19 and zone_type.get(station["zone_id"]) == "commercial":
                drain = 0.40
            occupancy = max(0.0, min(1.0, self.rng.betavariate(2.6, 2.6) - drain))
            bikes = int(capacity * occupancy)
            scooters = int(capacity * max(0.0, occupancy - self.rng.uniform(0, 0.4)) * 0.4)
            used = min(capacity, bikes + scooters)
            rows.append({
                "snapshot_id": i + 1,
                "station_id": station["station_id"],
                "captured_at": _iso(captured),
                "bikes_available": bikes,
                "scooters_available": scooters,
                "docks_available": capacity - used,
                "is_empty": used == 0,
                "is_full": capacity - used == 0,
            })
        return rows

    def gen_staff_shifts(self) -> list[dict]:
        rates = {"driver": 24.0, "technician": 28.5, "rebalancer": 18.0,
                 "supervisor": 34.0, "customer_service": 20.5}
        rows = []
        for i in range(self.scale["staff_shifts"]):
            role = self._weighted(tuple(rates), (46, 18, 16, 10, 10))
            start_day = START_DATE + timedelta(days=self.rng.randrange(DAYS))
            start_hour = self.rng.choice([5, 6, 7, 13, 14, 21, 22])
            shift_start = datetime(start_day.year, start_day.month, start_day.day, start_hour, 0)
            length = self.rng.choice([6, 8, 8, 8, 10, 12])
            overtime = _round(max(0.0, self.rng.gauss(0.4, 0.9)), 1)
            first, last = self._pick(FIRST_NAMES), self._pick(LAST_NAMES)
            rows.append({
                "shift_id": i + 1,
                "staff_name": f"{first} {last}",
                "operator_id": self.rng.randrange(1, self.scale["operators"] + 1),
                "zone_id": self.rng.randrange(1, self.scale["zones"] + 1),
                "role": role,
                "shift_start": _iso(shift_start),
                "shift_end": _iso(shift_start + timedelta(hours=length + overtime)),
                "hourly_rate_eur": _round(rates[role] * self.rng.uniform(0.92, 1.18)),
                "overtime_hours": overtime,
            })
        return rows

    # -- orchestration ---------------------------------------------------

    def generate(self) -> dict[str, list[dict[str, Any]]]:
        """Build every table, in dependency order."""
        self.tables["zones"] = self.gen_zones()
        self.tables["stations"] = self.gen_stations()
        self.tables["operators"] = self.gen_operators()
        self.tables["routes"] = self.gen_routes()
        self.tables["route_stops"] = self.gen_route_stops()
        self.tables["vehicle_models"] = self.gen_vehicle_models()
        self.tables["vehicles"] = self.gen_vehicles()
        self.tables["riders"] = self.gen_riders()
        self.tables["subscriptions"] = self.gen_subscriptions()
        self.tables["promotions"] = self.gen_promotions()
        self.tables["fares"] = self.gen_fares()
        self.tables["weather_hourly"] = self.gen_weather()
        self.tables["air_quality_hourly"] = self.gen_air_quality()
        # trips, trip_legs, payments and feedback are generated together.
        self.gen_trips_and_children()
        self.tables["maintenance_events"] = self.gen_maintenance()
        self.tables["charging_sessions"] = self.gen_charging()
        self.tables["incidents"] = self.gen_incidents()
        self.tables["service_alerts"] = self.gen_service_alerts()
        self.tables["station_snapshots"] = self.gen_station_snapshots()
        self.tables["staff_shifts"] = self.gen_staff_shifts()
        return self.tables


def generate_all(seed: int = DEFAULT_SEED, scale: dict[str, int] | None = None) -> dict[str, list[dict]]:
    """Generate the whole warehouse as ``{table_name: [row_dict, ...]}``."""
    return MobilityDataGenerator(seed=seed, scale=scale).generate()
