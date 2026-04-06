from __future__ import annotations

from data import FleetConfig


def get_fleet_config(depot_id: int) -> FleetConfig:
    """Default fleet configuration used by both GA and HGA runs."""
    return FleetConfig(
        num_trucks=2,
        drones_per_truck=1,
        max_flight_time=1.0,
        delta=5,
        truck_speed=40.0,
        drone_speed=80.0,
        depot_id=depot_id,
    )
