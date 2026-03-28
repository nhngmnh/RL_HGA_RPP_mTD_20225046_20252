"""
RPP-mTD: Rural Postman Problem with multiple Trucks and Drones
Hybrid Genetic Algorithm — entry point

Usage:
    python main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data import Chromosome, Individual, FleetConfig, HGAParams


def main():
    # --- Cấu hình fleet ---
    fleet = FleetConfig(
        num_trucks=2,
        drones_per_truck=2,
        max_flight_time=1.0,   # 1 giờ
        delta=5,
        truck_speed=40.0,
        drone_speed=80.0,
        depot_id=0,
    )

    # --- Hyperparameters ---
    params = HGAParams()

    print("Fleet:", fleet)
    print("Vehicles:", fleet.all_vehicle_ids())
    print("Params:", params)
    print()
    print("TODO: khởi tạo graph, chạy HGA")


if __name__ == "__main__":
    main()
