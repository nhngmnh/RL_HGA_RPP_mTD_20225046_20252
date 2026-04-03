"""
RPP-mTD: Rural Postman Problem with multiple Trucks and Drones
Hybrid Genetic Algorithm — entry point

Dataset demo (URPP-like):
    main.py mặc định load instance N10E30R10_01 từ thư mục dataset/N10.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data import FleetConfig, HGAParams
from hga import HGA
from utils.dataset_loader import load_urpp_like_instance

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  RPP-mTD — Hybrid Genetic Algorithm")
    print("=" * 55)

    # --- Dataset instance ---
    instance_path = os.path.join(
        os.path.dirname(__file__),
        "dataset",
        "N10",
        "N10E30R10",
        "N10E30R10_01.txt",
    )
    inst = load_urpp_like_instance(instance_path)

    required_ids = inst.required_edge_ids
    truck_dist = inst.truck_dist
    truck_path = inst.truck_path
    drone_dist = inst.drone_dist
    edge_info = inst.edge_info

    print(f"Instance: {inst.name}")
    print(f"Required edges: {len(required_ids)}")

    # --- Fleet ---
    fleet = FleetConfig(
        num_trucks=2,
        drones_per_truck=1,
        max_flight_time=1.0,
        delta=3,
        truck_speed=40.0,
        drone_speed=80.0,
        depot_id=inst.depot_id,
    )
    print(f"Fleet: {fleet.num_trucks} trucks × {fleet.drones_per_truck} drones")
    print(f"Vehicle IDs: {fleet.all_vehicle_ids()}")

    # --- HGA params ---
    params = HGAParams(
        PL=100, PH=200,
        G=100,
        Gm=10,
        pt=0.1,
        pm=0.2, pm_plus=0.3,
        seed=42,
    )

    # --- Run HGA ---
    print(f"\nRunning HGA: {params.G} generations, PL={params.PL}...\n")
    hga = HGA(fleet, params, required_ids, truck_dist, drone_dist, edge_info, truck_path)
    best = hga.run(verbose=True)

    # --- Results ---
    print("\n" + "=" * 55)
    print(f"  Best makespan : {best.makespan:.4f} hours")
    print(f"  Fitness       : {best.fitness:.4f}")
    print(f"  Service seq   : {best.chromosome.service_sequence}")
    print(f"  Vehicle asgn  : {best.chromosome.vehicle_assignment}")
    print("=" * 55)


if __name__ == "__main__":
    main()