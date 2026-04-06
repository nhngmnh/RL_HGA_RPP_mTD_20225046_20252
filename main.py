"""
RPP-mTD: Rural Postman Problem with multiple Trucks and Drones
Hybrid Genetic Algorithm — entry point

Dataset demo (URPP-like):
    main.py mặc định load instance N10E30R10_01 từ thư mục dataset/N10.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from algorithms.ga import GA
from algorithms.hga import HGA
from configs.algorithm_params import get_ga_params, get_hga_params
from configs.fleet_params import get_fleet_config
from utils.dataset_loader import load_urpp_like_instance
from utils.results_csv import append_result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  RPP-mTD — GA/HGA Runner")
    print("=" * 55)

    results_path = os.path.join(os.path.dirname(__file__), "results.csv")

    hga_params = get_hga_params()
    ga_params = get_ga_params()

    # --- Batch run: N10E30R10_01.txt .. N10E30R10_05.txt ---
    for idx in range(5, 6):
        instance_filename = f"N20E50R20_{idx:02d}.txt"
        instance_path = os.path.join(
            os.path.dirname(__file__),
            "dataset",
            "N20",
            "N20E50R20",
            instance_filename,
        )
        inst = load_urpp_like_instance(instance_path)

        required_ids = inst.required_edge_ids
        truck_dist = inst.truck_dist
        truck_path = inst.truck_path
        drone_dist = inst.drone_dist
        edge_info = inst.edge_info

        print("\n" + "-" * 55)
        print(f"Instance: {inst.name}")
        print(f"File    : {instance_filename}")
        print(f"Required edges: {len(required_ids)}")

        # --- Fleet ---
        fleet = get_fleet_config(inst.depot_id)
        print(f"Fleet: {fleet.num_trucks} trucks × {fleet.drones_per_truck} drones")
        print(f"Vehicle IDs: {fleet.all_vehicle_ids()}")

        # --- Run HGA ---
        print(f"\nRunning HGA: {hga_params.G} generations, PL={hga_params.PL}...\n")
        hga = HGA(fleet, hga_params, required_ids, truck_dist, drone_dist, edge_info, truck_path)
        t0 = time.perf_counter()
        best = hga.run(verbose=True)
        runtime_s = time.perf_counter() - t0

        print("\n" + "=" * 55)
        print(f"  Algorithm     : HGA")
        print(f"  Best makespan : {best.makespan:.4f} hours")
        print(f"  Fitness       : {best.fitness:.4f}")
        print("=" * 55)

        append_result(
            results_path,
            algorithm="RL_SPC_HGA",
            datasetname=inst.name,
            num_trucks=fleet.num_trucks,
            drones_per_truck=fleet.drones_per_truck,
            makespan_hours=best.makespan,
            fitness=best.fitness,
            runtime_seconds=runtime_s,
            service_seq=best.chromosome.service_sequence,
            vehicle_asgn=best.chromosome.vehicle_assignment,
        )

        # # --- Run GA ---
        # print(f"\nRunning RL_GA: {ga_params.G} generations, PL={ga_params.PL}...\n")
        # ga = GA(fleet, ga_params, required_ids, truck_dist, drone_dist, edge_info, truck_path)
        # t1 = time.perf_counter()
        # best = ga.run(verbose=True)
        # runtime_s = time.perf_counter() - t1

        # print("\n" + "=" * 55)
        # print(f"  Algorithm     : GA")
        # print(f"  Best makespan : {best.makespan:.4f} hours")
        # print(f"  Fitness       : {best.fitness:.4f}")
        # print("=" * 55)

        # append_result(
        #     results_path,
        #     algorithm="RL_SPC_GA",
        #     datasetname=inst.name,
        #     num_trucks=fleet.num_trucks,
        #     drones_per_truck=fleet.drones_per_truck,
        #     makespan_hours=best.makespan,
        #     fitness=best.fitness,
        #     runtime_seconds=runtime_s,
        #     service_seq=best.chromosome.service_sequence,
        #     vehicle_asgn=best.chromosome.vehicle_assignment,
        # )


if __name__ == "__main__":
    main()