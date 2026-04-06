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
from algorithms.rl_hga import RLHGA
from algorithms.rl_ga import RLGA
from configs.algorithm_params import get_ga_params, get_hga_params
from configs.fleet_params import get_fleet_config
from utils.dataset_loader import load_urpp_like_instance
from utils.results_csv import append_result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  RPP-mTD — GA/RLGA + HGA/RLHGA Runner")
    print("=" * 55)

    results_path = os.path.join(os.path.dirname(__file__), "results.csv")

    hga_params = get_hga_params()
    ga_params = get_ga_params()

    # --- Batch run: N10E30R10_01.txt .. N10E30R10_05.txt ---
    for idx in range(1, 6):
        instance_filename = f"N10E30R10_{idx:02d}.txt"
        instance_path = os.path.join(
            os.path.dirname(__file__),
            "dataset",
            "N10",
            "N10E30R10",
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

        # --- Run HGA (pure) ---
        print(f"\nRunning HGA: {hga_params.G} generations, PL={hga_params.PL}...\n")
        hga = HGA(fleet, hga_params, required_ids, truck_dist, drone_dist, edge_info, truck_path)
        t0 = time.perf_counter()
        best_hga = hga.run(verbose=True)
        runtime_hga_s = time.perf_counter() - t0

        print("\n" + "=" * 55)
        print("  Algorithm     : HGA")
        print(f"  Best makespan : {best_hga.makespan:.4f} hours")
        print(f"  Fitness       : {best_hga.fitness:.4f}")
        print("=" * 55)

        append_result(
            results_path,
            algorithm="HGA",
            datasetname=inst.name,
            num_trucks=fleet.num_trucks,
            drones_per_truck=fleet.drones_per_truck,
            makespan_hours=best_hga.makespan,
            fitness=best_hga.fitness,
            runtime_seconds=runtime_hga_s,
            service_seq=best_hga.chromosome.service_sequence,
            vehicle_asgn=best_hga.chromosome.vehicle_assignment,
        )

        # --- Run RLHGA ---
        print(f"\nRunning RLHGA: {hga_params.G} generations, PL={hga_params.PL}...\n")
        rlhga = RLHGA(fleet, hga_params, required_ids, truck_dist, drone_dist, edge_info, truck_path)
        t1 = time.perf_counter()
        best_rl = rlhga.run(verbose=True)
        runtime_rl_s = time.perf_counter() - t1

        print("\n" + "=" * 55)
        print("  Algorithm     : RLHGA")
        print(f"  Best makespan : {best_rl.makespan:.4f} hours")
        print(f"  Fitness       : {best_rl.fitness:.4f}")
        print("=" * 55)

        print("\n" + "-" * 55)
        print(f"Compare (RLHGA - HGA) makespan: {best_rl.makespan - best_hga.makespan:+.4f} hours")
        print(f"Compare (RLHGA - HGA) runtime : {runtime_rl_s - runtime_hga_s:+.2f} s")
        print("-" * 55)

        append_result(
            results_path,
            algorithm="RLHGA",
            datasetname=inst.name,
            num_trucks=fleet.num_trucks,
            drones_per_truck=fleet.drones_per_truck,
            makespan_hours=best_rl.makespan,
            fitness=best_rl.fitness,
            runtime_seconds=runtime_rl_s,
            service_seq=best_rl.chromosome.service_sequence,
            vehicle_asgn=best_rl.chromosome.vehicle_assignment,
        )

        # # --- Run GA (pure) ---
        # print(f"\nRunning GA: {ga_params.G} generations, PL={ga_params.PL}...\n")
        # ga = GA(fleet, ga_params, required_ids, truck_dist, drone_dist, edge_info, truck_path)
        # t1 = time.perf_counter()
        # best_ga = ga.run(verbose=True)
        # runtime_ga_s = time.perf_counter() - t1

        # print("\n" + "=" * 55)
        # print("  Algorithm     : GA")
        # print(f"  Best makespan : {best_ga.makespan:.4f} hours")
        # print(f"  Fitness       : {best_ga.fitness:.4f}")
        # print("=" * 55)

        # append_result(
        #     results_path,
        #     algorithm="GA",
        #     datasetname=inst.name,
        #     num_trucks=fleet.num_trucks,
        #     drones_per_truck=fleet.drones_per_truck,
        #     makespan_hours=best_ga.makespan,
        #     fitness=best_ga.fitness,
        #     runtime_seconds=runtime_ga_s,
        #     service_seq=best_ga.chromosome.service_sequence,
        #     vehicle_asgn=best_ga.chromosome.vehicle_assignment,
        # )

        # # --- Run RLGA ---
        # print(f"\nRunning RLGA: {ga_params.G} generations, PL={ga_params.PL}...\n")
        # rlga = RLGA(fleet, ga_params, required_ids, truck_dist, drone_dist, edge_info, truck_path)
        # t2 = time.perf_counter()
        # best_rlga = rlga.run(verbose=True)
        # runtime_rlga_s = time.perf_counter() - t2

        # print("\n" + "=" * 55)
        # print("  Algorithm     : RLGA")
        # print(f"  Best makespan : {best_rlga.makespan:.4f} hours")
        # print(f"  Fitness       : {best_rlga.fitness:.4f}")
        # print("=" * 55)

        # print("\n" + "-" * 55)
        # # print(f"Compare (RLGA - GA) makespan: {best_rlga.makespan - best_ga.makespan:+.4f} hours")
        # # print(f"Compare (RLGA - GA) runtime : {runtime_rlga_s - runtime_ga_s:+.2f} s")
        # print("-" * 55)

        # append_result(
        #     results_path,
        #     algorithm="RLGA",
        #     datasetname=inst.name,
        #     num_trucks=fleet.num_trucks,
        #     drones_per_truck=fleet.drones_per_truck,
        #     makespan_hours=best_rlga.makespan,
        #     fitness=best_rlga.fitness,
        #     runtime_seconds=runtime_rlga_s,
        #     service_seq=best_rlga.chromosome.service_sequence,
        #     vehicle_asgn=best_rlga.chromosome.vehicle_assignment,
        # )


if __name__ == "__main__":
    main()