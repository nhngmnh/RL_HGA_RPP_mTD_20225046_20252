"""
RPP-mTD: Rural Postman Problem with multiple Trucks and Drones
Hybrid Genetic Algorithm — entry point

Cách dùng:
    python main.py

Để tích hợp với graph thực (NetworkX, etc.):
    Thay 3 hàm mock bên dưới bằng hàm thực từ graph của bạn:
        truck_dist(u, v) -> float
        drone_dist(u, v) -> float
        edge_info(eid)   -> (u, v, length)
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(__file__))

from data import FleetConfig, HGAParams
from hga import HGA


# ---------------------------------------------------------------------------
# Mock graph — thay bằng graph thực của bạn
# ---------------------------------------------------------------------------

def build_mock_graph():
    """
    Graph ví dụ nhỏ: 10 nodes, 12 edges, 5 required edges.
    Coordinates trên lưới 10x10 km.
    """
    coords = {
        0:  (5.0, 5.0),   # depot
        1:  (0.0, 8.0),
        2:  (2.0, 8.0),
        3:  (4.0, 8.0),
        4:  (7.0, 8.0),
        5:  (9.0, 8.0),
        6:  (1.0, 4.0),
        7:  (5.0, 4.0),
        8:  (9.0, 4.0),
        9:  (3.0, 1.0),
        10: (7.0, 1.0),
    }

    # (eid, u, v, required)
    edges = [
        (1,  0, 7,  False),
        (2,  1, 2,  True),
        (3,  2, 3,  True),
        (4,  3, 4,  False),
        (5,  4, 5,  True),
        (6,  5, 8,  True),
        (7,  6, 7,  True),
        (8,  7, 8,  False),
        (9,  6, 9,  False),
        (10, 7, 10, False),
        (11, 9, 10, False),
        (12, 8, 10, False),
    ]

    # Tính Manhattan distance
    edge_map = {}
    for eid, u, v, req in edges:
        xu, yu = coords[u]
        xv, yv = coords[v]
        length = abs(xu - xv) + abs(yu - yv)
        edge_map[eid] = (u, v, length)

    required_ids = [eid for eid, u, v, req in edges if req]

    # Adjacency list cho Dijkstra
    adj = {i: [] for i in range(11)}
    for eid, u, v, req in edges:
        length = edge_map[eid][2]
        adj[u].append((v, length))
        adj[v].append((u, length))

    # Precompute shortest paths (Dijkstra đơn giản)
    import heapq
    def dijkstra(src):
        dist = {i: math.inf for i in range(11)}
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if d + w < dist[v]:
                    dist[v] = d + w
                    heapq.heappush(pq, (dist[v], v))
        return dist

    sp = {i: dijkstra(i) for i in range(11)}

    def truck_dist(u, v):
        return sp[u][v]

    def drone_dist(u, v):
        xu, yu = coords[u]
        xv, yv = coords[v]
        return math.hypot(xu - xv, yu - yv)

    def edge_info(eid):
        return edge_map[eid]

    return required_ids, truck_dist, drone_dist, edge_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  RPP-mTD — Hybrid Genetic Algorithm")
    print("=" * 55)

    # --- Graph ---
    required_ids, truck_dist, drone_dist, edge_info = build_mock_graph()
    print(f"Required edges: {required_ids}")

    # --- Fleet ---
    fleet = FleetConfig(
        num_trucks=2,
        drones_per_truck=1,
        max_flight_time=1.0,
        delta=3,
        truck_speed=40.0,
        drone_speed=80.0,
        depot_id=0,
    )
    print(f"Fleet: {fleet.num_trucks} trucks × {fleet.drones_per_truck} drones")
    print(f"Vehicle IDs: {fleet.all_vehicle_ids()}")

    # --- HGA params ---
    params = HGAParams(
        PL=20, PH=40,
        G=50,
        Gm=10,
        pt=0.1,
        pm=0.1, pm_plus=0.3,
        seed=42,
    )

    # --- Run HGA ---
    print(f"\nRunning HGA: {params.G} generations, PL={params.PL}...\n")
    hga = HGA(fleet, params, required_ids, truck_dist, drone_dist, edge_info)
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

