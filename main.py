"""
RPP-mTD: Rural Postman Problem with multiple Trucks and Drones
Hybrid Genetic Algorithm — entry point

Cách dùng:
    python main.py

Để tích hợp với graph thực (NetworkX, etc.):
    Thay 4 hàm mock bên dưới bằng hàm thực từ graph của bạn:
        truck_dist(u, v) -> float
        truck_path(u, v) -> list[int]  (danh sách nodes trên shortest path)
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
    coords = {
    0:  (5.0, 5.0),   # depot (center)

    # Hàng trên
    1:  (0.0, 10.0),  2: (2.5, 10.0), 3: (5.0, 10.0), 4: (7.5, 10.0), 5: (10.0, 10.0),

    # Hàng trên giữa
    6:  (0.0, 7.5),   7: (2.5, 7.5),  8: (5.0, 7.5),  9: (7.5, 7.5),  10:(10.0, 7.5),

    # Hàng giữa
    11: (0.0, 5.0),   12:(2.5, 5.0),  13:(7.5, 5.0),  14:(10.0, 5.0),

    # Hàng dưới giữa
    15: (0.0, 2.5),   16:(2.5, 2.5),  17:(5.0, 2.5),  18:(7.5, 2.5),  19:(10.0, 2.5),

    # Hàng dưới
    20: (0.0, 0.0),   21:(2.5, 0.0),  22:(5.0, 0.0),  23:(7.5, 0.0),  24:(10.0, 0.0),
}
    # (eid, u, v, required)
    edges = [
    # --- Horizontal edges (trên cùng)
    (1, 1, 2, True),
    (2, 2, 3, True),
    (3, 3, 4, False),
    (4, 4, 5, True),

    # --- Horizontal (trên giữa)
    (5, 6, 7, True),
    (6, 7, 8, False),
    (7, 8, 9, True),
    (8, 9, 10, True),

    # --- Horizontal (giữa)
    (9, 11, 12, False),
    (10, 12, 0, True),
    (11, 0, 13, True),
    (12, 13, 14, False),

    # --- Horizontal (dưới giữa)
    (13, 15, 16, True),
    (14, 16, 17, False),
    (15, 17, 18, True),
    (16, 18, 19, True),

    # --- Horizontal (dưới cùng)
    (17, 20, 21, False),
    (18, 21, 22, True),
    (19, 22, 23, True),
    (20, 23, 24, False),

    # --- Vertical edges
    (21, 1, 6, False),
    (22, 6, 11, True),
    (23, 11, 15, True),
    (24, 15, 20, False),

    (25, 2, 7, True),
    (26, 7, 12, False),
    (27, 12, 16, True),
    (28, 16, 21, True),

    (29, 3, 8, False),
    (30, 8, 0, True),
    (31, 0, 17, True),
    (32, 17, 22, False),

    (33, 4, 9, True),
    (34, 9, 13, False),
    (35, 13, 18, True),
    (36, 18, 23, True),

    (37, 5, 10, False),
    (38, 10, 14, True),
    (39, 14, 19, True),
    (40, 19, 24, False),

    # --- Một vài đường chéo (tăng độ khó cho drone)
    (41, 6, 12, False),
    (42, 7, 13, True),
    (43, 8, 14, False),
    (44, 11, 16, True),
    (45, 12, 17, False),
    (46, 13, 19, True),
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
    nodes = set(coords.keys())
    for _, u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)

    adj = {n: [] for n in nodes}
    for eid, u, v, req in edges:
        length = edge_map[eid][2]
        adj[u].append((v, length))
        adj[v].append((u, length))

    # Precompute shortest paths + predecessor (Dijkstra đơn giản)
    import heapq
    def dijkstra(src):
        dist = {n: math.inf for n in nodes}
        prev = {src: None}
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    sp = {}
    prevs = {}
    for n in sorted(nodes):
        dist, prev = dijkstra(n)
        sp[n] = dist
        prevs[n] = prev

    def truck_dist(u, v):
        return sp[u][v]

    def truck_path(u, v):
        """Trả về danh sách nodes trên shortest path u -> v."""
        if u == v:
            return [u]
        prev = prevs[u]
        if v not in prev:
            return [u, v]
        path = [v]
        cur = v
        while cur != u:
            cur = prev.get(cur)
            if cur is None:
                return [u, v]
            path.append(cur)
        return list(reversed(path))

    def drone_dist(u, v):
        xu, yu = coords[u]
        xv, yv = coords[v]
        return math.hypot(xu - xv, yu - yv)

    def edge_info(eid):
        return edge_map[eid]

    return required_ids, truck_dist, truck_path, drone_dist, edge_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  RPP-mTD — Hybrid Genetic Algorithm")
    print("=" * 55)

    # --- Graph ---
    required_ids, truck_dist, truck_path, drone_dist, edge_info = build_mock_graph()
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
        PL=30, PH=60,
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