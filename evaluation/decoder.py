from __future__ import annotations
import math
from dataclasses import dataclass, field
from data.chromosome import Chromosome
from data.fleet_config import FleetConfig


@dataclass
class Sortie:
    """Một lần bay của drone: từ launch_node đến land_node, phục vụ các edge."""
    drone_id:    int
    launch_node: int
    land_node:   int
    edge_ids:    list[int]        # signed edge ids theo thứ tự phục vụ
    flight_time: float = 0.0
    feasible:    bool  = True     # False nếu vi phạm τ hoặc δ


@dataclass
class TruckRoute:
    """Route đầy đủ của một truck và các drone của nó."""
    truck_id:      int
    node_sequence: list[int]      # các node truck đi qua (kể cả depot)
    served_edges:  list[int]      # signed edge ids truck trực tiếp phục vụ
    sorties:       list[Sortie] = field(default_factory=list)
    truck_time:    float = 0.0    # tổng thời gian truck (không tính chờ drone)
    finish_time:   float = 0.0    # thời gian truck về đến depot


@dataclass
class DecodedSolution:
    """Kết quả decode từ một chromosome."""
    truck_routes:        list[TruckRoute]
    total_violation:     float = 0.0   # tổng vi phạm τ (dùng cho penalty)
    makespan:            float = 0.0   # max finish time qua tất cả vehicles


class Decoder:
    """
    Chuyển đổi Chromosome → DecodedSolution.

    Quy trình:
      1. Tách chromosome theo truck system
      2. Với mỗi truck system, xây dựng TruckRoute:
         a. Truck đi theo thứ tự các edge được assign cho nó
            (di chuyển qua shortest path giữa các edge)
         b. Drone thực hiện các sortie, mỗi sortie = chuỗi edge liên tiếp
            được assign cho drone đó, launch/land tại node truck
      3. Tính finish time cho từng vehicle
      4. Makespan = max(tất cả finish times)

    Giả định đơn giản hóa cho sorties:
      - Drone launch khi truck đến start node của edge đầu tiên trong sortie
      - Drone land tại node truck sẽ đứng sau δ hop (hoặc node cuối truck nếu ít hơn)
      - Nếu flight_time > τ → đánh dấu infeasible, tính violation
    """

    def __init__(
        self,
        fleet:        FleetConfig,
        truck_dist_fn,    # (u, v) -> float: shortest path distance trên road network
        drone_dist_fn,    # (u, v) -> float: Euclidean distance
        edge_info_fn,     # (eid) -> (u, v, length): thông tin edge
    ):
        self.fleet      = fleet
        self.truck_dist = truck_dist_fn
        self.drone_dist = drone_dist_fn
        self.edge_info  = edge_info_fn  # trả về (u, v, length)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def decode(self, chrom: Chromosome) -> DecodedSolution:
        truck_routes = []

        for k in range(1, self.fleet.num_trucks + 1):
            route = self._decode_system(k, chrom)
            truck_routes.append(route)

        total_violation = sum(
            s.flight_time - self.fleet.max_flight_time
            for r in truck_routes
            for s in r.sorties
            if not s.feasible
        )

        makespan = max(r.finish_time for r in truck_routes) if truck_routes else 0.0

        return DecodedSolution(
            truck_routes=truck_routes,
            total_violation=total_violation,
            makespan=makespan,
        )

    # ------------------------------------------------------------------
    # Decode một truck system
    # ------------------------------------------------------------------

    def _decode_system(self, k: int, chrom: Chromosome) -> TruckRoute:
        fleet    = self.fleet
        depot    = fleet.depot_id
        tid      = fleet.truck_id(k)
        did_list = [fleet.drone_id(k, m) for m in range(1, fleet.drones_per_truck + 1)]
        system_vids = set(fleet.system_ids(k))

        # Tách các task thuộc system k, giữ thứ tự trong chromosome
        tasks = [
            (chrom.service_sequence[i], chrom.vehicle_assignment[i])
            for i in range(chrom.length)
            if chrom.vehicle_assignment[i] in system_vids
        ]

        # Nhóm các task liên tiếp của từng drone thành sorties
        # Truck task → phục vụ trực tiếp
        # Drone task liên tiếp (cùng drone) → gom thành một sortie
        truck_tasks: list[int] = []   # signed edge ids truck phục vụ
        drone_sorties_raw: dict[int, list[list[int]]] = {did: [] for did in did_list}

        # Gom drone tasks thành sorties: chuỗi liên tiếp cùng drone = 1 sortie
        current_drone_run: dict[int, list[int]] = {did: [] for did in did_list}

        for signed_eid, vid in tasks:
            if vid == tid:
                # Flush drone runs khi gặp truck task
                for did in did_list:
                    if current_drone_run[did]:
                        drone_sorties_raw[did].append(current_drone_run[did][:])
                        current_drone_run[did] = []
                truck_tasks.append(signed_eid)
            else:
                # Flush các drone khác nếu đang chạy
                for did2 in did_list:
                    if did2 != vid and current_drone_run[did2]:
                        drone_sorties_raw[did2].append(current_drone_run[did2][:])
                        current_drone_run[did2] = []
                current_drone_run[vid].append(signed_eid)

        # Flush cuối
        for did in did_list:
            if current_drone_run[did]:
                drone_sorties_raw[did].append(current_drone_run[did][:])

        # Xây dựng truck route
        truck_node = depot
        truck_time = 0.0
        node_seq   = [depot]
        served     = []

        # Theo dõi node truck sau mỗi hop để xác định rendezvous
        truck_hop_nodes: list[int] = [depot]  # node sau từng hop

        for signed_eid in truck_tasks:
            u, v, length = self._edge_endpoints(signed_eid)
            # Di chuyển đến start
            travel = self.truck_dist(truck_node, u) / self.fleet.truck_speed
            truck_time += travel
            # Traverse edge
            truck_time += length / self.fleet.truck_speed
            truck_node = v
            node_seq.append(v)
            truck_hop_nodes.append(v)
            served.append(signed_eid)

        # Về depot
        truck_time += self.truck_dist(truck_node, depot) / self.fleet.truck_speed
        node_seq.append(depot)

        # Xây dựng sorties cho từng drone
        sorties: list[Sortie] = []
        drone_times: dict[int, float] = {did: 0.0 for did in did_list}

        for did, sortie_groups in drone_sorties_raw.items():
            for group in sortie_groups:
                sortie = self._build_sortie(did, group, truck_hop_nodes)
                # Drone không thể bay trước khi truck launch
                # (ước lượng đơn giản: drone bắt đầu từ thời điểm truck ở launch_node)
                sortie.feasible = sortie.flight_time <= self.fleet.max_flight_time
                sorties.append(sortie)
                drone_times[did] += sortie.flight_time

        # finish_time = max(truck về depot, drone bay xong)
        max_drone_time = max(drone_times.values()) if drone_times else 0.0
        finish_time = max(truck_time, max_drone_time)

        return TruckRoute(
            truck_id=tid,
            node_sequence=node_seq,
            served_edges=served,
            sorties=sorties,
            truck_time=truck_time,
            finish_time=finish_time,
        )

    # ------------------------------------------------------------------
    # Build một sortie
    # ------------------------------------------------------------------

    def _build_sortie(
        self,
        drone_id: int,
        edge_group: list[int],   # signed edge ids
        truck_hop_nodes: list[int],
    ) -> Sortie:
        """
        Tính flight_time cho một sortie.
        Đơn giản hóa: drone bay Euclidean giữa các edge,
        traverse required edge theo độ dài thực.
        Launch từ truck_hop_nodes[0] (depot mặc định nếu không rõ).
        """
        if not edge_group:
            return Sortie(drone_id, 0, 0, [], 0.0, True)

        # Launch node: node đầu tiên của edge đầu tiên trong group
        first_u, _, _ = self._edge_endpoints(edge_group[0])
        launch_node   = truck_hop_nodes[0] if truck_hop_nodes else self.fleet.depot_id

        flight_time = 0.0
        # Bay từ launch đến start của edge đầu tiên
        flight_time += self.drone_dist(launch_node, first_u) / self.fleet.drone_speed

        current = first_u
        for signed_eid in edge_group:
            u, v, length = self._edge_endpoints(signed_eid)
            # Bay đến start nếu chưa đứng ở đó
            if current != u:
                flight_time += self.drone_dist(current, u) / self.fleet.drone_speed
            # Traverse edge (theo độ dài thực)
            flight_time += length / self.fleet.drone_speed
            current = v

        # Bay về rendezvous node (ước lượng: land tại launch_node nếu không rõ)
        land_node = truck_hop_nodes[min(self.fleet.delta, len(truck_hop_nodes) - 1)]
        flight_time += self.drone_dist(current, land_node) / self.fleet.drone_speed

        return Sortie(
            drone_id=drone_id,
            launch_node=launch_node,
            land_node=land_node,
            edge_ids=edge_group,
            flight_time=flight_time,
            feasible=flight_time <= self.fleet.max_flight_time,
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _edge_endpoints(self, signed_eid: int) -> tuple[int, int, float]:
        """Trả về (start, end, length) theo chiều traverse của chromosome."""
        u, v, length = self.edge_info(abs(signed_eid))
        if signed_eid > 0:
            return u, v, length
        else:
            return v, u, length

