from __future__ import annotations
import math
from dataclasses import dataclass, field
from data.chromosome import Chromosome
from data.fleet_config import FleetConfig


@dataclass
class Sortie:
    """Một lần bay của drone."""
    drone_id:    int
    launch_node: int
    launch_time: float
    land_node:   int
    land_time:   float
    edge_ids:    list[int]
    flight_time: float = 0.0
    feasible:    bool  = True


@dataclass
class TruckStop:
    """Một điểm dừng thực sự của truck trên lộ trình (kể cả intermediate nodes)."""
    node:      int
    time:      float    # wall-clock khi truck đến node này
    hop_index: int      # chỉ số hop từ điểm xuất phát (tính theo truck edges)


@dataclass
class TruckRoute:
    """Route đầy đủ của một truck system."""
    truck_id:     int
    stops:        list[TruckStop]
    served_edges: list[int]
    sorties:      list[Sortie] = field(default_factory=list)
    finish_time:  float = 0.0


@dataclass
class DecodedSolution:
    """Kết quả decode từ một chromosome."""
    truck_routes:    list[TruckRoute]
    total_violation: float = 0.0
    makespan:        float = 0.0


class Decoder:
    """
    Chuyển đổi Chromosome → DecodedSolution với timeline song song đúng.

    Điểm mấu chốt:
      - Truck stops = TẤT CẢ nodes truck đi qua thực sự, kể cả intermediate
        nodes trên shortest path giữa các required edges.
      - Drone launch tại truck_stop hiện tại khi sortie bắt đầu.
      - Drone land tại bất kỳ truck_stop nào trong δ hop tiếp theo
        (tính theo số truck-edge hops, không phải số node).
      - Chọn land node xa nhất feasible (thỏa τ và δ).
      - truck_clock tại land_node = max(truck đến, drone đến).
    """

    def __init__(
        self,
        fleet:          FleetConfig,
        truck_dist_fn,     # (u, v) -> float: shortest distance
        drone_dist_fn,     # (u, v) -> float: Euclidean
        edge_info_fn,      # (eid) -> (u, v, length)
        truck_path_fn,     # (u, v) -> list[int]: actual node sequence on shortest path
    ):
        self.fleet       = fleet
        self.truck_dist  = truck_dist_fn
        self.drone_dist  = drone_dist_fn
        self.edge_info   = edge_info_fn
        self.truck_path  = truck_path_fn   # trả về [u, ..., v] inclusive

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
        fleet       = self.fleet
        depot       = fleet.depot_id
        tid         = fleet.truck_id(k)
        system_vids = set(fleet.system_ids(k))

        # Tasks thuộc system k theo thứ tự chromosome
        tasks = [
            (chrom.service_sequence[i], chrom.vehicle_assignment[i])
            for i in range(chrom.length)
            if chrom.vehicle_assignment[i] in system_vids
        ]

        # Gom thành events
        events = self._group_events(tasks, tid)

        # --- Build truck full path với timestamps ---
        # truck_stops: TẤT CẢ nodes truck đi qua, kể cả intermediate
        # hop_index tăng mỗi khi truck traverse một required edge (không phải mỗi node)
        truck_stops: list[TruckStop] = []
        truck_clock = 0.0
        truck_node  = depot
        hop_index   = 0    # số truck-edge đã traverse

        # Precompute toàn bộ truck path trước
        # để drone có thể tra cứu candidate land nodes
        future_stops: list[TruckStop] = [TruckStop(depot, 0.0, 0)]
        sim_node  = depot
        sim_clock = 0.0
        sim_hop   = 0

        for event in events:
            if event[0] == 'truck':
                _, signed_eid = event
                u, v, length  = self._edge_endpoints(signed_eid)

                # Intermediate nodes trên đường đến u
                path_to_u = self.truck_path(sim_node, u)
                seg_time  = self.truck_dist(sim_node, u) / fleet.truck_speed
                if len(path_to_u) > 1:
                    for node in path_to_u[1:]:
                        # Tính thời gian đến từng intermediate node
                        d = self.truck_dist(sim_node, node)
                        future_stops.append(TruckStop(node, sim_clock + d / fleet.truck_speed, sim_hop))

                sim_clock += seg_time
                sim_node   = u

                # Traverse required edge: u → v
                # Intermediate nodes trên edge (nếu edge là polyline)
                # Đơn giản: chỉ thêm endpoint v, hop_index tăng
                sim_clock += length / fleet.truck_speed
                sim_hop   += 1
                sim_node   = v
                future_stops.append(TruckStop(v, sim_clock, sim_hop))

        # Về depot
        path_to_depot = self.truck_path(sim_node, depot)
        seg_time_d    = self.truck_dist(sim_node, depot) / fleet.truck_speed
        if len(path_to_depot) > 1:
            for node in path_to_depot[1:]:
                d = self.truck_dist(sim_node, node)
                future_stops.append(TruckStop(node, sim_clock + d / fleet.truck_speed, sim_hop))
        sim_clock += seg_time_d
        future_stops.append(TruckStop(depot, sim_clock, sim_hop))

        # --- Duyệt events với timeline thực ---
        sorties:      list[Sortie] = []
        served:       list[int]    = []
        truck_clock   = 0.0
        truck_node    = depot
        hop_index     = 0
        # pending_waits: {land_node: drone_land_time} — truck phải chờ tại node này
        pending_waits: dict[int, float] = {}

        for event in events:
            if event[0] == 'truck':
                _, signed_eid = event
                u, v, length  = self._edge_endpoints(signed_eid)

                # Di chuyển đến u qua intermediate nodes
                path_to_u = self.truck_path(truck_node, u)
                for node in path_to_u[1:]:
                    d = self.truck_dist(truck_node, node)
                    truck_clock += d / fleet.truck_speed
                    # Check xem có drone đang chờ ở node này không
                    if node in pending_waits:
                        truck_clock = max(truck_clock, pending_waits.pop(node))
                    truck_node = node

                # Traverse required edge u → v
                truck_clock += length / fleet.truck_speed
                hop_index   += 1
                truck_node   = v
                served.append(signed_eid)

                # Check pending wait tại v
                if v in pending_waits:
                    truck_clock = max(truck_clock, pending_waits.pop(v))

            else:  # sortie
                _, did, edge_group = event

                launch_node = truck_node
                launch_time = truck_clock
                launch_hop  = hop_index

                # Tìm land node tốt nhất trong δ hop từ launch_hop
                land_node, land_hop, land_time_truck = self._find_best_land(
                    launch_node, launch_time, launch_hop,
                    edge_group, future_stops
                )

                # Tính flight_time đầy đủ (kể cả bay về land_node)
                flight_time = self._calc_flight_time(edge_group, launch_node, land_node)

                drone_land_time  = launch_time + flight_time
                rendezvous_time  = max(drone_land_time, land_time_truck)

                feasible = (
                    flight_time <= fleet.max_flight_time
                    and (land_hop - launch_hop) <= fleet.delta
                )

                sorties.append(Sortie(
                    drone_id=did,
                    launch_node=launch_node,
                    launch_time=launch_time,
                    land_node=land_node,
                    land_time=rendezvous_time,
                    edge_ids=edge_group,
                    flight_time=flight_time,
                    feasible=feasible,
                ))

                # Nếu drone chậm hơn truck → đăng ký chờ tại land_node
                if drone_land_time > land_time_truck:
                    existing = pending_waits.get(land_node, 0.0)
                    pending_waits[land_node] = max(existing, drone_land_time)

        # Truck về depot
        path_to_depot = self.truck_path(truck_node, depot)
        for node in path_to_depot[1:]:
            d = self.truck_dist(truck_node, node)
            truck_clock += d / fleet.truck_speed
            if node in pending_waits:
                truck_clock = max(truck_clock, pending_waits.pop(node))
            truck_node = node

        # finish_time = max(truck về depot, tất cả drone land)
        max_drone = max((s.land_time for s in sorties), default=0.0)
        finish_time = max(truck_clock, max_drone)

        return TruckRoute(
            truck_id=tid,
            stops=future_stops,
            served_edges=served,
            sorties=sorties,
            finish_time=finish_time,
        )

    # ------------------------------------------------------------------
    # Tìm land node tốt nhất
    # ------------------------------------------------------------------

    def _find_best_land(
        self,
        launch_node:  int,
        launch_time:  float,
        launch_hop:   int,
        edge_group:   list[int],
        future_stops: list[TruckStop],
    ) -> tuple[int, int, float]:
        """
        Duyệt tất cả truck_stops sau launch_hop trong δ hop,
        chọn stop tối ưu theo 2 tiêu chí:
          1. Minimize rendezvous_time = max(drone_land_time, truck_time_at_stop)
             → minimize thời điểm cả hai gặp nhau, tức minimize makespan tăng thêm
          2. Tiebreak: hop_index lớn nhất (truck đi được xa hơn trong khi chờ)

        Fallback: land tại launch_node nếu không có stop nào trong δ hop feasible.
        """
        fleet = self.fleet

        flight_to_end = self._calc_flight_time_to_end(edge_group, launch_node)
        _, last_v, _  = self._edge_endpoints(edge_group[-1])

        # Fallback: land tại launch_node — drone bay ra rồi quay về chỗ cũ
        fly_back_to_launch = self.drone_dist(last_v, launch_node) / fleet.drone_speed
        fallback_flight    = flight_to_end + fly_back_to_launch
        fallback_rendezvous = launch_time + fallback_flight  # truck vẫn đứng đó

        best_node       = launch_node
        best_hop        = launch_hop
        best_truck_time = launch_time
        best_rendezvous = fallback_rendezvous

        for stop in future_stops:
            # Chỉ xét stops SAU launch theo hop (sequence constraint)
            if stop.hop_index <= launch_hop:
                continue
            if stop.hop_index - launch_hop > fleet.delta:
                break  # future_stops đã sắp xếp hop tăng dần

            fly_back     = self.drone_dist(last_v, stop.node) / fleet.drone_speed
            total_flight = flight_to_end + fly_back

            # Ràng buộc battery τ
            if total_flight > fleet.max_flight_time:
                continue

            drone_land_time = launch_time + total_flight
            rendezvous_time = max(drone_land_time, stop.time)

            # Chọn nếu tốt hơn theo tiêu chí 1, hoặc bằng nhau nhưng xa hơn (tiêu chí 2)
            if (rendezvous_time < best_rendezvous - 1e-9) or (
                abs(rendezvous_time - best_rendezvous) < 1e-9
                and stop.hop_index > best_hop
            ):
                best_node       = stop.node
                best_hop        = stop.hop_index
                best_truck_time = stop.time
                best_rendezvous = rendezvous_time

        return best_node, best_hop, best_truck_time

    # ------------------------------------------------------------------
    # Tính flight time
    # ------------------------------------------------------------------

    def _calc_flight_time_to_end(
        self, edge_group: list[int], launch_node: int
    ) -> float:
        """Flight time từ launch đến end của edge cuối (chưa tính bay về land)."""
        speed = self.fleet.drone_speed
        time  = 0.0
        cur   = launch_node

        for signed_eid in edge_group:
            u, v, length = self._edge_endpoints(signed_eid)
            time += self.drone_dist(cur, u) / speed
            time += length / speed
            cur   = v

        return time

    def _calc_flight_time(
        self, edge_group: list[int], launch_node: int, land_node: int
    ) -> float:
        """Flight time đầy đủ: launch → [edges] → land_node."""
        time = self._calc_flight_time_to_end(edge_group, launch_node)
        _, last_v, _ = self._edge_endpoints(edge_group[-1])
        time += self.drone_dist(last_v, land_node) / self.fleet.drone_speed
        return time

    # ------------------------------------------------------------------
    # Group tasks thành events
    # ------------------------------------------------------------------

    @staticmethod
    def _group_events(tasks: list[tuple[int, int]], tid: int) -> list[tuple]:
        """
        ('truck', signed_eid)
        ('sortie', did, [signed_eids])

        Drone tasks liên tiếp cùng drone → 1 sortie.
        Khi gặp truck task hoặc drone khác → flush.
        """
        events      = []
        current_did = None
        current_run = []

        def flush():
            if current_did is not None and current_run:
                events.append(('sortie', current_did, current_run[:]))

        for signed_eid, vid in tasks:
            if vid == tid:
                flush()
                current_did = None
                current_run = []
                events.append(('truck', signed_eid))
            else:
                if vid != current_did:
                    flush()
                    current_did = vid
                    current_run = []
                current_run.append(signed_eid)

        flush()
        return events

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _edge_endpoints(self, signed_eid: int) -> tuple[int, int, float]:
        u, v, length = self.edge_info(abs(signed_eid))
        return (u, v, length) if signed_eid > 0 else (v, u, length)