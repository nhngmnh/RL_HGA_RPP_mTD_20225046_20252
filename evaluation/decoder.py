from __future__ import annotations
import math
from dataclasses import dataclass, field
from data.chromosome import Chromosome
from data.fleet_config import FleetConfig


@dataclass
class Sortie:
    """Một lần bay của drone."""
    drone_id:           int
    launch_node:        int
    launch_time:        float
    service_start_time: float
    land_node:          int
    land_time:          float
    edge_ids:           list[int]
    flight_time:        float = 0.0
    feasible:           bool  = True


@dataclass
class TruckStop:
    """Một điểm trên lộ trình thực sự của truck."""
    node:       int
    time:       float   # wall-clock (được cập nhật khi có wait)
    edge_count: int     # số required-edge truck đã traverse tính đến đây


@dataclass
class TruckRoute:
    truck_id:     int
    stops:        list[TruckStop]
    served_edges: list[int]
    sorties:      list[Sortie] = field(default_factory=list)
    finish_time:  float = 0.0


@dataclass
class DecodedSolution:
    truck_routes:    list[TruckRoute]
    total_violation: float = 0.0
    makespan:        float = 0.0


class Decoder:
    """
    Decoder cho RPP-mTD.

    δ-hop theo đúng paper:
        Sau khi launch drone tại một node, truck được phép traverse
        TỐI ĐA δ required edges trước khi rendezvous với drone.
        Intermediate nodes trên shortest path KHÔNG tính là hop.

    Mỗi TruckStop lưu edge_count = số required-edge truck đã traverse.
    Khi tìm land candidates cho một sortie launch tại stop L:
        land candidates = stops S thỏa  S.edge_count - L.edge_count <= δ
    """

    def __init__(
        self,
        fleet:         FleetConfig,
        truck_dist_fn,
        drone_dist_fn,
        edge_info_fn,
        truck_path_fn,
    ):
        self.fleet      = fleet
        self.truck_dist = truck_dist_fn
        self.drone_dist = drone_dist_fn
        self.edge_info  = edge_info_fn
        self.truck_path = truck_path_fn

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def decode(self, chrom: Chromosome, w_inf: float = 1.0) -> DecodedSolution:
        truck_routes = []
        for k in range(1, self.fleet.num_trucks + 1):
            route = self._decode_system(k, chrom, w_inf)
            truck_routes.append(route)

        total_violation = sum(
            max(0.0, s.flight_time - self.fleet.max_flight_time)
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

    def _decode_system(self, k: int, chrom: Chromosome, w_inf: float = 1.0) -> TruckRoute:
        fleet       = self.fleet
        depot       = fleet.depot_id
        tid         = fleet.truck_id(k)
        system_vids = set(fleet.system_ids(k))

        tasks = [
            (chrom.service_sequence[i], chrom.vehicle_assignment[i])
            for i in range(chrom.length)
            if chrom.vehicle_assignment[i] in system_vids
        ]
        events = self._group_events(tasks, tid)

        # Skeleton truck path (không tính wait)
        raw_stops = self._build_raw_stops(events, depot)

        # Timeline thực
        served:        list[int]         = []
        sorties:       list[Sortie]      = []
        truck_clock    = 0.0
        truck_node     = depot
        truck_edge_cnt = 0               # số required-edge truck đã traverse
        t_prev_service = 0.0
        pending_waits: dict[int, float]  = {}

        for event in events:
            if event[0] == 'truck':
                _, signed_eid = event
                u, v, length  = self._edge_endpoints(signed_eid)

                for node in self.truck_path(truck_node, u)[1:]:
                    d = self.truck_dist(truck_node, node)
                    truck_clock += d / fleet.truck_speed
                    if node in pending_waits:
                        truck_clock = max(truck_clock, pending_waits.pop(node))
                    truck_node = node

                t_prev_service  = truck_clock
                truck_clock    += length / fleet.truck_speed
                truck_edge_cnt += 1
                truck_node      = v
                served.append(signed_eid)

                if v in pending_waits:
                    truck_clock = max(truck_clock, pending_waits.pop(v))

            else:  # sortie
                _, did, edge_group = event

                sortie = self._build_sortie(
                    did, edge_group,
                    t_prev_service,
                    truck_clock, truck_node, truck_edge_cnt,
                    raw_stops,
                    w_inf,
                )
                sorties.append(sortie)

                t_prev_service = sortie.service_start_time

                # Propagate wait vào raw_stops
                drone_done = sortie.launch_time + sortie.flight_time
                if drone_done > sortie.land_time - 1e-9:
                    land_idx = next(
                        (i for i, s in enumerate(raw_stops)
                         if s.node == sortie.land_node
                         and s.edge_count == sortie._land_edge_count),
                        None
                    )
                    if land_idx is not None:
                        wait = drone_done - raw_stops[land_idx].time
                        if wait > 1e-9:
                            for i in range(land_idx, len(raw_stops)):
                                raw_stops[i].time += wait
                        pending_waits[sortie.land_node] = max(
                            pending_waits.get(sortie.land_node, 0.0), drone_done
                        )

        for node in self.truck_path(truck_node, depot)[1:]:
            d = self.truck_dist(truck_node, node)
            truck_clock += d / fleet.truck_speed
            if node in pending_waits:
                truck_clock = max(truck_clock, pending_waits.pop(node))
            truck_node = node

        max_drone   = max((s.land_time for s in sorties), default=0.0)
        finish_time = max(truck_clock, max_drone)

        return TruckRoute(
            truck_id=tid,
            stops=raw_stops,
            served_edges=served,
            sorties=sorties,
            finish_time=finish_time,
        )

    # ------------------------------------------------------------------
    # Skeleton truck stops
    # ------------------------------------------------------------------

    def _build_raw_stops(self, events: list[tuple], depot: int) -> list[TruckStop]:
        """
        Xây dựng danh sách TruckStop.
        edge_count = số arc truck đã đi qua tính từ đầu.
        Tăng 1 sau MỖI ARC (kể cả intermediate nodes trên shortest path,
        kể cả unrequired edges) — đúng với định nghĩa δ trong paper.
        """
        fleet  = self.fleet
        stops  = [TruckStop(depot, 0.0, 0)]
        cur    = depot
        clock  = 0.0
        ec     = 0   # arc count

        for event in events:
            if event[0] != 'truck':
                continue
            _, signed_eid = event
            u, v, length  = self._edge_endpoints(signed_eid)

            # Intermediate nodes đến u — mỗi bước là 1 arc
            for node in self.truck_path(cur, u)[1:]:
                d = self.truck_dist(cur, node)
                clock += d / fleet.truck_speed
                ec    += 1
                stops.append(TruckStop(node, clock, ec))
                cur = node

            # Traverse required edge u → v — cũng là 1 arc
            clock += length / fleet.truck_speed
            ec    += 1
            cur    = v
            stops.append(TruckStop(v, clock, ec))

        # Về depot — mỗi bước cũng là 1 arc
        for node in self.truck_path(cur, depot)[1:]:
            d = self.truck_dist(cur, node)
            clock += d / fleet.truck_speed
            ec    += 1
            stops.append(TruckStop(node, clock, ec))
            cur = node

        return stops

    # ------------------------------------------------------------------
    # Build một sortie
    # ------------------------------------------------------------------

    def _build_sortie(
        self,
        drone_id:       int,
        edge_group:     list[int],
        t_prev_service: float,
        truck_clock:    float,
        truck_node:     int,
        truck_edge_cnt: int,
        raw_stops:      list[TruckStop],
        w_inf:          float = 1.0,
    ) -> Sortie:
        """
        Tìm land_stop minimize penalized rendezvous time:
            cost = rendezvous_time + w_inf * max(0, flight_time - τ)

        Xét tất cả candidates trong δ hop, kể cả vi phạm τ.
        Tiebreak: hops xa nhất.
        Fallback: land tại launch_node (0 hop).
        """
        fleet   = self.fleet
        first_u, _, _ = self._edge_endpoints(edge_group[0])
        _, last_v, _  = self._edge_endpoints(edge_group[-1])

        launch_node = truck_node
        launch_time = truck_clock
        launch_ec   = truck_edge_cnt

        fly_to_first   = self.drone_dist(launch_node, first_u) / fleet.drone_speed
        t_service_this = launch_time + fly_to_first
        flight_to_end  = self._calc_flight_time_to_end(edge_group, launch_node)
        seq_feasible   = t_service_this >= t_prev_service - 1e-9

        # Fallback: land tại launch_node (0 hop)
        fly_back_fallback = self.drone_dist(last_v, launch_node) / fleet.drone_speed
        fallback_flight   = flight_to_end + fly_back_fallback
        fallback_rv       = launch_time + fallback_flight
        fallback_cost     = fallback_rv + w_inf * max(0.0, fallback_flight - fleet.max_flight_time)

        best_land_node  = launch_node
        best_land_ec    = launch_ec
        best_flight     = fallback_flight
        best_rendezvous = fallback_rv
        best_cost       = fallback_cost
        best_hops       = 0

        for stop in raw_stops:
            hops = stop.edge_count - launch_ec
            if hops <= 0:
                continue
            if hops > fleet.delta:
                break

            fly_back        = self.drone_dist(last_v, stop.node) / fleet.drone_speed
            total_flight    = flight_to_end + fly_back
            drone_done      = launch_time + total_flight
            rendezvous_time = max(drone_done, stop.time)
            violation       = max(0.0, total_flight - fleet.max_flight_time)
            cost            = rendezvous_time + w_inf * violation

            # Chọn nếu cost tốt hơn, tiebreak: hops xa hơn
            if cost < best_cost - 1e-9 or (
                abs(cost - best_cost) < 1e-9 and hops > best_hops
            ):
                best_land_node  = stop.node
                best_land_ec    = stop.edge_count
                best_flight     = total_flight
                best_rendezvous = rendezvous_time
                best_cost       = cost
                best_hops       = hops

        sortie = Sortie(
            drone_id=drone_id,
            launch_node=launch_node,
            launch_time=launch_time,
            service_start_time=t_service_this,
            land_node=best_land_node,
            land_time=best_rendezvous,
            edge_ids=edge_group,
            flight_time=best_flight,
            feasible=seq_feasible and best_flight <= fleet.max_flight_time,
        )
        sortie._land_edge_count = best_land_ec
        return sortie

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_flight_time_to_end(self, edge_group: list[int], launch_node: int) -> float:
        speed = self.fleet.drone_speed
        time  = 0.0
        cur   = launch_node
        for signed_eid in edge_group:
            u, v, length = self._edge_endpoints(signed_eid)
            time += self.drone_dist(cur, u) / speed
            time += length / speed
            cur   = v
        return time

    @staticmethod
    def _group_events(tasks: list[tuple[int, int]], tid: int) -> list[tuple]:
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

    def _edge_endpoints(self, signed_eid: int) -> tuple[int, int, float]:
        u, v, length = self.edge_info(abs(signed_eid))
        return (u, v, length) if signed_eid > 0 else (v, u, length)