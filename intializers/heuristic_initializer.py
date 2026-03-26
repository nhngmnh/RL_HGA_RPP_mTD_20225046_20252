import random
from data.chromosome import Chromosome
from data.fleet_config import FleetConfig


class HeuristicInitializer:
    """
    Sinh Chromosome có hướng dẫn theo Section 4.5 của paper.

    Gồm 2 bước:
      1. Nearest-neighbor RPP sequence:
         Từ depot, lặp lại chọn required edge gần nhất (tính từ
         current node), quyết định chiều traverse (u→v hay v→u)
         sao cho điểm vào gần hơn.

      2. Greedy vehicle assignment:
         Duyệt sequence vừa tạo, với mỗi arc thử gán cho từng
         vehicle, chọn vehicle làm makespan tăng ít nhất.
         (Makespan ở đây là ước lượng đơn giản dựa trên
         completion time của từng vehicle — không cần full decode.)
    """

    def __init__(
        self,
        fleet: FleetConfig,
        required_edge_ids: list[int],
        truck_dist_fn,   # truck_dist(u, v) -> float
        drone_dist_fn,   # drone_dist(u, v) -> float
        edge_info_fn,    # edge_info(eid) -> (u, v, length)
    ):
        self.fleet = fleet
        self.required_edge_ids = required_edge_ids
        self.truck_dist = truck_dist_fn
        self.drone_dist = drone_dist_fn
        self.edge_info  = edge_info_fn  # trả về (u, v, length)
        self.vehicle_ids = fleet.all_vehicle_ids()
        self.truck_ids   = fleet.all_truck_ids()
        self.drone_ids   = fleet.all_drone_ids()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def create(self) -> Chromosome:
        seq  = self._build_sequence()
        asgn = self._greedy_assignment(seq)
        return Chromosome(seq, asgn)

    # ------------------------------------------------------------------
    # Step 1: Nearest-neighbor sequence
    # ------------------------------------------------------------------

    def _build_sequence(self) -> list[int]:
        """
        Bắt đầu từ depot, mỗi bước chọn required edge chưa serve
        có endpoint gần current_node nhất trên truck graph.
        Chiều traverse: chọn (u→v) hoặc (v→u) sao cho đầu vào gần hơn.
        """
        unvisited = set(self.required_edge_ids)
        current_node = self.fleet.depot_id
        seq = []

        while unvisited:
            best_eid      = None
            best_signed   = None
            best_dist     = float("inf")

            for eid in unvisited:
                u, v, _ = self.edge_info(eid)
                # Thử vào từ u (chiều u→v)
                d_uv = self.truck_dist(current_node, u)
                if d_uv < best_dist:
                    best_dist   = d_uv
                    best_eid    = eid
                    best_signed = eid          # dương = u→v
                # Thử vào từ v (chiều v→u)
                d_vu = self.truck_dist(current_node, v)
                if d_vu < best_dist:
                    best_dist   = d_vu
                    best_eid    = eid
                    best_signed = -eid         # âm = v→u

            seq.append(best_signed)
            unvisited.remove(best_eid)
            # Cập nhật current_node = điểm ra của edge vừa chọn
            u, v, _ = self.edge_info(best_eid)
            current_node = v if best_signed > 0 else u

        return seq

    # ------------------------------------------------------------------
    # Step 2: Greedy vehicle assignment
    # ------------------------------------------------------------------

    def _greedy_assignment(self, seq: list[int]) -> list[int]:
        """
        Duyệt seq từ trái sang phải. Với mỗi arc, thử gán cho từng
        vehicle và chọn vehicle làm completion_time tăng ít nhất.

        State theo dõi:
          - truck_node[tid]:  node hiện tại của truck tid
          - truck_time[tid]:  thời gian tích lũy của truck tid
          - drone_time[did]:  thời gian tích lũy của drone did
        """
        truck_ids = self.truck_ids
        drone_ids = self.drone_ids
        depot     = self.fleet.depot_id

        truck_node = {tid: depot for tid in truck_ids}
        truck_time = {tid: 0.0   for tid in truck_ids}
        drone_time = {did: 0.0   for did in drone_ids}

        asgn = []

        for signed_eid in seq:
            eid     = abs(signed_eid)
            u, v, length = self.edge_info(eid)
            start   = u if signed_eid > 0 else v
            end     = v if signed_eid > 0 else u

            best_vid  = None
            best_time = float("inf")

            # Thử gán cho từng truck
            for tid in truck_ids:
                travel = self.truck_dist(truck_node[tid], start)
                finish = truck_time[tid] + travel + length / self.fleet.truck_speed
                if finish < best_time:
                    best_time = finish
                    best_vid  = tid

            # Thử gán cho từng drone
            for did in drone_ids:
                parent_tid = self.fleet.parent_truck_id(did)
                # Drone xuất phát từ vị trí truck cha hiện tại
                launch_node = truck_node[parent_tid]
                fly_to      = self.drone_dist(launch_node, start)
                fly_edge    = length / self.fleet.drone_speed
                fly_back    = self.drone_dist(end, launch_node)
                flight_time = fly_to + fly_edge + fly_back
                finish      = max(drone_time[did], truck_time[parent_tid]) + flight_time
                if finish < best_time:
                    best_time = finish
                    best_vid  = did

            asgn.append(best_vid)

            # Cập nhật state
            if self.fleet.is_truck(best_vid):
                travel = self.truck_dist(truck_node[best_vid], start)
                truck_time[best_vid] += travel + length / self.fleet.truck_speed
                truck_node[best_vid]  = end
            else:
                parent_tid = self.fleet.parent_truck_id(best_vid)
                launch_node = truck_node[parent_tid]
                fly_time = (
                    self.drone_dist(launch_node, start)
                    + length / self.fleet.drone_speed
                    + self.drone_dist(end, launch_node)
                )
                drone_time[best_vid] = max(drone_time[best_vid], truck_time[parent_tid]) + fly_time

        return asgn
