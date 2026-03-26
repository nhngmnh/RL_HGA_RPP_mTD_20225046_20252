from __future__ import annotations
import random
from copy import deepcopy


class Chromosome:
    """
    Two-layer chromosome encoding theo Section 4.2 của paper.

    service_sequence:   list[int], độ dài R
        - giá trị: edge id (1-based)
        - dương (+eid): traverse chiều gốc u→v
        - âm  (-eid): traverse chiều ngược v→u

    vehicle_assignment: list[int], độ dài R
        - giá trị: vehicle id (truck hoặc drone)
        - theo FleetConfig.vehicle_ids convention

    Hai mảng luôn có cùng độ dài và được giữ đồng bộ.
    """

    __slots__ = ("service_sequence", "vehicle_assignment")

    def __init__(self, service_sequence: list[int], vehicle_assignment: list[int]):
        assert len(service_sequence) == len(vehicle_assignment), \
            "service_sequence và vehicle_assignment phải cùng độ dài"
        self.service_sequence  = service_sequence
        self.vehicle_assignment = vehicle_assignment

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        return len(self.service_sequence)

    def edge_id_at(self, pos: int) -> int:
        """Edge id tuyệt đối tại vị trí pos (bỏ dấu âm)."""
        return abs(self.service_sequence[pos])

    def direction_at(self, pos: int) -> int:
        """Chiều traverse tại pos: +1 = gốc, -1 = ngược."""
        return 1 if self.service_sequence[pos] > 0 else -1

    def vehicle_at(self, pos: int) -> int:
        return self.vehicle_assignment[pos]

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def clone(self) -> "Chromosome":
        return Chromosome(
            self.service_sequence[:],
            self.vehicle_assignment[:]
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_valid(self, required_edge_ids: list[int]) -> bool:
        """
        Kiểm tra chromosome hợp lệ:
        - Mỗi required edge xuất hiện đúng 1 lần (bất kể chiều).
        - Vehicle assignment chỉ chứa valid vehicle ids.
        """
        seen = sorted(abs(e) for e in self.service_sequence)
        expected = sorted(required_edge_ids)
        return seen == expected

    # ------------------------------------------------------------------
    # Segment helpers (dùng trong SegmentPreservingCrossover)
    # ------------------------------------------------------------------

    def segment_of_vehicle(self, vid: int) -> list[int]:
        """Trả về danh sách vị trí (indices) được assign cho vehicle vid."""
        return [i for i, v in enumerate(self.vehicle_assignment) if v == vid]

    def segment_of_system(self, system_vids: list[int]) -> list[int]:
        """Indices thuộc truck system (truck + drones của nó), theo thứ tự."""
        vset = set(system_vids)
        return [i for i, v in enumerate(self.vehicle_assignment) if v in vset]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        seq = " ".join(f"{e:+d}" for e in self.service_sequence)
        asgn = " ".join(str(v) for v in self.vehicle_assignment)
        return f"Chromosome(\n  seq : [{seq}]\n  asgn: [{asgn}]\n)"
