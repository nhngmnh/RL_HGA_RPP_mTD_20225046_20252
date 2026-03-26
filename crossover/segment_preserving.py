import random
from data.chromosome import Chromosome
from data.fleet_config import FleetConfig
from ..mutation.base import CrossoverOperator
from .ox_crossover import OXCrossover
from .pmx_crossover import PMXCrossover


class SegmentPreservingCrossover(CrossoverOperator):
    """
    Segment-Preserving Crossover (SPC) — novel operator trong Section 4.6.

    Ý tưởng: bảo toàn nguyên vẹn segment của một truck system
    (truck + drones của nó) từ một parent, trong khi phần còn lại
    được recombine từ parent kia.

    Các bước:
        1. Chọn ngẫu nhiên một truck system k
        2. Trích segment_k từ p1 (tất cả indices thuộc system k)
        3. Trích segment_k tương ứng từ p2
        4. Áp OX hoặc PMX (ngẫu nhiên) lên 2 segment đó
           → tạo ra 2 sub-segment mới
        5. Reintegrate: thay segment_k trong bản sao p1/p2
           bằng sub-segment mới
        6. Repair: sửa duplicate/missing edges
    """

    def __init__(self, fleet: FleetConfig):
        self.fleet   = fleet
        self._ox     = OXCrossover()
        self._pmx    = PMXCrossover()

    def cross(self, p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
        # Chọn ngẫu nhiên một truck system
        k = random.randint(1, self.fleet.num_trucks)
        system_vids = self.fleet.system_ids(k)

        # Lấy indices thuộc system k trong mỗi parent
        indices_p1 = p1.segment_of_system(system_vids)
        indices_p2 = p2.segment_of_system(system_vids)

        if not indices_p1 or not indices_p2:
            # Fallback: không có gì để trao đổi → trả về clone
            return p1.clone(), p2.clone()

        # Trích sub-chromosome của segment
        sub_p1 = self._extract_segment(p1, indices_p1)
        sub_p2 = self._extract_segment(p2, indices_p2)

        # Áp OX hoặc PMX lên segment (chọn ngẫu nhiên)
        sub_op = self._ox if random.random() < 0.5 else self._pmx

        # Chỉ cross nếu segment đủ dài VÀ cùng độ dài
        if sub_p1.length >= 2 and sub_p2.length >= 2 and sub_p1.length == sub_p2.length:
            sub_o1, sub_o2 = sub_op.cross(sub_p1, sub_p2)
            # Reintegrate vào bản sao của parent
            o1 = self._reintegrate(p1, indices_p1, sub_o1)
            o2 = self._reintegrate(p2, indices_p2, sub_o2)
        else:
            # Độ dài khác nhau: swap toàn bộ segment giữa hai parent
            o1 = self._reintegrate(p1, indices_p1, sub_p2 if sub_p2.length == len(indices_p1) else sub_p1)
            o2 = self._reintegrate(p2, indices_p2, sub_p1 if sub_p1.length == len(indices_p2) else sub_p2)
            # Nếu vẫn không khớp, trả về clone
            if sub_p2.length != len(indices_p1) or sub_p1.length != len(indices_p2):
                return p1.clone(), p2.clone()

        # Repair duplicate / missing
        all_eids = [abs(e) for e in p1.service_sequence]
        o1 = self._repair_chromosome(o1, all_eids)
        o2 = self._repair_chromosome(o2, all_eids)

        return o1, o2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_segment(chrom: Chromosome, indices: list[int]) -> Chromosome:
        """Tạo sub-chromosome từ các indices chỉ định."""
        seq  = [chrom.service_sequence[i]  for i in indices]
        asgn = [chrom.vehicle_assignment[i] for i in indices]
        return Chromosome(seq, asgn)

    @staticmethod
    def _reintegrate(original: Chromosome,
                     indices: list[int],
                     segment: Chromosome) -> Chromosome:
        """Thay các vị trí indices trong original bằng segment mới."""
        new_seq  = original.service_sequence[:]
        new_asgn = original.vehicle_assignment[:]
        for pos, idx in enumerate(indices):
            new_seq[idx]  = segment.service_sequence[pos]
            new_asgn[idx] = segment.vehicle_assignment[pos]
        return Chromosome(new_seq, new_asgn)

    @staticmethod
    def _repair_chromosome(chrom: Chromosome, all_eids: list[int]) -> Chromosome:
        """Sửa duplicate/missing trong service_sequence, giữ vehicle_assignment."""
        seq = chrom.service_sequence[:]
        seen: dict[int, list[int]] = {}
        for i, e in enumerate(seq):
            seen.setdefault(abs(e), []).append(i)

        dup_indices = sorted(
            idx for idxs in seen.values() if len(idxs) > 1 for idx in idxs[1:]
        )
        missing = [e for e in all_eids if e not in seen or len(seen[e]) == 0]

        for idx, eid in zip(dup_indices, missing):
            sign = 1 if random.random() < 0.5 else -1
            seq[idx] = sign * eid

        return Chromosome(seq, chrom.vehicle_assignment[:])
