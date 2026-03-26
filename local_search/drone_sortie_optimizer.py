import random
from itertools import permutations
from data.chromosome import Chromosome
from data.individual import Individual
from data.hga_params import HGAParams
from .base import LocalSearchOperator


class DroneSortieOptimizer(LocalSearchOperator):
    """
    Với mỗi drone sortie có >= sortie_min_len arcs,
    thử tất cả permutation của thứ tự các arc trong sortie
    và chọn thứ tự tốt nhất (giảm makespan).

    Đây thực chất là giải TSP nhỏ intra-sortie.
    Chỉ áp dụng khi sortie_len <= 5 để tránh bùng nổ tổ hợp (5! = 120).
    """

    def __init__(self, params: HGAParams):
        self.min_len = params.sortie_min_len
        self.max_len = 5  # 5! = 120, vẫn chấp nhận được

    def improve(self, ind: Individual, evaluator) -> Individual | None:
        chrom = ind.chromosome
        R = chrom.length

        # Tìm các drone sorties (chuỗi liên tiếp cùng vehicle, không phải truck)
        # Lấy danh sách các truck ids để phân biệt
        # Ta dùng heuristic: tìm run liên tiếp cùng vehicle_id
        best = ind

        i = 0
        while i < R:
            vid = chrom.vehicle_assignment[i]
            # Tìm end của run này
            j = i
            while j < R and chrom.vehicle_assignment[j] == vid:
                j += 1
            run_len = j - i

            if run_len >= self.min_len and run_len <= self.max_len:
                result = self._try_permutations(best, i, j, evaluator)
                if result is not None:
                    best = result

            i = j

        return best if best is not ind else None

    def _try_permutations(
        self, ind: Individual, i: int, j: int, evaluator
    ) -> Individual | None:
        chrom    = ind.chromosome
        seg_seq  = chrom.service_sequence[i:j]
        seg_asgn = chrom.vehicle_assignment[i:j]
        best_ind = ind

        for perm in permutations(range(len(seg_seq))):
            new_seg_seq = [seg_seq[p] for p in perm]
            if new_seg_seq == seg_seq:
                continue

            new_seq  = chrom.service_sequence[:i] + new_seg_seq + chrom.service_sequence[j:]
            new_asgn = chrom.vehicle_assignment[:i] + seg_asgn + chrom.vehicle_assignment[j:]

            neighbor = Individual(Chromosome(new_seq, new_asgn))
            evaluator.evaluate(neighbor)

            if neighbor.makespan < best_ind.makespan:
                best_ind = neighbor

        return best_ind if best_ind is not ind else None

