from __future__ import annotations
from data.individual import Individual
from data.chromosome import Chromosome


class DiversityCalculator:
    """
    Tính diversity score cho từng Individual trong population.
    δ(I) = normalized Hamming distance trung bình tới 2 nearest neighbors
    Hamming distance giữa 2 chromosomes:
        - Đếm số vị trí mà service_sequence khác nhau (theo absolute value)
          HOẶC vehicle_assignment khác nhau
        - Normalize bằng cách chia cho R (độ dài chromosome)
    "Nearest neighbor" = individual có Hamming distance nhỏ nhất.
    """

    def update_diversity(self, population: list[Individual]) -> None:
        """
        Tính và gán diversity score cho tất cả individuals.
        Complexity: O(N^2 * R) — (chấp nhận được với N <= 500, R <= 100.)
        """
        n = len(population)
        if n < 3:
            for ind in population:
                ind.diversity = 0.0
            return

        # Precompute per-individual arrays once to reduce inner-loop overhead.
        R = population[0].chromosome.length
        abs_seqs = [list(map(abs, ind.chromosome.service_sequence)) for ind in population]
        asgns = [ind.chromosome.vehicle_assignment for ind in population]

        for i, ind in enumerate(population):
            # Track the two nearest neighbors without sorting the full distance list.
            best1 = float("inf")
            best2 = float("inf")

            a_seq = abs_seqs[i]
            a_asg = asgns[i]

            for j in range(n):
                if i == j:
                    continue
                d = self._hamming_arrays(a_seq, a_asg, abs_seqs[j], asgns[j], R)
                if d < best1:
                    best2 = best1
                    best1 = d
                elif d < best2:
                    best2 = d

            # Trung bình 2 nearest neighbors
            ind.diversity = (best1 + best2) / 2.0

    # ------------------------------------------------------------------

    @staticmethod
    def _hamming(c1: Chromosome, c2: Chromosome, R: int) -> float:
        """Normalized Hamming distance trong [0, 1]."""
        diff = 0
        for i in range(R):
            if abs(c1.service_sequence[i]) != abs(c2.service_sequence[i]):
                diff += 1
            elif c1.vehicle_assignment[i] != c2.vehicle_assignment[i]:
                diff += 1
        return diff / R

    @staticmethod
    def _hamming_arrays(a_seq: list[int], a_asg: list[int], b_seq: list[int], b_asg: list[int], R: int) -> float:
        """Normalized Hamming distance using precomputed arrays (same result as _hamming)."""
        diff = 0
        for i in range(R):
            if a_seq[i] != b_seq[i]:
                diff += 1
            elif a_asg[i] != b_asg[i]:
                diff += 1
        return diff / R

