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

        # Precompute pairwise distances
        R = population[0].chromosome.length

        for i, ind in enumerate(population):
            distances = []
            for j, other in enumerate(population):
                if i == j:
                    continue
                d = self._hamming(ind.chromosome, other.chromosome, R)
                distances.append(d)

            distances.sort()
            # Trung bình 2 nearest neighbors
            ind.diversity = (distances[0] + distances[1]) / 2.0 if len(distances) >= 2 else distances[0]

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

