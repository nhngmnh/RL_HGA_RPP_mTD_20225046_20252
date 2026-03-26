from __future__ import annotations
import math
from .chromosome import Chromosome


class Individual:
    """
    Một cá thể trong population.

    Attributes:
        chromosome:  giải pháp được encode
        makespan:    T(I) — thời gian hoàn thành (objective value)
        fitness:     F(I) = T(I) * (nE/nP)^δ_diversity
        diversity:   normalized Hamming distance tới 2 nearest neighbors
    """

    __slots__ = ("chromosome", "makespan", "fitness", "diversity")

    def __init__(self, chromosome: Chromosome):
        self.chromosome = chromosome
        self.makespan:  float = math.inf
        self.fitness:   float = math.inf
        self.diversity: float = 0.0

    # ------------------------------------------------------------------
    # Fitness update (gọi sau khi có makespan và diversity)
    # ------------------------------------------------------------------

    def update_fitness(self, n_elite: int, pop_size: int) -> None:
        """
        F(I) = T(I) * (nE / nP) ^ δ(I)
        Solution với makespan thấp VÀ diversity cao được ưu tiên hơn.
        """
        if self.makespan == math.inf:
            self.fitness = math.inf
            return
        ratio = n_elite / pop_size if pop_size > 0 else 1.0
        self.fitness = self.makespan * (ratio ** self.diversity)

    # ------------------------------------------------------------------
    # Comparison (dùng cho sort, heapq)
    # ------------------------------------------------------------------

    def __lt__(self, other: "Individual") -> bool:
        return self.fitness < other.fitness

    def __le__(self, other: "Individual") -> bool:
        return self.fitness <= other.fitness

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def clone(self) -> "Individual":
        ind = Individual(self.chromosome.clone())
        ind.makespan  = self.makespan
        ind.fitness   = self.fitness
        ind.diversity = self.diversity
        return ind

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"Individual(makespan={self.makespan:.4f}, "
                f"fitness={self.fitness:.4f}, "
                f"diversity={self.diversity:.4f})")
