from __future__ import annotations
from data.individual import Individual
from data.hga_params import HGAParams
from .fitness_evaluator import FitnessEvaluator
from .diversity_calculator import DiversityCalculator


class Population:
    """
    Quản lý tập hợp các Individual trong GA.

    Trách nhiệm:
      - Giữ list individuals
      - Sau mỗi generation: evaluate fitness, update diversity,
        trim về PL, bảo toàn top 1% (elitism)
    """

    def __init__(
        self,
        params:    HGAParams,
        evaluator: FitnessEvaluator,
        diversity: DiversityCalculator,
    ):
        self.params    = params
        self.evaluator = evaluator
        self.diversity = diversity
        self.individuals: list[Individual] = []

    # ------------------------------------------------------------------
    # Khởi tạo
    # ------------------------------------------------------------------

    def initialize(self, individuals: list[Individual]) -> None:
        self.individuals = individuals
        self._evaluate_all()

    # ------------------------------------------------------------------
    # Sau mỗi generation
    # ------------------------------------------------------------------

    def update(self, new_offspring: list[Individual]) -> None:
        """
        Thêm offspring vào population, evaluate, rồi trim.
        Bảo toàn top 1% elites qua trim.
        """
        self.evaluator.evaluate_many(new_offspring)
        self.individuals.extend(new_offspring)
        self._trim()

    def _evaluate_all(self) -> None:
        """Evaluate fitness + diversity cho toàn bộ population."""
        self.evaluator.evaluate_many(self.individuals)
        self._update_fitness_scores()

    def _trim(self) -> None:
        """
        1. Update diversity scores
        2. Update fitness F(I) = T(I) * (nE/nP)^δ
        3. Sort theo fitness tăng dần
        4. Giữ top 1% elites + fill đến PL
        """
        self._update_diversity()
        self._update_fitness_scores()
        self.individuals.sort()

        PL = self.params.PL
        n_elite = max(1, int(0.01 * PL))  # top 1%
        elites = self.individuals[:n_elite]

        # Giữ PL individuals tốt nhất
        self.individuals = self.individuals[:PL]

        # Đảm bảo elites không bị mất (thường đã nằm trong top PL)
        elite_set = {id(e) for e in elites}
        if not all(id(e) in {id(x) for x in self.individuals} for e in elites):
            # Thêm lại elites nếu bị đẩy ra
            self.individuals = elites + [
                x for x in self.individuals if id(x) not in elite_set
            ]
            self.individuals = self.individuals[:PL]

    def _update_diversity(self) -> None:
        self.diversity.update_diversity(self.individuals)

    def _update_fitness_scores(self) -> None:
        nE = self.params.n_elite
        nP = len(self.individuals)
        for ind in self.individuals:
            ind.update_fitness(nE, nP)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def best(self) -> Individual:
        return min(self.individuals, key=lambda x: x.makespan)

    def size(self) -> int:
        return len(self.individuals)

    def sorted_by_fitness(self) -> list[Individual]:
        return sorted(self.individuals)

