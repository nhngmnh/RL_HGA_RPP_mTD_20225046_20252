from __future__ import annotations
import math
from data.individual import Individual
from data.hga_params import HGAParams
from .decoder import Decoder, DecodedSolution


class FitnessEvaluator:
    """
    Tính makespan và fitness cho một Individual.

    Công thức theo Section 4.4 của paper:
        T(I)   = makespan + penalty
        F(I)   = T(I) * (nE / nP) ^ δ_diversity(I)

    Penalty cho vi phạm τ:
        penalty = w_inf * sum_d( max(0, flight_time(d) - τ) )

    w_inf được điều chỉnh động từ bên ngoài (bởi HGA).
    """

    def __init__(
        self,
        decoder:  Decoder,
        params:   HGAParams,
        w_inf:    float = 1.0,   # penalty weight, được HGA điều chỉnh
    ):
        self.decoder = decoder
        self.params  = params
        self.w_inf   = w_inf

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(self, ind: Individual) -> Individual:
        """
        Decode chromosome, tính makespan + penalty, gán vào ind.makespan.
        Fitness (F(I)) được tính sau khi có diversity score —
        gọi update_fitness() sau khi DiversityCalculator chạy xong.
        """
        sol = self.decoder.decode(ind.chromosome)
        ind.makespan = self._penalized_makespan(sol)
        return ind

    def evaluate_many(self, individuals: list[Individual]) -> list[Individual]:
        for ind in individuals:
            self.evaluate(ind)
        return individuals

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _penalized_makespan(self, sol: DecodedSolution) -> float:
        """T(I) = makespan + w_inf * tổng vi phạm τ."""
        if sol.total_violation > 0:
            return sol.makespan + self.w_inf * sol.total_violation
        return sol.makespan

