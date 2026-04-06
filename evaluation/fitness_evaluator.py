from __future__ import annotations
import math
from data.individual import Individual
from data.hga_params import HGAParams
from .decoder import Decoder, DecodedSolution


class FitnessEvaluator:
    """
    Tính makespan và fitness cho một Individual.

    T(I)   = makespan + w_inf * tổng vi phạm τ
    F(I)   = T(I) * (nE / nP) ^ δ_diversity(I)
    """

    def __init__(self, decoder: Decoder, params: HGAParams, w_inf: float = 1.0):
        self.decoder = decoder
        self.params  = params
        self.w_inf   = w_inf

    def evaluate(self, ind: Individual) -> Individual:
        sol = self.decoder.decode(ind.chromosome, w_inf=self.w_inf)
        ind.makespan = sol.makespan + self.w_inf * sol.total_violation
        ind.system_finish_times = [r.finish_time for r in sol.truck_routes]
        return ind

    def evaluate_many(self, individuals: list[Individual]) -> list[Individual]:
        for ind in individuals:
            self.evaluate(ind)
        return individuals