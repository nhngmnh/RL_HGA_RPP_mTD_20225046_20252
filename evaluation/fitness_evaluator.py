from __future__ import annotations
import math
from typing import Dict, Tuple
from data.individual import Individual
from data.hga_params import HGAParams
from .decoder import Decoder, DecodedSolution


class FitnessEvaluator:
    """
    Tính makespan và fitness cho một Individual.

    T(I)   = makespan + w_inf * tổng vi phạm τ
    F(I)   = T(I) * (nE / nP) ^ δ_diversity(I)
    """

    def __init__(
        self,
        decoder: Decoder,
        params: HGAParams,
        w_inf: float = 1.0,
        *,
        enable_decode_cache: bool = True,
        decode_cache_max_size: int = 2000,
    ):
        self.decoder = decoder
        self.params  = params
        self.w_inf   = w_inf

        # Per-generation decode cache.
        # Key: (service_sequence tuple, vehicle_assignment tuple)
        # Value: (sol.makespan, sol.total_violation, system_finish_times tuple)
        self.enable_decode_cache = enable_decode_cache
        self.decode_cache_max_size = int(decode_cache_max_size)
        self._decode_cache: Dict[
            Tuple[Tuple[int, ...], Tuple[int, ...]],
            Tuple[float, float, Tuple[float, ...]],
        ] = {}
        self._decode_cache_full = False

    def reset_cache(self) -> None:
        """Reset per-generation decode cache."""
        self._decode_cache.clear()
        self._decode_cache_full = False

    def evaluate(self, ind: Individual) -> Individual:
        chrom = ind.chromosome

        if self.enable_decode_cache:
            key = (tuple(chrom.service_sequence), tuple(chrom.vehicle_assignment))
            cached = self._decode_cache.get(key)
            if cached is not None:
                makespan_raw, total_violation, system_times = cached
                ind.makespan = makespan_raw + self.w_inf * total_violation
                ind.system_finish_times = list(system_times)
                return ind

        sol = self.decoder.decode(chrom, w_inf=self.w_inf)
        ind.makespan = sol.makespan + self.w_inf * sol.total_violation
        ind.system_finish_times = [r.finish_time for r in sol.truck_routes]

        if self.enable_decode_cache and not self._decode_cache_full:
            if len(self._decode_cache) < self.decode_cache_max_size:
                key = (tuple(chrom.service_sequence), tuple(chrom.vehicle_assignment))
                self._decode_cache[key] = (
                    sol.makespan,
                    sol.total_violation,
                    tuple(ind.system_finish_times),
                )
            else:
                # Hard cap: once exceeded in a generation, stop storing new entries.
                self._decode_cache_full = True
        return ind

    def evaluate_many(self, individuals: list[Individual]) -> list[Individual]:
        for ind in individuals:
            self.evaluate(ind)
        return individuals