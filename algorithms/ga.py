from __future__ import annotations
import random
import math
import time

from data import Chromosome, Individual, FleetConfig, HGAParams
from initializers import PopulationInitializer
from crossover import OXCrossover, PMXCrossover, SegmentPreservingCrossover
from mutation import SwapMutation, InversionMutation, ReassignmentMutation
from local_search import (
    SubsequenceReversal, OrOpt, DroneSortieOptimizer,
    GreedyVehicleReassignment, RuinAndReconstruct,
)
from evaluation import Decoder, FitnessEvaluator, DiversityCalculator, Population


class GA:
    """
    Genetic Algorithm cho RPP-mTD.

    Vòng lặp chính:
        1. Khởi tạo population (pt% heuristic + rest random)
        2. Lặp G generations:
            a. Binary tournament selection
            b. Crossover (OX / PMX / SegmentPreserving, chọn ngẫu nhiên)
            c. Mutation (Swap / Inversion / Reassignment, chọn ngẫu nhiên)
            d. Local search trên top 20% offspring
            e. Trim population về PL, giữ top 1% elites
            f. Nếu Gm gen không cải thiện → tăng pm lên pm_plus
        3. Trả về best solution tìm được
    """

    def __init__(
        self,
        fleet:         FleetConfig,
        params:        HGAParams,
        required_edge_ids: list[int],
        truck_dist_fn,
        drone_dist_fn,
        edge_info_fn,
        truck_path_fn,
    ):
        self.fleet  = fleet
        self.params = params

        if params.seed is not None:
            random.seed(params.seed)

        # --- Evaluation ---
        self.decoder   = Decoder(fleet, truck_dist_fn, drone_dist_fn, edge_info_fn, truck_path_fn)
        self.evaluator = FitnessEvaluator(self.decoder, params, w_inf=params.winf_min)
        self.diversity = DiversityCalculator()
        self.pop       = Population(params, self.evaluator, self.diversity)

        # --- Initializer ---
        self.initializer = PopulationInitializer(
            fleet, params, required_edge_ids,
            truck_dist_fn, drone_dist_fn, edge_info_fn,
        )

        # --- Crossover operators ---
        self.crossovers = [
            OXCrossover(),
            PMXCrossover(),
            SegmentPreservingCrossover(fleet),
        ]

        # --- Mutation operators ---
        self.mutations = [
            SwapMutation(),
            InversionMutation(),
            ReassignmentMutation(fleet),
        ]

        # --- Local search operators ---
        self.ls_ops = [
            SubsequenceReversal(),
            OrOpt(params),
            DroneSortieOptimizer(params),
            GreedyVehicleReassignment(fleet),
            RuinAndReconstruct(fleet, params),
        ]

        # --- State ---
        self.best_individual: Individual | None = None
        self.best_history:    list[float] = []
        self.current_pm = params.pm

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> Individual:
        t0 = time.time()

        # Step 1: khởi tạo population
        initial_pop = self.initializer.create_population()
        self.pop.initialize(initial_pop)
        self.best_individual = self.pop.best().clone()

        if verbose:
            print(f"Gen   0 | best={self.best_individual.makespan:.4f} | pop={self.pop.size()}")

        no_improve_count = 0

        # Step 2: vòng lặp GA
        for gen in range(1, self.params.G + 1):
            # Per-generation decode cache
            self.evaluator.reset_cache()
            prev_best = self.best_individual.makespan

            # Sinh offspring đến PH
            offspring = self._generate_offspring()

            # Evaluate offspring trước (để local search chỉ dùng makespan đã có sẵn)
            self.evaluator.evaluate_many(offspring)

            # Cập nhật population
            self.pop.update(offspring, already_evaluated=True)

            # Cập nhật best
            current_best = self.pop.best()
            if current_best.makespan < self.best_individual.makespan:
                self.best_individual = current_best.clone()
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.best_history.append(self.best_individual.makespan)

            # Điều chỉnh mutation probability
            if no_improve_count >= self.params.Gm:
                self.current_pm = self.params.pm_plus
            else:
                self.current_pm = self.params.pm

            # Điều chỉnh w_inf (tăng dần để penalty mạnh hơn theo thời gian)
            progress = gen / self.params.G
            self.evaluator.w_inf = (
                self.params.winf_min
                + progress * (self.params.winf_max - self.params.winf_min)
            )

            if verbose and (gen % 10 == 0 or gen == self.params.G):
                elapsed = time.time() - t0
                print(f"Gen {gen:3d} | best={self.best_individual.makespan:.4f} "
                      f"| pm={self.current_pm:.2f} | winf={self.evaluator.w_inf:.2f} "
                      f"| {elapsed:.1f}s")

        return self.best_individual

    # ------------------------------------------------------------------
    # Generate offspring
    # ------------------------------------------------------------------

    def _generate_offspring(self) -> list[Individual]:
        offspring = []
        pop_sorted = self.pop.sorted_by_fitness()
        target = self.params.PH - self.pop.size()

        while len(offspring) < target:
            # Binary tournament selection
            p1 = self._tournament(pop_sorted)
            p2 = self._tournament(pop_sorted)

            # Crossover
            cx = random.choice(self.crossovers)
            c1, c2 = cx.cross(p1.chromosome, p2.chromosome)

            # Mutation
            if random.random() < self.current_pm:
                c1 = random.choice(self.mutations).mutate(c1)
            if c2 is not None and random.random() < self.current_pm:
                c2 = random.choice(self.mutations).mutate(c2)

            ind1 = Individual(c1)
            offspring.append(ind1)

            ind2: Individual | None = None
            if c2 is not None and len(offspring) < target:
                ind2 = Individual(c2)
                offspring.append(ind2)

        return offspring

    # ------------------------------------------------------------------
    # Local search
    # ------------------------------------------------------------------

    def _local_search(self, offspring: list[Individual]) -> list[Individual]:
        # Offspring phải được evaluate trước khi vào local search
        if any(ind.makespan == math.inf for ind in offspring):
            raise ValueError(
                "Offspring must be evaluated before local search. "
                "Call evaluator.evaluate_many(offspring) before _local_search()."
            )

        n_ls = max(1, int(self.params.ls_top_ratio * len(offspring)))
        offspring.sort(key=lambda x: x.makespan)
        to_improve = offspring[:n_ls]
        rest       = offspring[n_ls:]

        improved = []
        for ind in to_improve:
            current = ind
            for _ in range(self.params.ls_steps):
                op = random.choice(self.ls_ops)
                result = op.improve(current, self.evaluator)
                if result is not None:
                    current = result
            improved.append(current)

        return improved + rest

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    @staticmethod
    def _tournament(pop_sorted: list[Individual]) -> Individual:
        """Binary tournament: chọn 2 ngẫu nhiên, trả về individual tốt hơn."""
        a, b = random.sample(pop_sorted, 2)
        return a if a.fitness <= b.fitness else b
