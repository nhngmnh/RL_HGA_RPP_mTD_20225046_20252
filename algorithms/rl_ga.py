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




class RLGA:
    """Genetic Algorithm + RL-guided SegmentPreserving crossover.

    This is the RL-enhanced variant (kept separate from the pure GA).
    """

    def __init__(
        self,
        fleet: FleetConfig,
        params: HGAParams,
        required_edge_ids: list[int],
        truck_dist_fn,
        drone_dist_fn,
        edge_info_fn,
        truck_path_fn,
    ):
        self.fleet = fleet
        self.params = params

        if params.seed is not None:
            random.seed(params.seed)

        # --- Evaluation ---
        self.decoder = Decoder(fleet, truck_dist_fn, drone_dist_fn, edge_info_fn, truck_path_fn)
        self.evaluator = FitnessEvaluator(self.decoder, params, w_inf=params.winf_min)
        self.diversity = DiversityCalculator()
        self.pop = Population(params, self.evaluator, self.diversity)

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

        # --- Local search operators (kept for parity with existing GA params) ---
        self.ls_ops = [
            SubsequenceReversal(),
            OrOpt(params),
            DroneSortieOptimizer(params),
            GreedyVehicleReassignment(fleet),
            RuinAndReconstruct(fleet, params),
        ]

        # --- State ---
        self.best_individual: Individual | None = None
        self.best_history: list[float] = []
        self.current_pm = params.pm

    def run(self, verbose: bool = True) -> Individual:
        t0 = time.time()

        initial_pop = self.initializer.create_population()
        self.pop.initialize(initial_pop)
        self.best_individual = self.pop.best().clone()

        if verbose:
            print(f"Gen   0 | best={self.best_individual.makespan:.4f} | pop={self.pop.size()}")

        no_improve_count = 0

        for gen in range(1, self.params.G + 1):
            # Offspring
            offspring = self._generate_offspring()

            # Evaluate offspring
            self.evaluator.evaluate_many(offspring)

            # Update population
            self.pop.update(offspring, already_evaluated=True)

            # Update best
            current_best = self.pop.best()
            if current_best.makespan < self.best_individual.makespan:
                self.best_individual = current_best.clone()
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.best_history.append(self.best_individual.makespan)

            # Mutation probability schedule
            if no_improve_count >= self.params.Gm:
                self.current_pm = self.params.pm_plus
            else:
                self.current_pm = self.params.pm

            # w_inf schedule
            progress = gen / self.params.G
            self.evaluator.w_inf = (
                self.params.winf_min
                + progress * (self.params.winf_max - self.params.winf_min)
            )

            if verbose and (gen % 10 == 0 or gen == self.params.G):
                elapsed = time.time() - t0
                print(
                    f"Gen {gen:3d} | best={self.best_individual.makespan:.4f} "
                    f"| pm={self.current_pm:.2f} | winf={self.evaluator.w_inf:.2f} "
                    f"| {elapsed:.1f}s"
                )

        return self.best_individual

    def _generate_offspring(self) -> list[Individual]:
        offspring: list[Individual] = []
        pop_sorted = self.pop.sorted_by_fitness()
        target = self.params.PH - self.pop.size()

        while len(offspring) < target:
            p1 = self._tournament(pop_sorted)
            p2 = self._tournament(pop_sorted)

            cx = random.choice(self.crossovers)
            # Crossover behavior is the same as the original GA:
            # choose an operator at random and call cross(); SPC will randomize system k internally.
            c1, c2 = cx.cross(p1.chromosome, p2.chromosome)

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

    @staticmethod
    def _tournament(pop_sorted: list[Individual]) -> Individual:
        a, b = random.sample(pop_sorted, 2)
        return a if a.fitness <= b.fitness else b
