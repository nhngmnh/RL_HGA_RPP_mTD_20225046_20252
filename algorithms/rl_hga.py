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

from configs.qlearning_params import get_qlearning_config
from qlearning import QLearningAgent, build_spc_state
from qlearning.q_agent import QLearningConfig


class RLHGA:
    """Hybrid Genetic Algorithm + RL-guided SegmentPreserving crossover.

    This is the RL-enhanced variant (kept separate from the pure HGA).
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
        self.best_history: list[float] = []
        self.current_pm = params.pm

        # --- Q-learning (RL-guided SPC) ---
        qcfg = QLearningConfig.from_dict(get_qlearning_config())
        self.q_agent = QLearningAgent(num_actions=fleet.num_trucks, config=qcfg, seed=params.seed)
        # Stored as: (state, action_k, p1_chrom, p2_chrom, child1, child2_or_None)
        self._q_transitions: list[
            tuple[object, int, Chromosome, Chromosome, Individual, Individual | None]
        ] = []

    def run(self, verbose: bool = True) -> Individual:
        t0 = time.time()

        self.q_agent.start_episode()

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

            # Q-learning update (per-system, 1 decision per pair)
            ft_cache: dict[int, list[float]] = {}

            def _finish_times(chrom: Chromosome) -> list[float]:
                key = id(chrom)
                if key not in ft_cache:
                    sol = self.decoder.decode(chrom, w_inf=self.evaluator.w_inf)
                    ft_cache[key] = [r.finish_time for r in sol.truck_routes]
                return ft_cache[key]

            for state, action_k, p1_chrom, p2_chrom, c1_ind, c2_ind in self._q_transitions:
                t_p1 = _finish_times(p1_chrom)[action_k - 1]
                t_p2 = _finish_times(p2_chrom)[action_k - 1]
                best_parent_t = min(t_p1, t_p2)

                t_c1 = _finish_times(c1_ind.chromosome)[action_k - 1]
                best_child_t = t_c1
                if c2_ind is not None:
                    t_c2 = _finish_times(c2_ind.chromosome)[action_k - 1]
                    best_child_t = min(best_child_t, t_c2)

                reward = max(0.0, best_parent_t - best_child_t)
                self.q_agent.update(state, action_k, reward, done=True)
            self._q_transitions.clear()

            # Local search
            offspring = self._local_search(offspring)

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

        if verbose:
            print(
                f"Q-learning | epsilon_final={self.q_agent.epsilon:.6f} "
                f"| total_steps={self.q_agent.total_steps}"
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
            if isinstance(cx, SegmentPreservingCrossover):
                state = build_spc_state(
                    p1.chromosome, p2.chromosome, self.fleet, self.decoder, w_inf=self.evaluator.w_inf
                )
                action_k = self.q_agent.select_action(state)
                c1, c2 = cx.cross_with_system(p1.chromosome, p2.chromosome, action_k)
            else:
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

            if isinstance(cx, SegmentPreservingCrossover):
                self._q_transitions.append((state, action_k, p1.chromosome, p2.chromosome, ind1, ind2))

        return offspring

    def _local_search(self, offspring: list[Individual]) -> list[Individual]:
        if any(ind.makespan == math.inf for ind in offspring):
            raise ValueError(
                "Offspring must be evaluated before local search. "
                "Call evaluator.evaluate_many(offspring) before _local_search()."
            )

        n_ls = max(1, int(self.params.ls_top_ratio * len(offspring)))
        offspring.sort(key=lambda x: x.makespan)
        to_improve = offspring[:n_ls]
        rest = offspring[n_ls:]

        improved: list[Individual] = []
        for ind in to_improve:
            current = ind
            for _ in range(self.params.ls_steps):
                op = random.choice(self.ls_ops)
                result = op.improve(current, self.evaluator)
                if result is not None:
                    current = result
            improved.append(current)

        return improved + rest

    @staticmethod
    def _tournament(pop_sorted: list[Individual]) -> Individual:
        a, b = random.sample(pop_sorted, 2)
        return a if a.fitness <= b.fitness else b
