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
from qlearning import QLearningAgent, build_ls_state
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

        qcfg = QLearningConfig.from_dict(get_qlearning_config())
        # --- Q-learning (RL-guided local search) ---
        # Actions (5): SubsequenceReversal, OrOpt, DroneSortieOptimizer, GreedyVehicleReassignment, RuinAndReconstruct
        self.ls_agent = QLearningAgent(num_actions=5, config=qcfg, seed=params.seed)

    def run(self, verbose: bool = True) -> Individual:
        t0 = time.time()
        self.ls_agent.start_episode()

        initial_pop = self.initializer.create_population()
        self.pop.initialize(initial_pop)
        self.best_individual = self.pop.best().clone()

        if verbose:
            print(f"Gen   0 | best={self.best_individual.makespan:.4f} | pop={self.pop.size()}")

        no_improve_count = 0

        for gen in range(1, self.params.G + 1):
            # Per-generation decode cache
            self.evaluator.reset_cache()
            # Offspring
            offspring = self._generate_offspring()

            # Evaluate offspring
            self.evaluator.evaluate_many(offspring)

            # Local search
            offspring, ls_map = self._local_search_with_map(offspring, gen)

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
                f"Q-learning (local-search) | epsilon_final={self.ls_agent.epsilon:.6f} "
                f"| total_steps={self.ls_agent.total_steps}"
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
            # Crossover behavior is the same as the original HGA:
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

    def _local_search_with_map(
        self, offspring: list[Individual]
        , gen: int
    ) -> tuple[list[Individual], dict[int, Individual]]:
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
        ls_map: dict[int, Individual] = {}
        for ind in to_improve:
            current = ind
            for _ in range(self.params.ls_steps):
                # RL chooses one of 5 LS actions
                state = build_ls_state(
                    current.chromosome,
                    self.fleet,
                    self.decoder,
                    gen=gen,
                    total_gens=self.params.G,
                    sortie_min_len=self.params.sortie_min_len,
                    system_finish_times=current.system_finish_times,
                    w_inf=self.evaluator.w_inf,
                )

                # If no drone is actionable, skip DroneSortieOptimizer action.
                # Actions are 1-based: 1=SubsequenceReversal, 2=OrOpt, 3=DroneSortieOptimizer, 4=GreedyVehicleReassignment, 5=RuinAndReconstruct
                drone_actionable_count = state[1]
                valid_actions = [1, 2, 4, 5] if drone_actionable_count == 0 else [1, 2, 3, 4, 5]
                a = self.ls_agent.select_action(state, valid_actions=valid_actions)
                op = self.ls_ops[a - 1]

                before = current.makespan
                result = op.improve(current, self.evaluator)
                after_ind = result if result is not None else current

                reward = max(0.0, before - after_ind.makespan)
                self.ls_agent.update(state, a, reward, done=True)

                current = after_ind
            improved.append(current)
            ls_map[id(ind)] = current

        for ind in rest:
            ls_map[id(ind)] = ind

        return improved + rest, ls_map

    @staticmethod
    def _tournament(pop_sorted: list[Individual]) -> Individual:
        a, b = random.sample(pop_sorted, 2)
        return a if a.fitness <= b.fitness else b
