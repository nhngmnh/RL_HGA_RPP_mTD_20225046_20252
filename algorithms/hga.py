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


class HGA:
    """
    Hybrid Genetic Algorithm cho RPP-mTD — Section 4 của paper.

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

        # --- Q-learning (RL-guided SPC) ---
        qcfg = QLearningConfig.from_dict(get_qlearning_config())
        self.q_agent = QLearningAgent(num_actions=fleet.num_trucks, config=qcfg, seed=params.seed)
        self._q_transitions: list[tuple[object, int, float, Individual]] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> Individual:
        t0 = time.time()

        self.q_agent.start_episode()

        # Step 1: khởi tạo population
        initial_pop = self.initializer.create_population()
        self.pop.initialize(initial_pop)
        self.best_individual = self.pop.best().clone()

        if verbose:
            print(f"Gen   0 | best={self.best_individual.makespan:.4f} | pop={self.pop.size()}")

        no_improve_count = 0

        # Step 2: vòng lặp GA
        for gen in range(1, self.params.G + 1):
            prev_best = self.best_individual.makespan

            # Sinh offspring đến PH
            offspring = self._generate_offspring()

            # Evaluate offspring trước (để local search chỉ dùng makespan đã có sẵn)
            self.evaluator.evaluate_many(offspring)

            # Q-learning update: reward = giảm makespan so với parent p1
            for state, action, p_primary_ms, child_ind in self._q_transitions:
                reward = p_primary_ms - child_ind.makespan
                self.q_agent.update(state, action, reward, done=True)
            self._q_transitions.clear()

            # Local search trên top ls_top_ratio offspring
            offspring = self._local_search(offspring)

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
            if isinstance(cx, SegmentPreservingCrossover):
                state1 = build_spc_state(
                    p1.chromosome, p2.chromosome, self.fleet, self.decoder, w_inf=self.evaluator.w_inf
                )
                action_k1 = self.q_agent.select_action(state1)
                c1 = cx.cross_one(p1.chromosome, p2.chromosome, action_k1)

                c2 = None
                if len(offspring) + 1 < target:
                    state2 = build_spc_state(
                        p2.chromosome, p1.chromosome, self.fleet, self.decoder, w_inf=self.evaluator.w_inf
                    )
                    action_k2 = self.q_agent.select_action(state2)
                    c2 = cx.cross_one(p2.chromosome, p1.chromosome, action_k2)
            else:
                c1, c2 = cx.cross(p1.chromosome, p2.chromosome)

            # Mutation
            if random.random() < self.current_pm:
                c1 = random.choice(self.mutations).mutate(c1)
            if c2 is not None and random.random() < self.current_pm:
                c2 = random.choice(self.mutations).mutate(c2)

            ind1 = Individual(c1)
            offspring.append(ind1)

            if isinstance(cx, SegmentPreservingCrossover):
                self._q_transitions.append((state1, action_k1, p1.makespan, ind1))

            if c2 is not None and len(offspring) < target:
                ind2 = Individual(c2)
                offspring.append(ind2)
                if isinstance(cx, SegmentPreservingCrossover):
                    self._q_transitions.append((state2, action_k2, p2.makespan, ind2))

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
