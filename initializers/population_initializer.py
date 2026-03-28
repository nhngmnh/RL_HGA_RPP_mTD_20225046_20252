import random
from data.chromosome import Chromosome
from data.individual import Individual
from data.fleet_config import FleetConfig
from data.hga_params import HGAParams
from .random_initializer import RandomInitializer
from .heuristic_initializer import HeuristicInitializer


class PopulationInitializer:
    """
    Tạo initial population theo Section 4.5 của paper.

    - pt * PL cá thể được khởi tạo bằng HeuristicInitializer
    - (1 - pt) * PL cá thể còn lại bằng RandomInitializer
    """

    def __init__(
        self,
        fleet: FleetConfig,
        params: HGAParams,
        required_edge_ids: list[int],
        truck_dist_fn,
        drone_dist_fn,
        edge_info_fn,
    ):
        self.params = params

        self.random_init = RandomInitializer(
            fleet=fleet,
            required_edge_ids=required_edge_ids,
        )
        self.heuristic_init = HeuristicInitializer(
            fleet=fleet,
            required_edge_ids=required_edge_ids,
            truck_dist_fn=truck_dist_fn,
            drone_dist_fn=drone_dist_fn,
            edge_info_fn=edge_info_fn,
        )

    def create_population(self) -> list[Individual]:
        """
        Trả về list Individual với PL cá thể.
        Chromosome đã được tạo, fitness chưa được tính
        (sẽ do FitnessEvaluator xử lý sau).
        """
        population = []
        n_heuristic = self.params.n_targeted_init
        n_random    = self.params.PL - n_heuristic

        # Heuristic individuals
        for _ in range(n_heuristic):
            chrom = self.heuristic_init.create()
            population.append(Individual(chrom))

        # Random individuals
        for _ in range(n_random):
            chrom = self.random_init.create()
            population.append(Individual(chrom))

        random.shuffle(population)
        return population
