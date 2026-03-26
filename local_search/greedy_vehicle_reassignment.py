import random
from data.chromosome import Chromosome
from data.individual import Individual
from data.fleet_config import FleetConfig
from .base import LocalSearchOperator


class GreedyVehicleReassignment(LocalSearchOperator):
    """
    Với mỗi arc, thử reassign sang tất cả vehicle khác,
    chọn reassignment cho makespan giảm nhiều nhất.

    Để tránh O(R * V) evaluations mỗi lần gọi,
    chỉ thử một arc ngẫu nhiên mỗi lần improve() được gọi.
    HGA sẽ gọi nhiều lần trong ls_steps.
    """

    def __init__(self, fleet: FleetConfig):
        self.vehicle_ids = fleet.all_vehicle_ids()

    def improve(self, ind: Individual, evaluator) -> Individual | None:
        chrom = ind.chromosome
        R = chrom.length

        # Chọn ngẫu nhiên một vị trí để thử reassign
        i = random.randrange(R)
        current_vid = chrom.vehicle_assignment[i]
        best_ind = ind

        for new_vid in self.vehicle_ids:
            if new_vid == current_vid:
                continue

            asgn = chrom.vehicle_assignment[:]
            asgn[i] = new_vid

            neighbor = Individual(Chromosome(chrom.service_sequence[:], asgn))
            evaluator.evaluate(neighbor)

            if neighbor.makespan < best_ind.makespan:
                best_ind = neighbor

        return best_ind if best_ind is not ind else None

