import random
from data.chromosome import Chromosome
from data.individual import Individual
from data.fleet_config import FleetConfig
from data.hga_params import HGAParams
from .base import LocalSearchOperator


class RuinAndReconstruct(LocalSearchOperator):
    """
    Xóa pruin% arc ngẫu nhiên khỏi chromosome,
    sau đó greedy insert lại từng arc vào vị trí và vehicle
    tốt nhất (minimize makespan tăng thêm).
    """

    def __init__(self, fleet: FleetConfig, params: HGAParams):
        self.vehicle_ids = fleet.all_vehicle_ids()
        self.pruin       = params.pruin

    def improve(self, ind: Individual, evaluator) -> Individual | None:
        chrom = ind.chromosome
        R = chrom.length
        n_ruin = max(1, int(self.pruin * R))

        # Chọn ngẫu nhiên các vị trí bị xóa
        ruin_indices = set(random.sample(range(R), n_ruin))

        # Tách: giữ lại và bị xóa
        kept_seq  = [chrom.service_sequence[i]  for i in range(R) if i not in ruin_indices]
        kept_asgn = [chrom.vehicle_assignment[i] for i in range(R) if i not in ruin_indices]
        ruined    = [(chrom.service_sequence[i], chrom.vehicle_assignment[i])
                     for i in sorted(ruin_indices)]

        # Greedy insert từng arc bị xóa
        cur_seq  = kept_seq[:]
        cur_asgn = kept_asgn[:]

        for signed_eid, orig_vid in ruined:
            best_cost = float('inf')
            best_pos  = 0
            best_vid  = orig_vid

            # Thử từng vị trí chèn x từng vehicle
            for pos in range(len(cur_seq) + 1):
                for vid in self.vehicle_ids:
                    trial_seq  = cur_seq[:pos]  + [signed_eid]  + cur_seq[pos:]
                    trial_asgn = cur_asgn[:pos] + [vid]         + cur_asgn[pos:]
                    trial_ind  = Individual(Chromosome(trial_seq, trial_asgn))
                    evaluator.evaluate(trial_ind)
                    if trial_ind.makespan < best_cost:
                        best_cost = trial_ind.makespan
                        best_pos  = pos
                        best_vid  = vid

            cur_seq.insert(best_pos, signed_eid)
            cur_asgn.insert(best_pos, best_vid)

        neighbor = Individual(Chromosome(cur_seq, cur_asgn))
        evaluator.evaluate(neighbor)

        if neighbor.makespan < ind.makespan:
            return neighbor
        return None

