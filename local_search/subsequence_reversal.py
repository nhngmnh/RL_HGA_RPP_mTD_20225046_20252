import random
from data.chromosome import Chromosome
from data.individual import Individual
from .base import LocalSearchOperator


class SubsequenceReversal(LocalSearchOperator):
    """
    Chọn ngẫu nhiên một đoạn con [i,j] của service_sequence,
    đảo ngược nó (kèm flip dấu edge và đảo vehicle_assignment).
    Giống InversionMutation nhưng chỉ accept nếu cải thiện makespan.
    """

    def improve(self, ind: Individual, evaluator) -> Individual | None:
        chrom = ind.chromosome
        if chrom.length < 2:
            return None

        i, j = sorted(random.sample(range(chrom.length), 2))

        seq  = chrom.service_sequence[:]
        asgn = chrom.vehicle_assignment[:]

        seq[i:j+1]  = [-e for e in reversed(seq[i:j+1])]
        asgn[i:j+1] = list(reversed(asgn[i:j+1]))

        neighbor = Individual(Chromosome(seq, asgn))
        evaluator.evaluate(neighbor)

        if neighbor.makespan < ind.makespan:
            return neighbor
        return None

