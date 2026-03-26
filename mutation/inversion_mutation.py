import random
from data.chromosome import Chromosome
from .base import MutationOperator


class InversionMutation(MutationOperator):
    """
    Inversion Mutation theo Section 4.6 của paper.

    Chọn đoạn con ngẫu nhiên [i, j] và đảo ngược cả hai phần:
        service_sequence[i:j+1]   -> reversed
        vehicle_assignment[i:j+1] -> reversed

    Lưu ý: đảo ngược service_sequence còn đảo chiều traverse
    của từng edge (dấu âm/dương), phản ánh đúng ý nghĩa
    "đảo chiều đi qua đoạn đường".
    """

    def mutate(self, chrom: Chromosome) -> Chromosome:
        if chrom.length < 2:
            return chrom.clone()

        i, j = sorted(random.sample(range(chrom.length), 2))

        seq  = chrom.service_sequence[:]
        asgn = chrom.vehicle_assignment[:]

        # Đảo đoạn [i:j+1] và flip dấu của từng edge trong đoạn
        seq[i:j+1]  = [-e for e in reversed(seq[i:j+1])]
        asgn[i:j+1] = list(reversed(asgn[i:j+1]))

        return Chromosome(seq, asgn)
