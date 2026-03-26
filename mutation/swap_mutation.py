import random
from data.chromosome import Chromosome
from .base import MutationOperator


class SwapMutation(MutationOperator):
    """
    Swap Mutation theo Section 4.6 của paper.

    Chọn 2 vị trí ngẫu nhiên i, j và hoán đổi gene ở cả hai phần:
        service_sequence[i]  <-> service_sequence[j]
        vehicle_assignment[i] <-> vehicle_assignment[j]
    """

    def mutate(self, chrom: Chromosome) -> Chromosome:
        if chrom.length < 2:
            return chrom.clone()

        i, j = random.sample(range(chrom.length), 2)

        seq  = chrom.service_sequence[:]
        asgn = chrom.vehicle_assignment[:]

        seq[i],  seq[j]  = seq[j],  seq[i]
        asgn[i], asgn[j] = asgn[j], asgn[i]

        return Chromosome(seq, asgn)
