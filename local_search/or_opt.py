import random
from data.chromosome import Chromosome
from data.individual import Individual
from data.hga_params import HGAParams
from .base import LocalSearchOperator


class OrOpt(LocalSearchOperator):
    """
    Dời một block liên tiếp tối đa b arcs sang vị trí khác trong sequence.
    Block size được chọn ngẫu nhiên trong [1, or_opt_max_block].
    Vị trí đích được chọn ngẫu nhiên ngoài block hiện tại.
    """

    def __init__(self, params: HGAParams):
        self.max_block = params.or_opt_max_block

    def improve(self, ind: Individual, evaluator) -> Individual | None:
        chrom = ind.chromosome
        R = chrom.length
        if R < 3:
            return None

        block_size = random.randint(1, min(self.max_block, R - 1))
        i = random.randint(0, R - block_size)
        j = i + block_size  # block = seq[i:j]

        # Chọn vị trí chèn ngoài block (insert_pos là vị trí trong seq sau khi xóa block)
        remaining_len = R - block_size
        if remaining_len < 1:
            return None
        insert_pos = random.randint(0, remaining_len)

        seq  = chrom.service_sequence[:]
        asgn = chrom.vehicle_assignment[:]

        # Xóa block
        block_seq  = seq[i:j]
        block_asgn = asgn[i:j]
        seq_wo  = seq[:i]  + seq[j:]
        asgn_wo = asgn[:i] + asgn[j:]

        # Chèn lại tại vị trí mới
        new_seq  = seq_wo[:insert_pos]  + block_seq  + seq_wo[insert_pos:]
        new_asgn = asgn_wo[:insert_pos] + block_asgn + asgn_wo[insert_pos:]

        if new_seq == chrom.service_sequence:
            return None

        neighbor = Individual(Chromosome(new_seq, new_asgn))
        evaluator.evaluate(neighbor)

        if neighbor.makespan < ind.makespan:
            return neighbor
        return None

