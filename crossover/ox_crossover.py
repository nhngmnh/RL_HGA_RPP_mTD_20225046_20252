import random
from data.chromosome import Chromosome
from mutation.base import CrossoverOperator


class OXCrossover(CrossoverOperator):
    """
    Order Crossover (OX) theo Section 4.6 của paper.

    Áp dụng độc lập lên service_sequence và vehicle_assignment:

    service_sequence (permutation-aware):
        1. Chọn 2 cut points ngẫu nhiên [i, j]
        2. Copy đoạn p1[i:j+1] vào offspring1
        3. Fill phần còn lại bằng các gene từ p2 theo thứ tự xuất hiện,
           bỏ qua những gene đã có (so sánh theo abs value)

    vehicle_assignment:
        OX đơn giản trên số nguyên thông thường (không cần abs):
        cùng logic nhưng không có ràng buộc permutation —
        ta chỉ swap đoạn giữa 2 cut points.
    """

    def cross(self, p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
        R = p1.length
        if R < 2:
            return p1.clone(), p2.clone()
        i, j = sorted(random.sample(range(R), 2))

        seq1  = self._ox_sequence(p1.service_sequence, p2.service_sequence, i, j)
        seq2  = self._ox_sequence(p2.service_sequence, p1.service_sequence, i, j)
        # vehicle_assignment: swap đoạn, đảm bảo cùng độ dài R
        asgn1 = p1.vehicle_assignment[:i] + p2.vehicle_assignment[i:j+1] + p1.vehicle_assignment[j+1:]
        asgn2 = p2.vehicle_assignment[:i] + p1.vehicle_assignment[i:j+1] + p2.vehicle_assignment[j+1:]

        return Chromosome(seq1, asgn1), Chromosome(seq2, asgn2)

    # ------------------------------------------------------------------

    @staticmethod
    def _ox_sequence(primary: list[int], secondary: list[int],
                     i: int, j: int) -> list[int]:
        """OX trên service_sequence — so sánh theo abs để tránh nhầm dấu."""
        R = len(primary)
        offspring = [None] * R

        # Copy đoạn giữa từ primary
        offspring[i:j+1] = primary[i:j+1]
        copied_eids = {abs(e) for e in primary[i:j+1]}

        # Fill từ secondary, theo thứ tự xoay vòng từ j+1
        fill_pos = [(j + 1 + k) % R for k in range(R - (j - i + 1))]
        fill_vals = [e for e in (secondary[j+1:] + secondary[:j+1])
                     if abs(e) not in copied_eids]

        for pos, val in zip(fill_pos, fill_vals):
            offspring[pos] = val

        return offspring

    @staticmethod
    def _ox_assignment(primary: list[int], secondary: list[int],
                       i: int, j: int) -> list[int]:
        """
        Với vehicle_assignment không có ràng buộc permutation:
        chỉ swap đoạn [i:j+1] giữa primary và secondary.
        """
        result = primary[:]
        result[i:j+1] = secondary[i:j+1]
        return result
