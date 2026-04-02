import random
from data.chromosome import Chromosome
from mutation.base import CrossoverOperator


class PMXCrossover(CrossoverOperator):
    """
    Partially Mapped Crossover (PMX) theo Section 4.6 của paper.

    service_sequence (permutation-aware):
        1. Chọn 2 cut points [i, j]
        2. Copy đoạn p1[i:j+1] vào offspring1
        3. Với mỗi gene trong p2[i:j+1] chưa có trong offspring1:
           - Tìm vị trí của gene đó trong p2 → tra mapping → điền vào
             vị trí tương ứng, lặp lại nếu vị trí đó đã bị chiếm
        4. Fill phần còn lại từ p2 theo thứ tự

    vehicle_assignment: swap đoạn [i:j+1] giống OX.
    """

    def cross(self, p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
        R = p1.length
        if R < 2:
            return p1.clone(), p2.clone()
        i, j = sorted(random.sample(range(R), 2))

        seq1  = self._pmx_sequence(p1.service_sequence, p2.service_sequence, i, j)
        seq2  = self._pmx_sequence(p2.service_sequence, p1.service_sequence, i, j)
        asgn1 = p1.vehicle_assignment[:i] + p2.vehicle_assignment[i:j+1] + p1.vehicle_assignment[j+1:]
        asgn2 = p2.vehicle_assignment[:i] + p1.vehicle_assignment[i:j+1] + p2.vehicle_assignment[j+1:]

        return Chromosome(seq1, asgn1), Chromosome(seq2, asgn2)

    # ------------------------------------------------------------------

    @staticmethod
    def _pmx_sequence(primary: list[int], secondary: list[int],
                      i: int, j: int) -> list[int]:
        """
        PMX chuẩn dùng position lookup.
        Làm việc trên abs values, khôi phục dấu cuối cùng.

        Idea: pos_in_p[v] = vị trí của abs-value v trong primary.
        Khi secondary[k] bị conflict với segment:
          → tìm vị trí của secondary[k] trong primary
          → điền secondary[k] vào vị trí đó trong offspring
          → nếu vị trí đó vẫn conflict, tiếp tục chain
        """
        R = len(primary)
        p_abs = [abs(e) for e in primary]
        s_abs = [abs(e) for e in secondary]

        # Vị trí của mỗi abs-value trong primary
        pos_in_p = {v: idx for idx, v in enumerate(p_abs)}

        offspring_abs = [None] * R
        # Copy segment từ primary
        offspring_abs[i:j+1] = p_abs[i:j+1]
        in_seg = set(p_abs[i:j+1])

        # Điền ngoài segment
        for k in list(range(0, i)) + list(range(j + 1, R)):
            val = s_abs[k]
            # Follow chain nếu val đã có trong segment
            while val in in_seg:
                # val nằm ở vị trí pos_in_p[val] trong primary
                # secondary tại vị trí đó có giá trị s_abs[pos_in_p[val]]
                val = s_abs[pos_in_p[val]]
            offspring_abs[k] = val

        # Khôi phục dấu
        sign_map = {abs(e): (1 if e > 0 else -1) for e in secondary}
        for k in range(i, j + 1):
            sign_map[abs(primary[k])] = 1 if primary[k] > 0 else -1

        return [sign_map.get(v, 1) * v for v in offspring_abs]

    @staticmethod
    def _swap_segment(primary: list[int], secondary: list[int],
                      i: int, j: int) -> list[int]:
        result = primary[:]
        result[i:j+1] = secondary[i:j+1]
        return result
