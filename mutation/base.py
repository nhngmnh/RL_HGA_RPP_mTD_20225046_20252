from abc import ABC, abstractmethod
from data.chromosome import Chromosome


class MutationOperator(ABC):
    """
    Base class cho tất cả mutation operators.

    Convention:
        - Input : một Chromosome (không bị modify)
        - Output: Chromosome mới đã mutate
        - Mutation áp dụng lên CẢ HAI phần:
          service_sequence và vehicle_assignment
    """

    @abstractmethod
    def mutate(self, chrom: Chromosome) -> Chromosome:
        """Trả về chromosome mới sau mutation."""
        ...


class CrossoverOperator(ABC):
    """
    Base class cho tất cả crossover operators.

    Convention:
        - Input : 2 parent Chromosomes (không bị modify)
        - Output: 2 offspring Chromosomes mới
        - Cả 2 offspring phải là valid permutation của required_edge_ids
    """

    @abstractmethod
    def cross(self, p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """Trả về (offspring1, offspring2)."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers dùng trong OX, PMX, SegmentPreserving
    # ------------------------------------------------------------------

    @staticmethod
    def _signed_to_unsigned(seq: list[int]) -> list[int]:
        """[+3, -1, +2] -> [3, 1, 2] — bỏ dấu để so sánh vị trí."""
        return [abs(e) for e in seq]

    @staticmethod
    def _restore_sign(unsigned_seq: list[int], sign_source: list[int]) -> list[int]:
        """
        Khôi phục dấu từ sign_source cho unsigned_seq.
        sign_source là dict {eid: signed_eid} hoặc list gốc để tra.
        """
        sign_map = {abs(e): e for e in sign_source}
        return [sign_map[e] for e in unsigned_seq]

    @staticmethod
    def _repair(seq: list[int], all_eids: list[int]) -> list[int]:
        """
        Sau crossover có thể bị thiếu / trùng edge.
        Repair: giữ các edge xuất hiện đúng 1 lần, thay thế duplicate
        bằng các edge bị thiếu (theo thứ tự xuất hiện trong all_eids).
        Dấu của edge bị thiếu được giữ ngẫu nhiên.
        """
        import random
        seen = {}
        for i, e in enumerate(seq):
            eid = abs(e)
            seen.setdefault(eid, []).append(i)

        # Tìm duplicate và missing
        duplicates = [idxs[1:] for idxs in seen.values() if len(idxs) > 1]
        dup_indices = sorted(idx for idxs in duplicates for idx in idxs)
        missing_eids = [e for e in all_eids if e not in seen or len(seen[e]) == 0]

        result = seq[:]
        for idx, missing in zip(dup_indices, missing_eids):
            sign = 1 if random.random() < 0.5 else -1
            result[idx] = sign * missing

        return result
