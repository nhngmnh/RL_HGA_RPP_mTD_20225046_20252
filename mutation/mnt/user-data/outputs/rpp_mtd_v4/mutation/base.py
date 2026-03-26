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
