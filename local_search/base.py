from abc import ABC, abstractmethod

from data.individual import Individual


class LocalSearchOperator(ABC):
    """Base class cho tất cả local search operators."""

    @abstractmethod
    def improve(self, ind: Individual, evaluator) -> Individual | None:
        """Trả về cá thể mới nếu cải thiện, ngược lại None."""
        ...
