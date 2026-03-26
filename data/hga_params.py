from dataclasses import dataclass, field


@dataclass
class HGAParams:
    """
    Hyperparameters của HGA theo Section 5 của paper.

    Defaults khớp chính xác với paper:
        nE = 0.8 * nP
        G = 100, Gm = 10
        PL = 100, PH = 200
        winf in [0.01, 100.0]
        pt = 0.1, pm = 0.1, pm_plus = 0.3
        pruin = 0.2, ls_steps = 30
    """
    # Population
    PL: int   = 100       # population size tối thiểu (sau trim)
    PH: int   = 200       # population size tối đa (trước trim)
    elite_ratio: float = 0.8   # nE = elite_ratio * PL

    # Generations
    G: int    = 100       # số generation tối đa
    Gm: int   = 10        # số gen không cải thiện → tăng mutation

    # Initializer
    pt: float = 0.1       # tỉ lệ targeted (heuristic) trong initial population

    # Mutation
    pm: float      = 0.1  # mutation probability ban đầu
    pm_plus: float = 0.3  # mutation probability khi stagnate

    # Penalty
    winf_min: float = 0.01
    winf_max: float = 100.0

    # Local search
    ls_top_ratio: float = 0.2   # top 20% được local search
    ls_steps: int       = 30    # số bước tối đa mỗi individual

    # Ruin-and-reconstruct
    pruin: float = 0.2    # tỉ lệ arc bị xóa

    # Or-opt
    or_opt_max_block: int = 3   # block tối đa b arcs

    # Drone sortie optimization threshold
    sortie_min_len: int = 3     # sortie phải >= 3 arc mới optimize

    # Random seed (None = không cố định)
    seed: int | None = None

    def __post_init__(self):
        assert self.PL <= self.PH
        assert 0 < self.elite_ratio < 1
        assert 0 < self.pt < 1
        assert 0 < self.pm <= 1
        assert 0 < self.pm_plus <= 1
        assert 0 < self.ls_top_ratio <= 1

    @property
    def n_elite(self) -> int:
        return int(self.elite_ratio * self.PL)

    @property
    def n_targeted_init(self) -> int:
        """Số individual được khởi tạo bằng heuristic."""
        return max(1, int(self.pt * self.PL))
