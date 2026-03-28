import random
from data.chromosome import Chromosome
from data.fleet_config import FleetConfig


class RandomInitializer:
    """
    Sinh một Chromosome hoàn toàn ngẫu nhiên.

    - service_sequence: permutation ngẫu nhiên của required edge ids,
      mỗi edge được gán chiều ngẫu nhiên (+/-).
    - vehicle_assignment: mỗi vị trí được gán ngẫu nhiên một vehicle id
      hợp lệ từ fleet.
    """

    def __init__(self, fleet: FleetConfig, required_edge_ids: list[int]):
        self.fleet = fleet
        self.required_edge_ids = required_edge_ids
        self.vehicle_ids = fleet.all_vehicle_ids()

    def create(self) -> Chromosome:
        # Shuffle thứ tự các required edges
        seq = self.required_edge_ids[:]
        random.shuffle(seq)

        # Gán chiều ngẫu nhiên: 50% giữ dương, 50% đổi âm
        seq = [eid if random.random() < 0.5 else -eid for eid in seq]

        # Gán vehicle ngẫu nhiên cho mỗi vị trí
        asgn = [random.choice(self.vehicle_ids) for _ in seq]

        return Chromosome(seq, asgn)
