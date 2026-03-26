import random
from data.chromosome import Chromosome
from data.fleet_config import FleetConfig
from .base import MutationOperator


class ReassignmentMutation(MutationOperator):
    """
    Reassignment Mutation theo Section 4.6 của paper.

    Chỉ tác động lên vehicle_assignment — service_sequence giữ nguyên.
    Chọn ngẫu nhiên một vị trí i và gán nó sang một vehicle khác
    (khác với vehicle hiện tại).

    Đây là operator duy nhất chỉ thay đổi phần assignment,
    giúp khám phá không gian phân công mà không làm xáo trộn thứ tự service.
    """

    def __init__(self, fleet: FleetConfig):
        self.vehicle_ids = fleet.all_vehicle_ids()

    def mutate(self, chrom: Chromosome) -> Chromosome:
        if len(self.vehicle_ids) < 2:
            return chrom.clone()

        i = random.randrange(chrom.length)
        current_vid = chrom.vehicle_assignment[i]

        # Chọn vehicle khác với vehicle hiện tại
        candidates = [v for v in self.vehicle_ids if v != current_vid]
        new_vid = random.choice(candidates)

        asgn = chrom.vehicle_assignment[:]
        asgn[i] = new_vid

        return Chromosome(chrom.service_sequence[:], asgn)
