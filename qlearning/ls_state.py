from __future__ import annotations

from typing import Hashable

from data.chromosome import Chromosome
from data.fleet_config import FleetConfig
from evaluation.decoder import Decoder


def build_ls_state(
    chrom: Chromosome,
    fleet: FleetConfig,
    decoder: Decoder,
    *,
    gen: int,
    total_gens: int,
    sortie_min_len: int,
    sortie_max_len: int = 5,
    imbalance_low: float = 0.10,
    imbalance_high: float = 0.25,
    system_finish_times: list[float] | None = None,
    w_inf: float = 1.0,
) -> Hashable:
    """Local-search RL state.

    Features (3):
      1) phase_generation: 0=begin, 1=middle, 2=last
      2) drone_actionable_count: number of drones having at least one consecutive run
         with length in [sortie_min_len, sortie_max_len] (actionable for DroneSortieOptimizer)
      3) imbalance_bin: discretized imbalance of system finish times
    """
    if total_gens <= 0:
        phase = 0
    else:
        third = max(1, total_gens // 3)
        if gen <= third:
            phase = 0
        elif gen <= 2 * third:
            phase = 1
        else:
            phase = 2

    # Count actionable drones based on consecutive runs in vehicle_assignment.
    drones = fleet.all_drone_ids()
    max_run: dict[int, int] = {d: 0 for d in drones}
    i = 0
    R = chrom.length
    while i < R:
        vid = chrom.vehicle_assignment[i]
        j = i + 1
        while j < R and chrom.vehicle_assignment[j] == vid:
            j += 1
        run_len = j - i
        if vid in max_run:
            if run_len > max_run[vid]:
                max_run[vid] = run_len
        i = j

    drone_actionable_count = sum(
        1
        for d, run_len in max_run.items()
        if sortie_min_len <= run_len <= sortie_max_len
    )

    # Imbalance based on per-system finish_time.
    # Prefer cached times from the last evaluation (avoid extra decode).
    times = system_finish_times
    if times is None or len(times) == 0:
        sol = decoder.decode(chrom, w_inf=w_inf)
        times = [r.finish_time for r in sol.truck_routes]
    if not times:
        imbalance = 0.0
    else:
        t_max = max(times)
        t_min = min(times)
        imbalance = (t_max - t_min) / (t_max + 1e-9)

    if imbalance <= imbalance_low:
        imbalance_bin = 0
    elif imbalance <= imbalance_high:
        imbalance_bin = 1
    else:
        imbalance_bin = 2

    return (phase, int(drone_actionable_count), int(imbalance_bin))
