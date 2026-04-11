from __future__ import annotations

from typing import Hashable

from data.chromosome import Chromosome
from data.fleet_config import FleetConfig
from evaluation.decoder import Decoder


def count_actionable_drones(
    chrom: Chromosome,
    fleet: FleetConfig,
    *,
    sortie_min_len: int,
    sortie_max_len: int = 5,
) -> int:
    """Count drones having at least one consecutive run with length in [min,max]."""

    drones = set(fleet.all_drone_ids())
    max_run: dict[int, int] = {d: 0 for d in drones}
    i = 0
    R = chrom.length
    while i < R:
        vid = chrom.vehicle_assignment[i]
        j = i + 1
        while j < R and chrom.vehicle_assignment[j] == vid:
            j += 1
        run_len = j - i
        if vid in max_run and run_len > max_run[vid]:
            max_run[vid] = run_len
        i = j

    return sum(1 for run_len in max_run.values() if sortie_min_len <= run_len <= sortie_max_len)


def _edge_endpoints(decoder: Decoder, signed_eid: int) -> tuple[int, int, float]:
    """Return directed endpoints (u->v) and length for a signed required-edge id."""
    u, v, length = decoder.edge_info(abs(signed_eid))
    return (u, v, length) if signed_eid > 0 else (v, u, length)


def _bin_ratio(x: float, low: float, high: float) -> int:
    if x <= low:
        return 0
    if x <= high:
        return 1
    return 2


def _bin_stagnation(steps: int, low: int, high: int) -> int:
    if steps <= low:
        return 0
    if steps <= high:
        return 1
    return 2


def _seq_disorder_ratio(chrom: Chromosome, fleet: FleetConfig, decoder: Decoder) -> float:
    """Estimate truck travel waste ratio based on deadhead vs service time.

    travel_waste = deadhead_time / (deadhead_time + service_time)

    - deadhead_time: depot->edge_start, between consecutive truck edges, edge_end->depot
      using shortest-path distance truck_dist.
    - service_time: sum(edge_length / truck_speed) for truck-served required edges.

    This ignores waiting effects induced by drone rendezvous, so it acts as a
    sequence-quality proxy (bad ordering => more deadhead).
    """

    depot = fleet.depot_id
    deadhead = 0.0
    service = 0.0

    # Process each truck independently (system k=1..K)
    for k in range(1, fleet.num_trucks + 1):
        tid = fleet.truck_id(k)
        truck_edges = [
            chrom.service_sequence[i]
            for i in range(chrom.length)
            if chrom.vehicle_assignment[i] == tid
        ]
        if not truck_edges:
            continue

        cur = depot
        for signed_eid in truck_edges:
            u, v, length = _edge_endpoints(decoder, signed_eid)
            deadhead += decoder.truck_dist(cur, u) / fleet.truck_speed
            service += length / fleet.truck_speed
            cur = v

        deadhead += decoder.truck_dist(cur, depot) / fleet.truck_speed

    total = deadhead + service
    if total <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, deadhead / total))


def _drone_waste_ratio(chrom: Chromosome, fleet: FleetConfig, decoder: Decoder) -> float:
    """Estimate drone inefficiency as reposition/(reposition+service) within sorties.

    For each maximal consecutive run of a drone id in vehicle_assignment, treat it as
    one sortie in chromosome space. Compute:
      - service_time: sum(edge_length / drone_speed)
      - reposition_time: sum(drone_dist(prev_edge_end, next_edge_start) / drone_speed)

    This ignores launch/land reposition legs because those depend on rendezvous.
    """

    drones = set(fleet.all_drone_ids())
    reposition = 0.0
    service = 0.0

    i = 0
    R = chrom.length
    while i < R:
        vid = chrom.vehicle_assignment[i]
        j = i + 1
        while j < R and chrom.vehicle_assignment[j] == vid:
            j += 1
        if vid in drones and (j - i) >= 1:
            run = chrom.service_sequence[i:j]
            prev_end: int | None = None
            for signed_eid in run:
                u, v, length = _edge_endpoints(decoder, signed_eid)
                service += length / fleet.drone_speed
                if prev_end is not None:
                    reposition += decoder.drone_dist(prev_end, u) / fleet.drone_speed
                prev_end = v
        i = j

    total = reposition + service
    if total <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, reposition / total))


def build_ls_state(
    chrom: Chromosome,
    fleet: FleetConfig,
    decoder: Decoder,
    *,
    sortie_min_len: int,
    sortie_max_len: int = 5,
    system_finish_times: list[float] | None = None,
    # Thresholds (3-bin discretization)
    assign_imbalance_low: float = 0.10,
    assign_imbalance_high: float = 0.25,
    seq_disorder_low: float = 0.25,
    seq_disorder_high: float = 0.45,
    drone_waste_low: float = 0.20,
    drone_waste_high: float = 0.40,
    stagnation: int = 0,
    stagnation_low: int = 2,
    stagnation_high: int = 8,
) -> Hashable:
    """Local-search RL state.

    Features (4):
      1) assign_imbalance_bin: bin((max_finish - min_finish) / max_finish)
         (high -> GreedyVehicleReassignment may help)
      2) seq_disorder_bin: bin(truck_travel_waste / total_truck_time)
         (high -> SubsequenceReversal / OrOpt may help)
      3) drone_waste_bin: bin(reposition / (reposition + service))
         (high -> DroneSortieOptimizer may help)
      4) stagnation_bin: bin(#consecutive LS steps without improvement)
    """

    # Ensure we touch these params (they are used outside for action filtering).
    _ = (sortie_min_len, sortie_max_len)

    times = system_finish_times
    if times is None or len(times) == 0:
        sol = decoder.decode(chrom, w_inf=1.0)
        times = [r.finish_time for r in sol.truck_routes]

    if not times:
        assign_ratio = 0.0
    else:
        t_max = max(times)
        t_min = min(times)
        assign_ratio = 0.0 if t_max <= 1e-12 else (t_max - t_min) / t_max
        assign_ratio = max(0.0, min(1.0, assign_ratio))

    seq_ratio = _seq_disorder_ratio(chrom, fleet, decoder)
    drone_ratio = _drone_waste_ratio(chrom, fleet, decoder)

    assign_bin = _bin_ratio(assign_ratio, assign_imbalance_low, assign_imbalance_high)
    seq_bin = _bin_ratio(seq_ratio, seq_disorder_low, seq_disorder_high)
    drone_bin = _bin_ratio(drone_ratio, drone_waste_low, drone_waste_high)
    stag_bin = _bin_stagnation(int(stagnation), stagnation_low, stagnation_high)

    return (int(assign_bin), int(seq_bin), int(drone_bin), int(stag_bin))
