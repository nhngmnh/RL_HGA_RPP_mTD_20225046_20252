from __future__ import annotations

from typing import Hashable

from data.chromosome import Chromosome
from data.fleet_config import FleetConfig
from evaluation.decoder import Decoder


def _ranks_desc(values: list[float]) -> list[int]:
    """Return 1-based ranks (descending). Ties get the same rank."""
    # Pair each value with original index
    indexed = list(enumerate(values))
    indexed.sort(key=lambda t: t[1], reverse=True)

    ranks = [0] * len(values)
    rank = 1
    prev = None
    for pos, (idx, val) in enumerate(indexed):
        if prev is None:
            ranks[idx] = rank
            prev = val
            continue
        if val != prev:
            rank = pos + 1
            prev = val
        ranks[idx] = rank
    return ranks


def build_spc_state(
    p1: Chromosome,
    p2: Chromosome,
    fleet: FleetConfig,
    decoder: Decoder,
    w_inf: float = 1.0,
) -> Hashable:
    """Build SPC state as tuple((rank_idx_k, rank_time_k) for k=1..K).

    Definitions (p1 is base):
      - idx distance for system k: |A_k \ B_k| where A_k/B_k are index sets
      - time distance for system k: |T_k(p1) - T_k(p2)| where T_k is system finish_time
    """
    K = fleet.num_trucks

    # Index distances
    idx_dists: list[float] = []
    for k in range(1, K + 1):
        sys_vids = fleet.system_ids(k)
        A = set(p1.segment_of_system(sys_vids))
        B = set(p2.segment_of_system(sys_vids))
        idx_dists.append(float(len(A - B)))

    # Time distances
    sol1 = decoder.decode(p1, w_inf=w_inf)
    sol2 = decoder.decode(p2, w_inf=w_inf)
    # truck_routes are in order k=1..K
    t1 = [r.finish_time for r in sol1.truck_routes]
    t2 = [r.finish_time for r in sol2.truck_routes]
    time_dists = [abs(a - b) for a, b in zip(t1, t2)]

    r_idx = _ranks_desc(idx_dists)
    r_time = _ranks_desc(time_dists)

    return tuple((r_idx[k - 1], r_time[k - 1]) for k in range(1, K + 1))
