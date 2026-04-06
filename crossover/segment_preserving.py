import random
from data.chromosome import Chromosome
from data.fleet_config import FleetConfig
from mutation.base import CrossoverOperator
from .ox_crossover import OXCrossover
from .pmx_crossover import PMXCrossover


class SegmentPreservingCrossover(CrossoverOperator):
    """
    Segment-Preserving Crossover (SPC) — novel operator trong Section 4.6.

    Ý tưởng: bảo toàn nguyên vẹn segment của một truck system
    (truck + drones của nó) từ một parent, trong khi phần còn lại
    được recombine từ parent kia.

    Các bước:
        1. Chọn ngẫu nhiên một truck system k
        2. Trích segment_k từ p1 (tất cả indices thuộc system k)
        3. Trích segment_k tương ứng từ p2
        4. Áp OX hoặc PMX (ngẫu nhiên) lên 2 segment đó
           → tạo ra 2 sub-segment mới
        5. Reintegrate: thay segment_k trong bản sao p1/p2
           bằng sub-segment mới
        6. Repair: sửa duplicate/missing edges
    """

    def __init__(self, fleet: FleetConfig):
        self.fleet   = fleet
        self._ox     = OXCrossover()
        self._pmx    = PMXCrossover()

    def cross(self, p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
        # Default behavior: chọn ngẫu nhiên một truck system
        k = random.randint(1, self.fleet.num_trucks)
        return self.cross_with_system(p1, p2, k)

    def cross_with_system(self, p1: Chromosome, p2: Chromosome, k: int) -> tuple[Chromosome, Chromosome]:
        """Crossover với truck system k được chỉ định (phục vụ RL action)."""
        if k < 1 or k > self.fleet.num_trucks:
            raise ValueError(f"Invalid system k={k}")

        # Áp OX hoặc PMX lên segment (chọn ngẫu nhiên) — dùng CHUNG cho cả hai con
        sub_op = self._ox if random.random() < 0.5 else self._pmx

        o1 = self._cross_one_with_subop(p1, p2, k, sub_op)
        o2 = self._cross_one_with_subop(p2, p1, k, sub_op)
        return o1, o2

    def cross_one(self, primary: Chromosome, secondary: Chromosome, k: int) -> Chromosome:
        """Tạo 1 offspring base theo primary, phối hợp từ secondary.

        Dùng khi RL coi (primary, secondary) là một quyết định riêng.
        """
        if k < 1 or k > self.fleet.num_trucks:
            raise ValueError(f"Invalid system k={k}")

        sub_op = self._ox if random.random() < 0.5 else self._pmx
        return self._cross_one_with_subop(primary, secondary, k, sub_op)

    def _cross_one_with_subop(
        self,
        primary: Chromosome,
        secondary: Chromosome,
        k: int,
        sub_op: CrossoverOperator,
    ) -> Chromosome:
        system_vids = self.fleet.system_ids(k)

        # Lấy indices thuộc system k trong mỗi parent
        indices_p1 = primary.segment_of_system(system_vids)
        indices_p2 = secondary.segment_of_system(system_vids)

        if not indices_p1 or not indices_p2:
            # Fallback: không có gì để trao đổi → trả về clone của primary
            return primary.clone()

        # Trích sub-chromosome của segment
        sub_p1 = self._extract_segment(primary, indices_p1)
        sub_p2 = self._extract_segment(secondary, indices_p2)

        # Nếu segment cùng độ dài thì dùng OX/PMX chuẩn.
        # Nếu lệch độ dài, vẫn lai theo biến thể OX/PMX (variable-length):
        # mapping vượt index -> gán "invalid" (None) rồi lấp đầy.
        if sub_p1.length == sub_p2.length and sub_p1.length >= 2:
            sub_o1, _ = sub_op.cross(sub_p1, sub_p2)  # lấy offspring base theo primary
        else:
            if isinstance(sub_op, OXCrossover):
                sub_o1 = self._variable_length_ox(sub_p1, sub_p2)
            else:
                sub_o1 = self._variable_length_pmx(sub_p1, sub_p2)

        # Reintegrate vào bản sao của primary
        o1 = self._reintegrate(primary, indices_p1, sub_o1)

        # Repair duplicate / missing
        all_eids = [abs(e) for e in primary.service_sequence]
        o1 = self._repair_chromosome(o1, all_eids)

        return o1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_segment(chrom: Chromosome, indices: list[int]) -> Chromosome:
        """Tạo sub-chromosome từ các indices chỉ định."""
        seq  = [chrom.service_sequence[i]  for i in indices]
        asgn = [chrom.vehicle_assignment[i] for i in indices]
        return Chromosome(seq, asgn)

    @staticmethod
    def _reintegrate(original: Chromosome,
                     indices: list[int],
                     segment: Chromosome) -> Chromosome:
        """Thay các vị trí indices trong original bằng segment mới."""
        new_seq  = original.service_sequence[:]
        new_asgn = original.vehicle_assignment[:]
        for pos, idx in enumerate(indices):
            new_seq[idx]  = segment.service_sequence[pos]
            new_asgn[idx] = segment.vehicle_assignment[pos]
        return Chromosome(new_seq, new_asgn)

    @staticmethod
    def _repair_chromosome(chrom: Chromosome, all_eids: list[int]) -> Chromosome:
        """Sửa duplicate/missing trong service_sequence, giữ vehicle_assignment."""
        seq = chrom.service_sequence[:]
        seen: dict[int, list[int]] = {}
        for i, e in enumerate(seq):
            seen.setdefault(abs(e), []).append(i)

        dup_indices = sorted(
            idx for idxs in seen.values() if len(idxs) > 1 for idx in idxs[1:]
        )
        missing = [e for e in all_eids if e not in seen or len(seen[e]) == 0]

        for idx, eid in zip(dup_indices, missing):
            sign = 1 if random.random() < 0.5 else -1
            seq[idx] = sign * eid

        return Chromosome(seq, chrom.vehicle_assignment[:])

    @staticmethod
    def _variable_length_ox(primary: Chromosome, secondary: Chromosome) -> Chromosome:
        """Biến thể OX cho 2 segment lệch độ dài.

        Output có đúng độ dài của primary.
        - Copy đoạn [i:j] từ primary
        - Fill phần còn lại theo thứ tự xuất hiện từ secondary (bỏ trùng theo abs(eid))
        - Nếu thiếu gene (do secondary ngắn), fill tiếp từ phần còn lại của primary
        """
        Lp = primary.length
        if Lp == 0:
            return Chromosome([], [])
        if Lp < 2:
            return primary.clone()

        i, j = sorted(random.sample(range(Lp), 2))

        out_seq: list[int | None] = [None] * Lp
        out_asg: list[int | None] = [None] * Lp

        out_seq[i:j+1] = primary.service_sequence[i:j+1]
        out_asg[i:j+1] = primary.vehicle_assignment[i:j+1]
        copied = {abs(e) for e in primary.service_sequence[i:j+1]}

        # Vị trí cần lấp theo kiểu OX (xoay vòng từ j+1)
        fill_pos = [(j + 1 + k) % Lp for k in range(Lp - (j - i + 1))]

        # Candidate từ secondary (xoay vòng cho gần OX), rồi fallback sang primary
        sec_pairs = list(zip(secondary.service_sequence, secondary.vehicle_assignment))
        if sec_pairs:
            start = (j + 1) % len(sec_pairs)
            sec_pairs = sec_pairs[start:] + sec_pairs[:start]

        prim_pairs = list(zip(primary.service_sequence, primary.vehicle_assignment))
        prim_start = (j + 1) % len(prim_pairs)
        prim_pairs = prim_pairs[prim_start:] + prim_pairs[:prim_start]

        fill_pairs: list[tuple[int, int]] = []
        for eid, vid in sec_pairs:
            if abs(eid) not in copied:
                fill_pairs.append((eid, vid))
                copied.add(abs(eid))

        for eid, vid in prim_pairs:
            if len(fill_pairs) >= len(fill_pos):
                break
            if abs(eid) not in copied:
                fill_pairs.append((eid, vid))
                copied.add(abs(eid))

        for (pos, (eid, vid)) in zip(fill_pos, fill_pairs):
            out_seq[pos] = eid
            out_asg[pos] = vid

        # Safety: nếu còn None (không kỳ vọng), giữ lại gene của primary
        for idx in range(Lp):
            if out_seq[idx] is None:
                out_seq[idx] = primary.service_sequence[idx]
            if out_asg[idx] is None:
                out_asg[idx] = primary.vehicle_assignment[idx]

        return Chromosome(out_seq, out_asg)

    @staticmethod
    def _variable_length_pmx(primary: Chromosome, secondary: Chromosome) -> Chromosome:
        """Biến thể PMX cho 2 segment lệch độ dài.

        Ý tưởng:
        - Copy đoạn [i:j] từ primary
        - Tạo mapping theo vị trí giữa segment của primary và secondary:
            map[p_abs[k]] = s_abs[k] nếu k < len(secondary)
          Nếu k vượt index của secondary -> coi như invalid.
        - Với vị trí ngoài segment: lấy candidate từ secondary nếu có (k < Ls),
          nếu conflict thì follow chain mapping; nếu chain rơi vào invalid/không có,
          tạm để None rồi sẽ lấp đầy.
        - Cuối cùng lấp các None bằng các gene còn thiếu (ưu tiên theo thứ tự của secondary,
          rồi primary) và gán assignment theo nguồn của gene.
        """
        Lp = primary.length
        if Lp == 0:
            return Chromosome([], [])
        if Lp < 2:
            return primary.clone()

        Ls = secondary.length
        i, j = sorted(random.sample(range(Lp), 2))

        p_abs = [abs(e) for e in primary.service_sequence]
        s_abs = [abs(e) for e in secondary.service_sequence]

        out_seq: list[int | None] = [None] * Lp
        out_asg: list[int | None] = [None] * Lp

        # Quick lookup abs(eid) -> (signed_eid, vid)
        sec_lookup: dict[int, tuple[int, int]] = {
            abs(e): (e, v) for e, v in zip(secondary.service_sequence, secondary.vehicle_assignment)
        }
        prim_lookup: dict[int, tuple[int, int]] = {
            abs(e): (e, v) for e, v in zip(primary.service_sequence, primary.vehicle_assignment)
        }

        # Copy segment from primary
        out_seq[i:j+1] = primary.service_sequence[i:j+1]
        out_asg[i:j+1] = primary.vehicle_assignment[i:j+1]
        in_seg = set(p_abs[i:j+1])

        # Build mapping (primary segment abs -> secondary segment abs) when secondary has that index
        mapping: dict[int, int] = {}
        for k in range(i, j + 1):
            if k < Ls:
                mapping[p_abs[k]] = s_abs[k]

        def resolve(val_abs: int) -> int | None:
            """Follow PMX mapping chain; return a non-conflicting abs eid or None if invalid."""
            seen_chain: set[int] = set()
            cur = val_abs
            while cur in in_seg:
                if cur in seen_chain:
                    return None
                seen_chain.add(cur)
                nxt = mapping.get(cur)
                if nxt is None:
                    return None
                cur = nxt
            return cur

        # Fill outside segment using secondary candidates when available
        for k in list(range(0, i)) + list(range(j + 1, Lp)):
            if k >= Ls:
                continue  # invalid -> leave None for now
            cand = s_abs[k]
            resolved = resolve(cand)
            if resolved is None:
                continue
            signed, vid = sec_lookup.get(resolved) or prim_lookup.get(resolved)  # type: ignore[assignment]
            out_seq[k] = signed
            out_asg[k] = vid
            in_seg.add(resolved)

        # Determine which abs-ids are still missing in this offspring segment
        used_abs = {abs(e) for e in out_seq if e is not None}

        # Fill remaining slots: prefer secondary order, then primary order
        candidates: list[int] = []
        for v in s_abs:
            if v not in used_abs:
                candidates.append(v)
                used_abs.add(v)
        for v in p_abs:
            if v not in used_abs:
                candidates.append(v)
                used_abs.add(v)

        cand_it = iter(candidates)
        for idx in range(Lp):
            if out_seq[idx] is None:
                v = next(cand_it, None)
                if v is None:
                    # Fallback: keep primary gene (should be rare)
                    out_seq[idx] = primary.service_sequence[idx]
                    out_asg[idx] = primary.vehicle_assignment[idx]
                else:
                    signed, vid = sec_lookup.get(v) or prim_lookup.get(v)  # type: ignore[assignment]
                    out_seq[idx] = signed
                    out_asg[idx] = vid

        return Chromosome(out_seq, out_asg)
