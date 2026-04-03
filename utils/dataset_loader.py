from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_EDGE_RE = re.compile(
	r"\(\s*(?P<u>\d+)\s*,\s*(?P<v>\d+)\s*\)\s+coste\s+(?P<c>\d+(?:\.\d+)?)",
	flags=re.IGNORECASE,
)

_COORD_LINE_RE = re.compile(
	r"^(?P<node>\d+)\s+(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class LoadedInstance:
	name: str
	nodes: list[int]
	depot_id: int
	coords: dict[int, tuple[float, float]] | None

	required_edge_ids: list[int]
	edge_info: Callable[[int], tuple[int, int, float]]
	truck_dist: Callable[[int, int], float]
	truck_path: Callable[[int, int], list[int]]
	drone_dist: Callable[[int, int], float]


def load_urpp_like_instance(
	file_path: str | Path,
	*,
	depot_id: int | None = None,
	drone_dist_scale: float = 1.0,
	require_connected: bool = True,
	precompute_all_pairs_if_n_leq: int = 250,
) -> LoadedInstance:
	"""Load a URPP-style dataset instance.

	Supported sections:
	  - LISTA_ARISTAS_REQ :    required edges (served exactly once)
	  - LISTA_ARISTAS_NOREQ : non-required edges (for shortest paths)
	  - COORDS :              optional node coordinates (one per line: "node x y")

	Returns callables compatible with the rest of the project:
	  - required_edge_ids: list[int] (1..R)
	  - edge_info(eid) -> (u, v, length)
	  - truck_dist(u, v) -> float  (shortest path on full graph)
	  - truck_path(u, v) -> list[int]
	  - drone_dist(u, v) -> float  (euclidean)
	"""

	path = Path(file_path)
	lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

	name = path.stem
	mode: str | None = None

	coords: dict[int, tuple[float, float]] | None = None
	req_edges: list[tuple[int, int, float]] = []
	all_edges: list[tuple[int, int, float]] = []
	nodes_set: set[int] = set()

	for raw in lines:
		line = raw.strip()
		if not line:
			continue

		upper = line.upper()
		if upper.startswith("NOMBRE"):
			parts = line.split(":", 1)
			if len(parts) == 2 and parts[1].strip():
				name = parts[1].strip()
			continue

		if upper.startswith("COORDS"):
			coords = coords or {}
			mode = "coords"
			continue

		if upper.startswith("LISTA_ARISTAS_REQ"):
			mode = "req"
			continue

		if upper.startswith("LISTA_ARISTAS_NOREQ"):
			mode = "noreq"
			continue

		if mode == "coords":
			m = _COORD_LINE_RE.match(line)
			if m:
				nid = int(m.group("node"))
				x = float(m.group("x"))
				y = float(m.group("y"))
				coords[nid] = (x, y)
				nodes_set.add(nid)
			continue

		m = _EDGE_RE.search(line)
		if not m or mode not in {"req", "noreq"}:
			continue

		u = int(m.group("u"))
		v = int(m.group("v"))
		c = float(m.group("c"))

		nodes_set.add(u)
		nodes_set.add(v)
		all_edges.append((u, v, c))
		if mode == "req":
			req_edges.append((u, v, c))

	if not nodes_set or not all_edges or not req_edges:
		raise ValueError(f"Failed to parse instance (no edges/required edges): {path}")

	nodes = sorted(nodes_set)
	if depot_id is None:
		depot_id = 1 if 1 in nodes_set else nodes[0]

	# Build adjacency with minimal weight per undirected pair
	adj_map: dict[int, dict[int, float]] = {n: {} for n in nodes}
	for u, v, w in all_edges:
		prev = adj_map[u].get(v)
		if prev is None or w < prev:
			adj_map[u][v] = w
			adj_map[v][u] = w

	if require_connected:
		reachable = _bfs_reachable(adj_map, depot_id)
		if len(reachable) != len(nodes):
			missing = [n for n in nodes if n not in reachable]
			raise ValueError(
				f"Graph not connected from depot {depot_id}. "
				f"Unreachable nodes: {missing[:10]}" + ("..." if len(missing) > 10 else "")
			)

	# Required edge ids remap: 1..R in file order
	required_edge_ids = list(range(1, len(req_edges) + 1))
	req_map: dict[int, tuple[int, int, float]] = {eid: req_edges[eid - 1] for eid in required_edge_ids}

	# Shortest path caches
	dist_cache: dict[int, dict[int, float]] = {}
	prev_cache: dict[int, dict[int, int | None]] = {}

	def ensure_source(src: int) -> None:
		if src in dist_cache:
			return
		dist, prev = _dijkstra(src, adj_map)
		dist_cache[src] = dist
		prev_cache[src] = prev

	if len(nodes) <= precompute_all_pairs_if_n_leq:
		for src in nodes:
			ensure_source(src)

	def truck_dist(u: int, v: int) -> float:
		ensure_source(u)
		return dist_cache[u].get(v, math.inf)

	def truck_path(u: int, v: int) -> list[int]:
		if u == v:
			return [u]
		ensure_source(u)
		prev = prev_cache[u]
		if v not in prev:
			return [u, v]
		out = [v]
		cur = v
		while cur != u:
			cur = prev.get(cur)
			if cur is None:
				return [u, v]
			out.append(cur)
		out.reverse()
		return out

	def edge_info(eid: int) -> tuple[int, int, float]:
		return req_map[eid]

	if coords and len(coords) >= 2:
		def drone_dist(u: int, v: int) -> float:
			xu, yu = coords[u]
			xv, yv = coords[v]
			return math.hypot(xu - xv, yu - yv)
	else:
		coords = None

		def drone_dist(u: int, v: int) -> float:
			return drone_dist_scale * truck_dist(u, v)

	return LoadedInstance(
		name=name,
		nodes=nodes,
		depot_id=depot_id,
		coords=coords,
		required_edge_ids=required_edge_ids,
		edge_info=edge_info,
		truck_dist=truck_dist,
		truck_path=truck_path,
		drone_dist=drone_dist,
	)


def _bfs_reachable(adj_map: dict[int, dict[int, float]], start: int) -> set[int]:
	if start not in adj_map:
		return set()
	q = [start]
	seen = {start}
	while q:
		u = q.pop()
		for v in adj_map[u].keys():
			if v not in seen:
				seen.add(v)
				q.append(v)
	return seen


def _dijkstra(
	src: int,
	adj_map: dict[int, dict[int, float]],
) -> tuple[dict[int, float], dict[int, int | None]]:
	import heapq

	dist: dict[int, float] = {n: math.inf for n in adj_map.keys()}
	prev: dict[int, int | None] = {src: None}

	dist[src] = 0.0
	pq: list[tuple[float, int]] = [(0.0, src)]

	while pq:
		d, u = heapq.heappop(pq)
		if d != dist[u]:
			continue
		for v, w in adj_map[u].items():
			nd = d + w
			if nd < dist[v]:
				dist[v] = nd
				prev[v] = u
				heapq.heappush(pq, (nd, v))

	return dist, prev

