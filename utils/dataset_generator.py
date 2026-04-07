from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable


def manhattan_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
	return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _all_undirected_pairs(nodes: list[int]) -> list[tuple[int, int]]:
	out: list[tuple[int, int]] = []
	for i, u in enumerate(nodes):
		for v in nodes[i + 1 :]:
			out.append((u, v))
	return out


def generate_urpp_grid_instance_text(
	*,
	n_nodes: int,
	n_edges_total: int,
	n_required_edges: int,
	grid_size: int = 10,
	coord_scale: int = 10,
	seed: int | None = None,
	name: str | None = None,
	comment: str = "Generated instance",
	depot_id: int = 1,
) -> str:
	"""Generate a URPP-like instance text on a fixed grid.

	- Coordinates are sampled without replacement from a finer lattice of size
	  `(grid_size*coord_scale) x (grid_size*coord_scale)`, then divided by `coord_scale`.
	  Example: grid_size=10, coord_scale=10 => coordinates in [0, 10) with step 0.1.
	- Edge costs are Manhattan distances between node coordinates.
	- The full graph (required + non-required edges) is guaranteed connected from `depot_id`.

	The produced text is compatible with `utils.dataset_loader.load_urpp_like_instance`.
	"""

	if n_nodes < 2:
		raise ValueError("n_nodes must be >= 2")
	if grid_size <= 0:
		raise ValueError("grid_size must be > 0")
	if coord_scale <= 0:
		raise ValueError("coord_scale must be > 0")
	max_nodes = (grid_size * coord_scale) * (grid_size * coord_scale)
	if n_nodes > max_nodes:
		raise ValueError(
			f"n_nodes={n_nodes} exceeds grid capacity {max_nodes} "
			f"for grid_size={grid_size} and coord_scale={coord_scale}"
		)

	max_edges = n_nodes * (n_nodes - 1) // 2
	if n_edges_total < n_nodes - 1:
		raise ValueError("n_edges_total must be >= n_nodes-1 to allow connectivity")
	if n_edges_total > max_edges:
		raise ValueError(f"n_edges_total={n_edges_total} exceeds complete graph edges {max_edges}")
	if not (0 < n_required_edges <= n_edges_total):
		raise ValueError("n_required_edges must be in [1, n_edges_total]")
	if not (1 <= depot_id <= n_nodes):
		raise ValueError("depot_id must be in [1, n_nodes]")

	rng = random.Random(seed)

	# 1) Nodes and unique coordinates on a scaled grid (real-valued, step = 1/coord_scale)
	nodes = list(range(1, n_nodes + 1))
	size = grid_size * coord_scale
	grid_points = [(ix, iy) for ix in range(size) for iy in range(size)]
	rng.shuffle(grid_points)
	picked_points = grid_points[:n_nodes]
	coords: dict[int, tuple[float, float]] = {}
	for nid, (ix, iy) in zip(nodes, picked_points, strict=True):
		x = ix / coord_scale
		y = iy / coord_scale
		coords[nid] = (x, y)

	# 2) Ensure connectivity via a random spanning tree, then add random extra edges
	edges: set[tuple[int, int]] = set()
	order = nodes[:]
	rng.shuffle(order)

	connected: list[int] = [order[0]]
	for nid in order[1:]:
		parent = rng.choice(connected)
		u, v = sorted((nid, parent))
		edges.add((u, v))
		connected.append(nid)

	remaining = n_edges_total - len(edges)
	if remaining > 0:
		candidates = [e for e in _all_undirected_pairs(nodes) if e not in edges]
		rng.shuffle(candidates)
		edges.update(candidates[:remaining])

	edge_list = sorted(edges)

	# 3) Choose required edges among selected edges
	req_edge_set = set(rng.sample(edge_list, k=n_required_edges))
	req_edges = [e for e in edge_list if e in req_edge_set]
	noreq_edges = [e for e in edge_list if e not in req_edge_set]

	# 4) Build text
	instance_name = name or f"N{n_nodes}E{n_edges_total}R{n_required_edges}_01"

	lines: list[str] = []
	lines.append(f"NOMBRE : {instance_name}")
	lines.append(f"COMENTARIO : {comment}")
	lines.append(f"VERTICES : {n_nodes}")
	lines.append(f"ARISTAS_REQ : {n_required_edges}")
	lines.append(f"ARISTAS_NOREQ : {n_edges_total - n_required_edges}")
	lines.append("COORDS :")
	for nid in nodes:
		x, y = coords[nid]
		lines.append(f"{nid} {x:.3f} {y:.3f}")

	def fmt_edge(u: int, v: int) -> str:
		c = manhattan_dist(coords[u], coords[v])
		return f"(  {u:>2}, {v:>2})  coste     {c:.3f}    {c:.3f}"

	lines.append("LISTA_ARISTAS_REQ :")
	for u, v in req_edges:
		lines.append(fmt_edge(u, v))

	lines.append("LISTA_ARISTAS_NOREQ :")
	for u, v in noreq_edges:
		lines.append(fmt_edge(u, v))

	# Trailing newline for POSIX friendliness; ok on Windows too.
	return "\n".join(lines) + "\n"


def write_urpp_grid_instance(
	out_path: str | Path,
	*,
	n_nodes: int,
	n_edges_total: int,
	n_required_edges: int,
	grid_size: int = 10,
	coord_scale: int = 10,
	seed: int | None = None,
	name: str | None = None,
	comment: str = "Generated instance",
	depot_id: int = 1,
	encoding: str = "utf-8",
) -> Path:
	"""Write an instance to disk and return the resolved output path."""
	p = Path(out_path)
	p.parent.mkdir(parents=True, exist_ok=True)
	text = generate_urpp_grid_instance_text(
		n_nodes=n_nodes,
		n_edges_total=n_edges_total,
		n_required_edges=n_required_edges,
		grid_size=grid_size,
		coord_scale=coord_scale,
		seed=seed,
		name=name,
		comment=comment,
		depot_id=depot_id,
	)
	p.write_text(text, encoding=encoding)
	return p.resolve()


DEFAULT_BASE_SEED = 123


def ensure_urpp_grid_dataset_file(
	*,
	nodes: int,
	edges_total: int,
	required_edges: int,
	dataset_root: str | Path = "dataset",
	grid_size: int = 10,
	coord_scale: int = 10,
	base_seed: int = DEFAULT_BASE_SEED,
	index: int | None = None,
	encoding: str = "utf-8",
) -> Path:
	"""Ensure a dataset file exists following the workspace folder convention.

	Creates (if missing):
	  dataset/N{nodes}/N{nodes}E{E}R{R}/N{nodes}E{E}R{R}_{index:02d}.txt

	- If the target file already exists, it is returned as-is.
	- If index is None, picks the smallest positive index that doesn't exist yet.
	- Seed is deterministic: `base_seed + (index-1)`.
	"""

	E = edges_total
	R = required_edges
	if nodes < 2:
		raise ValueError("nodes must be >= 2")
	if E < nodes - 1:
		raise ValueError("edges_total must be >= nodes-1 to allow connectivity")
	max_edges = nodes * (nodes - 1) // 2
	if E > max_edges:
		raise ValueError(f"edges_total={E} exceeds complete graph edges {max_edges}")
	if not (1 <= R <= E):
		raise ValueError("required_edges must be in [1, edges_total]")

	prefix = f"N{nodes}E{E}R{R}"
	root = Path(dataset_root)
	folder = root / f"N{nodes}" / prefix
	folder.mkdir(parents=True, exist_ok=True)

	def build_path(k: int) -> Path:
		return folder / f"{prefix}_{k:02d}.txt"

	if index is None:
		k = 1
		while build_path(k).exists():
			k += 1
		index = k
	else:
		if index <= 0:
			raise ValueError("index must be >= 1")

	out_path = build_path(index)
	if out_path.exists():
		return out_path.resolve()

	seed = base_seed + (index - 1)
	text = generate_urpp_grid_instance_text(
		n_nodes=nodes,
		n_edges_total=E,
		n_required_edges=R,
		grid_size=grid_size,
		coord_scale=coord_scale,
		seed=seed,
		name=f"{prefix}_{index:02d}",
		comment="Generated instance",
	)
	out_path.write_text(text, encoding=encoding)
	return out_path.resolve()


def _build_cli(argv: Iterable[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Generate a URPP-like grid dataset instance (.txt).")
	parser.add_argument("--nodes", type=int, required=True, help="Number of nodes")
	parser.add_argument("--edges", type=int, required=True, help="Total undirected edges")
	parser.add_argument("--required", type=int, required=True, help="Number of required edges")
	parser.add_argument("--grid", type=int, default=10, help="Grid size (default: 10 => 10x10)")
	parser.add_argument(
		"--coord-scale",
		type=int,
		default=10,
		help="Coordinate scale within the grid (default: 10 => step 0.1 within each unit)",
	)
	parser.add_argument("--seed", type=int, default=None, help="Random seed (only used when --out is set)")
	parser.add_argument("--out", default=None, help="Output .txt path. If omitted, uses dataset/ folder convention")
	parser.add_argument("--dataset-root", default="dataset", help="Dataset root folder (default: dataset)")
	parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="Base seed for auto-index generation")
	parser.add_argument("--index", type=int, default=None, help="Explicit index for auto path (e.g. 1 => _01)")
	parser.add_argument("--name", type=str, default=None, help="Override NOMBRE (only used when --out is set)")
	parser.add_argument("--comment", type=str, default="Generated instance", help="COMENTARIO")
	args = parser.parse_args(list(argv) if argv is not None else None)

	if args.out:
		write_urpp_grid_instance(
			args.out,
			n_nodes=args.nodes,
			n_edges_total=args.edges,
			n_required_edges=args.required,
			grid_size=args.grid,
			coord_scale=args.coord_scale,
			seed=args.seed,
			name=args.name,
			comment=args.comment,
		)
	else:
		out_path = ensure_urpp_grid_dataset_file(
			nodes=args.nodes,
			edges_total=args.edges,
			required_edges=args.required,
			dataset_root=args.dataset_root,
			grid_size=args.grid,
			coord_scale=args.coord_scale,
			base_seed=args.base_seed,
			index=args.index,
		)
		print(str(out_path))
	return 0


if __name__ == "__main__":
	raise SystemExit(_build_cli())
