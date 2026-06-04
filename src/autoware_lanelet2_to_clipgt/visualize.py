"""Top-down (XY) preview of a ClipGT HDMap scene.

Usage:
    python visualize_clipgt_topdown.py path/to/clipgt-<uuid>.usdz
    python visualize_clipgt_topdown.py path/to/extracted/clipgt/

Pass a ``.usdz`` (extracted to a temp dir) or a directory that contains the
``*.parquet`` files directly. No CUDA / Ludus required.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection


LINE_ELEMENTS = ("road_boundary", "wait_line")
POLY_ELEMENTS = (
    "crosswalk",
    "road_marking",
    "intersection_area",
    "road_island",
)
POINT_ELEMENTS = ("pole",)

STYLE = {
    "road_boundary": {"color": "#444444", "linewidth": 1.2, "zorder": 3},
    "wait_line": {"color": "#cc2222", "linewidth": 1.8, "zorder": 4},
    "crosswalk": {"facecolor": "#ffe066", "edgecolor": "#bb9900", "alpha": 0.55, "zorder": 2},
    "road_marking": {"facecolor": "#aaaaff", "edgecolor": "#5555aa", "alpha": 0.5, "zorder": 2},
    "intersection_area": {"facecolor": "#dddddd", "edgecolor": "none", "alpha": 0.35, "zorder": 1},
    "road_island": {"facecolor": "#88cc88", "edgecolor": "#447744", "alpha": 0.5, "zorder": 2},
    "pole": {"color": "#333333", "marker": "o", "s": 6, "zorder": 5},
    "lane_white_solid": {"color": "#ffffff", "linewidth": 0.9, "zorder": 3},
    "lane_white_dashed": {"color": "#ffffff", "linewidth": 0.9, "linestyle": (0, (4, 4)), "zorder": 3},
    "lane_yellow_solid": {"color": "#ffcc00", "linewidth": 1.0, "zorder": 3},
    "lane_yellow_dashed": {"color": "#ffcc00", "linewidth": 1.0, "linestyle": (0, (4, 4)), "zorder": 3},
    "traffic_light": {"color": "#ff3333", "marker": "s", "s": 14, "zorder": 6},
    "traffic_sign": {"color": "#3366ff", "marker": "^", "s": 14, "zorder": 6},
    "ego": {"color": "#ff0066", "linewidth": 2.0, "zorder": 7},
}


def resolve_parquet(scene_dir: Path, element: str) -> Path | None:
    simple = scene_dir / f"{element}.parquet"
    if simple.exists():
        return simple
    matches = list(scene_dir.glob(f"*.{element}.parquet"))
    return matches[0] if matches else None


def extract_points(point_list) -> list[tuple[float, float]] | None:
    if point_list is None or len(point_list) == 0:
        return None
    out = []
    for p in point_list:
        if p is None:
            continue
        x, y = p.get("x"), p.get("y")
        if x is None or y is None:
            continue
        out.append((float(x), float(y)))
    return out or None


def read_polylines(path: Path, key: str, sub: str = "location") -> list[list[tuple[float, float]]]:
    df = pd.read_parquet(path)
    polylines: list[list[tuple[float, float]]] = []
    for _, row in df.iterrows():
        elem = row[key]
        pts = elem.get(sub) if isinstance(elem, dict) else getattr(elem, sub, None)
        xy = extract_points(pts)
        if xy and len(xy) >= 2:
            polylines.append(xy)
    return polylines


def read_polygons(path: Path, key: str) -> list[list[tuple[float, float]]]:
    df = pd.read_parquet(path)
    polys: list[list[tuple[float, float]]] = []
    for _, row in df.iterrows():
        elem = row[key]
        pts = elem.get("location") if isinstance(elem, dict) else getattr(elem, "location", None)
        xy = extract_points(pts)
        if xy and len(xy) >= 3:
            polys.append(xy)
    return polys


def read_points(path: Path, key: str) -> list[tuple[float, float]]:
    df = pd.read_parquet(path)
    pts: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        elem = row[key]
        loc = elem.get("location") if isinstance(elem, dict) else getattr(elem, "location", None)
        xy = extract_points(loc)
        if xy:
            pts.extend(xy)
    return pts


def read_lane_lines(path: Path):
    """Split lane lines into (white_solid, white_dashed, yellow_solid, yellow_dashed)."""
    buckets = {"lane_white_solid": [], "lane_white_dashed": [],
               "lane_yellow_solid": [], "lane_yellow_dashed": []}
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        ll = row["lane_line"]
        pts = ll.get("line_rail") if isinstance(ll, dict) else getattr(ll, "line_rail", None)
        if pts is None:
            pts = ll.get("path") if isinstance(ll, dict) else getattr(ll, "path", None)
        xy = extract_points(pts)
        if not xy or len(xy) < 2:
            continue
        colors = ll.get("colors", []) if isinstance(ll, dict) else getattr(ll, "colors", [])
        styles = ll.get("styles", []) if isinstance(ll, dict) else getattr(ll, "styles", [])
        color = (str(colors[0]).upper() if len(colors) else "WHITE")
        style = (str(styles[0]).upper() if len(styles) else "SOLID")
        is_yellow = color == "YELLOW"
        is_dashed = "DASHED" in style or "DOT" in style
        bucket = (
            "lane_yellow_dashed" if is_yellow and is_dashed
            else "lane_yellow_solid" if is_yellow
            else "lane_white_dashed" if is_dashed
            else "lane_white_solid"
        )
        buckets[bucket].append(xy)
    return buckets


def read_centers(path: Path, key: str) -> list[tuple[float, float]]:
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        elem = row[key]
        c = elem.get("center") if isinstance(elem, dict) else getattr(elem, "center", None)
        if c is None:
            continue
        x, y = c.get("x"), c.get("y")
        if x is None or y is None:
            continue
        out.append((float(x), float(y)))
    return out


def read_ego_xy(path: Path) -> list[tuple[float, float]]:
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        ego = row["egomotion_estimate"]
        loc = ego.get("location") if isinstance(ego, dict) else getattr(ego, "location", None)
        if loc is None:
            continue
        x, y = loc.get("x"), loc.get("y")
        if x is None or y is None:
            continue
        out.append((float(x), float(y)))
    return out


def find_scene_dir(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if input_path.is_dir():
        if list(input_path.glob("*.parquet")):
            return input_path, None
        for sub in ("clipgt", "."):
            cand = input_path / sub
            if cand.is_dir() and list(cand.glob("*.parquet")):
                return cand, None
        raise FileNotFoundError(f"No *.parquet found under {input_path}")
    if input_path.suffix.lower() in {".usdz", ".zip"}:
        tmp = tempfile.TemporaryDirectory(prefix="clipgt-vis-")
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        cand = root / "clipgt"
        if cand.is_dir() and list(cand.glob("*.parquet")):
            return cand, tmp
        if list(root.glob("*.parquet")):
            return root, tmp
        for sub in root.iterdir():
            if sub.is_dir() and list(sub.glob("*.parquet")):
                return sub, tmp
        raise FileNotFoundError(f"No *.parquet inside archive {input_path}")
    raise ValueError(f"Unsupported input: {input_path}")


def plot_scene(scene_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor("#222222")

    for elem in POLY_ELEMENTS:
        p = resolve_parquet(scene_dir, elem)
        if p is None:
            continue
        polys = read_polygons(p, elem)
        if not polys:
            continue
        ax.add_collection(PolyCollection(polys, **STYLE[elem]))
        print(f"  {elem}: {len(polys)} polygons")

    for elem in LINE_ELEMENTS:
        p = resolve_parquet(scene_dir, elem)
        if p is None:
            continue
        lines = read_polylines(p, elem)
        if not lines:
            continue
        ax.add_collection(LineCollection(lines, **STYLE[elem]))
        print(f"  {elem}: {len(lines)} polylines")

    ll_path = resolve_parquet(scene_dir, "lane_line")
    if ll_path is not None:
        buckets = read_lane_lines(ll_path)
        for name, lines in buckets.items():
            if not lines:
                continue
            ax.add_collection(LineCollection(lines, **STYLE[name]))
            print(f"  {name}: {len(lines)} polylines")

    for elem in POINT_ELEMENTS:
        p = resolve_parquet(scene_dir, elem)
        if p is None:
            continue
        pts = read_points(p, elem)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, **STYLE[elem])
        print(f"  {elem}: {len(pts)} points")

    for elem in ("traffic_light", "traffic_sign"):
        p = resolve_parquet(scene_dir, elem)
        if p is None:
            continue
        pts = read_centers(p, elem)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, **STYLE[elem], label=elem)
        print(f"  {elem}: {len(pts)} items")

    ego_path = resolve_parquet(scene_dir, "egomotion_estimate")
    if ego_path is not None:
        ego = read_ego_xy(ego_path)
        if ego:
            xs, ys = zip(*ego)
            ax.plot(xs, ys, **STYLE["ego"], label="ego trajectory")
            ax.scatter([xs[0]], [ys[0]], color="#ff0066", marker="*", s=120, zorder=8, label="ego start")
            print(f"  ego: {len(ego)} poses")

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.2, color="#888888")
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to .usdz or scene dir with *.parquet")
    parser.add_argument("--save", type=Path, default=None, help="If set, save PNG instead of showing window")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} does not exist", file=sys.stderr)
        return 1

    scene_dir, tmp = find_scene_dir(args.input)
    print(f"Reading {scene_dir}")
    try:
        if args.save:
            import matplotlib
            matplotlib.use("Agg")
            plot_scene(scene_dir, args.input.name)
            plt.savefig(args.save, dpi=150)
            print(f"Saved {args.save}")
        else:
            plot_scene(scene_dir, args.input.name)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
