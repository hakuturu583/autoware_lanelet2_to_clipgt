"""Convert an Autoware Lanelet2 .osm map into a ClipGT parquet bundle.

Reference: ``docs/clipgt-format.md`` (consumed by ``ludus_renderer``).

The output is a directory of parquet files using the loose-directory layout
that :func:`visualize_clipgt_topdown.find_scene_dir` can read directly:

    <out_dir>/
      lane.parquet
      lane_line.parquet
      road_boundary.parquet
      wait_line.parquet
      crosswalk.parquet
      road_marking.parquet
      intersection_area.parquet
      road_island.parquet
      pole.parquet
      traffic_light.parquet
      traffic_sign.parquet
      egomotion_estimate.parquet   (empty placeholder)

Coordinates are projected with ``UtmProjector(useOffset=True)`` so the scene
sits near the origin in metres (ClipGT convention: right-handed, Z-up, m).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import schemas

LABEL_CLASS_ID = "lanelet2:autoware:v0"
MAP_VERSION = "1"
SCHEMA_VERSION = 1


# --- Style / category lookup tables (lanelet2 → ClipGT) ---

# linestring (type, subtype) → ClipGT lane_line `styles` enum value.
LANE_LINE_STYLE = {
    ("line_thin", "solid"): "SOLID_SINGLE",
    ("line_thin", "dashed"): "LONG_DASHED_SINGLE",
    ("line_thin", ""): "SOLID_SINGLE",
    ("line_thick", "solid"): "SOLID_GROUP",
    ("line_thick", "dashed"): "DASHED_SOLID",
    ("line_thick", ""): "SOLID_GROUP",
}

# linestring type → ClipGT road_boundary `category` enum value.
ROAD_BOUNDARY_CATEGORY = {
    "road_border": "road_boundary",
    "road_shoulder": "road_boundary",
    "fence": "fence",
    "guard_rail": "barrier",
    "wall": "wall",
}

# polygon (type, subtype) → ClipGT road_marking `category` enum value.
ROAD_MARKING_CATEGORY = {
    ("hatched_road_markings", ""): "ROI_POLYGON_KEEP_CLEAR",
    ("no_parking_area", ""): "ROI_POLYGON_KEEP_CLEAR",
    ("no_stopping_area", ""): "ROI_POLYGON_KEEP_CLEAR",
    ("pedestrian_marking", ""): "ROI_POLYGON_TEXT_PAINT",
}


# --- Helpers ---


def _pt(p) -> dict:
    return {"x": float(p.x), "y": float(p.y), "z": float(p.z)}


def _polyline(ls) -> list[dict]:
    return [_pt(p) for p in ls]


def _polygon_from_lanelet(ll) -> list[dict]:
    """Build a closed polygon from a lanelet's left + reversed right bound."""
    pts = [_pt(p) for p in ll.leftBound]
    pts.extend(_pt(p) for p in reversed(list(ll.rightBound)))
    return pts


def _attr(elem, name: str, default: str = "") -> str:
    attrs = elem.attributes
    return str(attrs[name]) if name in attrs else default


def _yaw_to_quat(yaw: float) -> dict:
    half = 0.5 * yaw
    return {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)}


def _make_key(clip_id: str, map_id: int | str) -> dict:
    return {
        "clip_id": clip_id,
        "label_class_id": LABEL_CLASS_ID,
        "map_id": str(map_id),
        "map_id_version": MAP_VERSION,
    }


def _write(rows: list[dict], schema: pa.Schema, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        empty = {field.name: [] for field in schema}
        table = pa.table(empty, schema=schema)
    else:
        table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)
    return len(rows)


# --- Per-element extractors ---


def _extract_lanes(map_, clip_id: str) -> tuple[list[dict], list[dict]]:
    """Return (lane rows, crosswalk rows from crosswalk lanelets)."""
    lanes: list[dict] = []
    crosswalks: list[dict] = []
    for ll in map_.laneletLayer:
        subtype = _attr(ll, "subtype")
        speed_limit = _attr(ll, "speed_limit")
        left_pts = _polyline(ll.leftBound)
        right_pts = _polyline(ll.rightBound)
        if len(left_pts) < 2 or len(right_pts) < 2:
            continue
        if subtype == "crosswalk":
            crosswalks.append(
                {
                    "key": _make_key(clip_id, ll.id),
                    "crosswalk": {
                        "category": "PEDESTRIAN",
                        "location": _polygon_from_lanelet(ll),
                        "egomotion_label_class_id": LABEL_CLASS_ID,
                    },
                    "version": SCHEMA_VERSION,
                }
            )
            continue
        if subtype in {"walkway", "pedestrian_lane"}:
            continue
        use_types: list[str] = []
        if subtype == "road_shoulder":
            use_types.append("SHOULDER_LANE")
        if subtype == "bicycle_lane":
            use_types.append("BICYCLE")
        lanes.append(
            {
                "key": _make_key(clip_id, ll.id),
                "lane": {
                    "left_rail": left_pts,
                    "right_rail": right_pts,
                    "left_edge_styles": [],
                    "right_edge_styles": [],
                    "left_edge_colors": [],
                    "right_edge_colors": [],
                    "lane_direction": "STRAIGHT",
                    "map_end": "NONE",
                    "speed_limit": speed_limit,
                    "vehicle_types": ["CAR"] if subtype == "road" else [],
                    "use_types": use_types,
                    "suicide_lane_special_tag": "",
                    "egomotion_label_class_id": LABEL_CLASS_ID,
                },
                "version": SCHEMA_VERSION,
            }
        )
    return lanes, crosswalks


def _extract_linestring_elements(
    map_, clip_id: str
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {
        "lane_line": [],
        "road_boundary": [],
        "wait_line": [],
        "traffic_light": [],
        "traffic_sign": [],
    }

    for ls in map_.lineStringLayer:
        ls_type = _attr(ls, "type")
        ls_subtype = _attr(ls, "subtype")
        pts = _polyline(ls)
        if len(pts) < 2:
            continue

        if ls_type in {"line_thin", "line_thick"}:
            style = LANE_LINE_STYLE.get((ls_type, ls_subtype), "OTHER")
            color = _attr(ls, "color", "WHITE").upper()
            if color not in {"WHITE", "YELLOW", "RED"}:
                color = "WHITE"
            out["lane_line"].append(
                {
                    "key": _make_key(clip_id, ls.id),
                    "lane_line": {
                        "line_rail": pts,
                        "styles": [style],
                        "colors": [color],
                        "left_driving_direction": ["UNKNOWN"],
                        "right_driving_direction": ["UNKNOWN"],
                        "is_first_point_physical_end": "false",
                        "is_last_point_physical_end": "false",
                        "egomotion_label_class_id": LABEL_CLASS_ID,
                    },
                    "version": SCHEMA_VERSION,
                }
            )
        elif ls_type in ROAD_BOUNDARY_CATEGORY:
            out["road_boundary"].append(
                {
                    "key": _make_key(clip_id, ls.id),
                    "road_boundary": {
                        "category": ROAD_BOUNDARY_CATEGORY[ls_type],
                        "location": pts,
                        "left_driving_direction": ["UNKNOWN"],
                        "right_driving_direction": ["UNKNOWN"],
                        "is_first_point_physical_end": "false",
                        "is_last_point_physical_end": "false",
                        "egomotion_label_class_id": LABEL_CLASS_ID,
                    },
                    "version": SCHEMA_VERSION,
                }
            )
        elif ls_type == "stop_line":
            out["wait_line"].append(
                {
                    "key": _make_key(clip_id, ls.id),
                    "wait_line": {
                        "category": "STOP",
                        "location": pts,
                        "is_implicit": False,
                        "intersection_subtype": "NOT_APPLICABLE",
                        "egomotion_label_class_id": LABEL_CLASS_ID,
                    },
                    "version": SCHEMA_VERSION,
                }
            )
        elif ls_type == "traffic_light":
            row = _linestring_to_oriented_bbox(
                ls, pts, height=float(_attr(ls, "height", "0.5") or 0.5),
                default_thickness=0.15, category="traffic_light",
            )
            if row is not None:
                out["traffic_light"].append(
                    {
                        "key": _make_key(clip_id, ls.id),
                        "traffic_light": row,
                        "version": SCHEMA_VERSION,
                    }
                )
        elif ls_type == "traffic_sign":
            cat = ls_subtype.upper() if ls_subtype else "UNKNOWN"
            row = _linestring_to_oriented_bbox(
                ls, pts, height=0.8, default_thickness=0.01,
                category=f"TRAFFIC_SIGN_{cat}",
            )
            if row is not None:
                out["traffic_sign"].append(
                    {
                        "key": _make_key(clip_id, ls.id),
                        "traffic_sign": row,
                        "version": SCHEMA_VERSION,
                    }
                )

    return out


def _linestring_to_oriented_bbox(
    ls, pts: list[dict], *, height: float, default_thickness: float, category: str
) -> dict | None:
    """Build a ClipGT oriented bbox from a 2-point linestring representing the
    bottom edge of a panel/light. The panel extends ``height`` metres in +Z."""
    if len(pts) < 2:
        return None
    p0, p1 = pts[0], pts[-1]
    dx = p1["x"] - p0["x"]
    dy = p1["y"] - p0["y"]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    cx = 0.5 * (p0["x"] + p1["x"])
    cy = 0.5 * (p0["y"] + p1["y"])
    cz = 0.5 * (p0["z"] + p1["z"]) + 0.5 * height
    yaw = math.atan2(dy, dx)
    return {
        "center": {"x": cx, "y": cy, "z": cz},
        "dimensions": {"x": length, "y": default_thickness, "z": height},
        "orientation": _yaw_to_quat(yaw),
        "category": category,
        "egomotion_label_class_id": LABEL_CLASS_ID,
    }


def _extract_polygons(map_, clip_id: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {
        "crosswalk": [],
        "road_marking": [],
        "intersection_area": [],
        "road_island": [],
    }
    for poly in map_.polygonLayer:
        ptype = _attr(poly, "type")
        psubtype = _attr(poly, "subtype")
        pts = _polyline(poly)
        if len(pts) < 3:
            continue
        key = _make_key(clip_id, poly.id)
        if ptype == "crosswalk_polygon":
            out["crosswalk"].append(
                {
                    "key": key,
                    "crosswalk": {
                        "category": "PEDESTRIAN",
                        "location": pts,
                        "egomotion_label_class_id": LABEL_CLASS_ID,
                    },
                    "version": SCHEMA_VERSION,
                }
            )
        elif ptype == "intersection_area":
            out["intersection_area"].append(
                {
                    "key": key,
                    "intersection_area": {
                        "category": "FOUR_WAY",
                        "location": pts,
                        "is_complete": True,
                        "egomotion_label_class_id": LABEL_CLASS_ID,
                    },
                    "version": SCHEMA_VERSION,
                }
            )
        elif ptype == "area" and psubtype == "vegetation":
            out["road_island"].append(
                {
                    "key": key,
                    "road_island": {
                        "category": "vegetation",
                        "location": pts,
                        "egomotion_label_class_id": LABEL_CLASS_ID,
                    },
                    "version": SCHEMA_VERSION,
                }
            )
        elif (ptype, psubtype) in ROAD_MARKING_CATEGORY:
            out["road_marking"].append(
                {
                    "key": key,
                    "road_marking": {
                        "category": ROAD_MARKING_CATEGORY[(ptype, psubtype)],
                        "location": pts,
                        "egomotion_label_class_id": LABEL_CLASS_ID,
                    },
                    "version": SCHEMA_VERSION,
                }
            )
    return out


def _extract_poles(traffic_lights: list[dict], traffic_signs: list[dict], clip_id: str) -> list[dict]:
    """Synthesize pole rows under each traffic light/sign bbox."""
    rows: list[dict] = []
    for src, label in ((traffic_lights, "SIGN"), (traffic_signs, "SIGN")):
        for r in src:
            payload = r[next(iter(k for k in r if k not in {"key", "version"}))]
            c = payload["center"]
            half_h = payload["dimensions"]["z"] * 0.5
            base = {"x": c["x"], "y": c["y"], "z": c["z"] - half_h - 0.1}
            top = {"x": c["x"], "y": c["y"], "z": c["z"] - half_h}
            rows.append(
                {
                    "key": _make_key(clip_id, f"pole-{r['key']['map_id']}"),
                    "pole": {
                        "category": label,
                        "location": [base, top],
                        "egomotion_label_class_id": LABEL_CLASS_ID,
                    },
                    "version": SCHEMA_VERSION,
                }
            )
    return rows


# --- Public API ---


@dataclass
class ConversionStats:
    out_dir: Path
    counts: dict[str, int]

    def __str__(self) -> str:  # pragma: no cover - debug helper
        lines = [f"Wrote ClipGT bundle to {self.out_dir}"]
        for name, n in sorted(self.counts.items()):
            lines.append(f"  {name:<20s} {n}")
        return "\n".join(lines)


def convert(
    osm_path: str | Path,
    out_dir: str | Path,
    origin,
    *,
    clip_id: str | None = None,
) -> ConversionStats:
    """Parse ``osm_path`` and write a ClipGT parquet bundle to ``out_dir``.

    ``origin`` is a :class:`lanelet2.io.Origin` defining where local
    ``(x, y, z) = (0, 0, 0)`` lands in UTM. Output coordinates are metres in
    that local Cartesian frame (right-handed, Z-up — same as ClipGT).
    """
    import lanelet2  # imported lazily so the module is import-safe without ROS

    osm_path = Path(osm_path)
    out_dir = Path(out_dir)

    proj = lanelet2.projection.UtmProjector(origin, True, False)
    map_, _errors = lanelet2.io.loadRobust(str(osm_path), proj)

    clip_id = clip_id or f"lanelet2-{uuid.uuid4()}"

    lanes, crosswalk_lanelets = _extract_lanes(map_, clip_id)
    line_elems = _extract_linestring_elements(map_, clip_id)
    poly_elems = _extract_polygons(map_, clip_id)
    crosswalks = poly_elems["crosswalk"] + crosswalk_lanelets
    poles = _extract_poles(line_elems["traffic_light"], line_elems["traffic_sign"], clip_id)

    counts: dict[str, int] = {}
    counts["lane"] = _write(lanes, schemas.LANE, out_dir / "lane.parquet")
    counts["lane_line"] = _write(line_elems["lane_line"], schemas.LANE_LINE, out_dir / "lane_line.parquet")
    counts["road_boundary"] = _write(line_elems["road_boundary"], schemas.ROAD_BOUNDARY, out_dir / "road_boundary.parquet")
    counts["wait_line"] = _write(line_elems["wait_line"], schemas.WAIT_LINE, out_dir / "wait_line.parquet")
    counts["traffic_light"] = _write(line_elems["traffic_light"], schemas.TRAFFIC_LIGHT, out_dir / "traffic_light.parquet")
    counts["traffic_sign"] = _write(line_elems["traffic_sign"], schemas.TRAFFIC_SIGN, out_dir / "traffic_sign.parquet")
    counts["crosswalk"] = _write(crosswalks, schemas.CROSSWALK, out_dir / "crosswalk.parquet")
    counts["road_marking"] = _write(poly_elems["road_marking"], schemas.ROAD_MARKING, out_dir / "road_marking.parquet")
    counts["intersection_area"] = _write(poly_elems["intersection_area"], schemas.INTERSECTION_AREA, out_dir / "intersection_area.parquet")
    counts["road_island"] = _write(poly_elems["road_island"], schemas.ROAD_ISLAND, out_dir / "road_island.parquet")
    counts["pole"] = _write(poles, schemas.POLE, out_dir / "pole.parquet")
    counts["egomotion_estimate"] = _write([], schemas.EGOMOTION_ESTIMATE, out_dir / "egomotion_estimate.parquet")

    return ConversionStats(out_dir=out_dir, counts=counts)


