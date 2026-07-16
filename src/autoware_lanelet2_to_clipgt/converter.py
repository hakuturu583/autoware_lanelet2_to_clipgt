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
      association.parquet          (lane relations: NEXT/PREVIOUS/LEFT/RIGHT)
      clip.parquet                 (single-row clip metadata)

Coordinates are projected with ``UtmProjector(useOffset=True)`` so the scene
sits near the origin in metres (ClipGT convention: right-handed, Z-up, m).
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import schemas

LABEL_CLASS_ID = "lanelet2:autoware:v0"
MAP_VERSION = "1"
SCHEMA_VERSION = 1

# When set, every point emitted by _pt is first mapped from the projector's
# native frame (ECEF, when a scene alignment is requested) into the target
# scene-local frame. Configured by ``convert()`` and cleared afterwards.
_POINT_TRANSFORM: np.ndarray | None = None


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
    x, y, z = float(p.x), float(p.y), float(p.z)
    if _POINT_TRANSFORM is not None:
        v = _POINT_TRANSFORM @ np.array([x, y, z, 1.0])
        x, y, z = float(v[0]), float(v[1]), float(v[2])
    return {"x": x, "y": y, "z": z}


def _load_ecef_scene_transform(tileset_json_path: Path) -> np.ndarray:
    """Return ``T_enu_ecef`` (4x4) for mapping ECEF points into an ENU frame
    centered at the tileset's origin.

    ``tileset.json`` root ``transform`` is ``T_ecef_scene`` (column-major, per
    3D Tiles spec) — its translation is the scene origin in ECEF, and its
    rotation maps a scene-local basis (typically East-Down-North for
    Autoware/NuRec exports) into ECEF. Alpasim's runtime, however, drives its
    ego trajectory in a standard ENU frame (X=East, Y=North, Z=Up), so we
    return a transform that leaves the *translation* alone (same origin) but
    replaces the *rotation* with ENU-at-origin — regardless of the tileset's
    stored scene axes.
    """
    with open(tileset_json_path) as f:
        tileset = json.load(f)
    transform_flat = tileset["root"]["transform"]
    T_ecef_scene = np.array(transform_flat, dtype=np.float64).reshape(4, 4).T
    ecef_origin = T_ecef_scene[:3, 3]
    return _enu_from_ecef_at(ecef_origin)


def _enu_from_ecef_at(ecef_origin: np.ndarray) -> np.ndarray:
    """Build the 4x4 ``T_enu_ecef`` for the ENU frame centered at ``ecef_origin``."""
    x, y, z = ecef_origin
    # Geodetic lat/lon on the WGS-84 ellipsoid (Bowring's closed-form solution).
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    b = a * (1.0 - f)
    ep2 = (a * a - b * b) / (b * b)
    p = math.hypot(x, y)
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - e2 * a * math.cos(theta) ** 3,
    )
    lon = math.atan2(y, x)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    # Rows of R_enu_ecef are the ENU basis vectors expressed in ECEF.
    R_enu_ecef = np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R_enu_ecef
    T[:3, 3] = -R_enu_ecef @ ecef_origin
    return T


def _polyline(ls) -> list[dict]:
    return [_pt(p) for p in ls]


def _resample_polyline(pts: list[dict], n: int) -> list[dict]:
    """Resample a polyline to exactly ``n`` points via arc-length linear interp.

    Downstream consumers (e.g. trajdata ``populate_vector_map``) compute the
    per-lane centerline as ``(left + right) / 2`` and therefore require both
    rails to share the same length.
    """
    if len(pts) == n:
        return list(pts)
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    zs = [p["z"] for p in pts]
    seg_lens = [
        math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i], zs[i + 1] - zs[i])
        for i in range(len(pts) - 1)
    ]
    cum = [0.0]
    for s in seg_lens:
        cum.append(cum[-1] + s)
    total = cum[-1]
    if total <= 0.0:
        # Degenerate rail (all points coincident); replicate the first point.
        return [dict(pts[0]) for _ in range(n)]
    out: list[dict] = []
    for i in range(n):
        target = total * i / (n - 1)
        # Find segment containing `target`.
        j = 0
        while j + 1 < len(cum) - 1 and cum[j + 1] < target:
            j += 1
        seg = cum[j + 1] - cum[j]
        t = 0.0 if seg <= 0.0 else (target - cum[j]) / seg
        out.append(
            {
                "x": xs[j] + t * (xs[j + 1] - xs[j]),
                "y": ys[j] + t * (ys[j + 1] - ys[j]),
                "z": zs[j] + t * (zs[j + 1] - zs[j]),
            }
        )
    return out


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


def _make_association_key(clip_id: str, map_id: str, kind: str) -> dict:
    return {
        "clip_id": clip_id,
        "label_class_id": LABEL_CLASS_ID,
        "map_id": str(map_id),
        "kind": kind,
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
        # populate_vector_map computes centerline as (left+right)/2, requiring
        # matched lengths. Resample the shorter rail up to the longer one.
        n = max(len(left_pts), len(right_pts))
        left_pts = _resample_polyline(left_pts, n)
        right_pts = _resample_polyline(right_pts, n)
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
            # wait_line rows are keyed as ``{stop_line_id}-{lane_id}`` because
            # ``trajdata.mads_utils.populate_vector_map`` derives the associated
            # lane from that split. Emission is deferred to _extract_wait_lines
            # so we can iterate over stop_line ↔ lanelet pairs.
            pass
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


def _extract_wait_lines(
    map_,
    clip_id: str,
    stop_line_to_lanelets: dict[str, set[str]],
    emitted_lane_ids: set[str],
) -> list[dict]:
    """Emit one wait_line row per (stop_line, lanelet) pair, keyed ``{stop_line_id}-{lane_id}``."""
    rows: list[dict] = []
    for ls in map_.lineStringLayer:
        if _attr(ls, "type") != "stop_line":
            continue
        pts = _polyline(ls)
        if len(pts) < 2:
            continue
        lanelets = stop_line_to_lanelets.get(str(ls.id), set())
        # Only keep lanelets we actually emitted as lanes; skip the stop_line
        # entirely if none remain (downstream `.split("-")[1]` needs a lane id).
        for lane_id in sorted(l for l in lanelets if l in emitted_lane_ids):
            rows.append(
                {
                    "key": _make_key(clip_id, f"{ls.id}-{lane_id}"),
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
    return rows


def _parse_stop_line_lanelet_pairs(osm_path: Path) -> dict[str, set[str]]:
    """Return ``{stop_line_way_id: {lanelet_id, ...}}`` from raw OSM XML.

    Lanelet2's Python bindings surface ``RegulatoryElement.parameters`` as an
    empty map for un-typed regs, so we walk the XML directly:

        <relation type="lanelet"> --member role="regulatory_element"-->
        <relation type="regulatory_element"> --member role="ref_line"-->
        <way type="stop_line">
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(osm_path))
    root = tree.getroot()

    reg_to_ref_lines: dict[str, list[str]] = {}
    lanelet_to_regs: dict[str, list[str]] = {}
    stop_line_ways: set[str] = set()

    for rel in root.findall("relation"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in rel.findall("tag")}
        rel_type = tags.get("type")
        if rel_type == "regulatory_element":
            refs = [
                m.attrib["ref"]
                for m in rel.findall("member")
                if m.attrib.get("role") == "ref_line"
                and m.attrib.get("type") == "way"
            ]
            if refs:
                reg_to_ref_lines[rel.attrib["id"]] = refs
        elif rel_type == "lanelet":
            regs = [
                m.attrib["ref"]
                for m in rel.findall("member")
                if m.attrib.get("role") == "regulatory_element"
                and m.attrib.get("type") == "relation"
            ]
            if regs:
                lanelet_to_regs[rel.attrib["id"]] = regs

    for way in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        if tags.get("type") == "stop_line":
            stop_line_ways.add(way.attrib["id"])

    pairs: dict[str, set[str]] = {}
    for lid, regs in lanelet_to_regs.items():
        for rid in regs:
            for sid in reg_to_ref_lines.get(rid, []):
                if sid in stop_line_ways:
                    pairs.setdefault(sid, set()).add(lid)
    return pairs


def _extract_associations(
    map_, clip_id: str, emitted_lane_ids: set[str]
) -> list[dict]:
    """Emit lane-lane relations (NEXT/PREVIOUS/LEFT/RIGHT) via the lanelet2 routing graph.

    Consumers such as ``trajdata.dataset_specific.mads.mads_utils.populate_vector_map``
    treat ``Association.subjects`` as a single-element list, so we emit one row per
    (source_lane, kind) pair with ``objects`` listing every reachable target.
    """
    import lanelet2

    rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle,
    )
    graph = lanelet2.routing.RoutingGraph(map_, rules)

    rows: list[dict] = []

    def _emit(source_id: str, kind: str, objects: list[str]) -> None:
        objects = [o for o in objects if o in emitted_lane_ids]
        if not objects:
            return
        rows.append(
            {
                "key": _make_association_key(
                    clip_id, f"{source_id}-{kind}", kind
                ),
                "association": {
                    "subjects": [source_id],
                    "objects": objects,
                },
                "version": SCHEMA_VERSION,
            }
        )

    for ll in map_.laneletLayer:
        source_id = str(ll.id)
        if source_id not in emitted_lane_ids:
            continue
        _emit(source_id, "NEXT_LANE", [str(x.id) for x in graph.following(ll)])
        _emit(source_id, "PREVIOUS_LANE", [str(x.id) for x in graph.previous(ll)])
        left = graph.left(ll)
        if left is not None:
            _emit(source_id, "LEFT_LANE", [str(left.id)])
        right = graph.right(ll)
        if right is not None:
            _emit(source_id, "RIGHT_LANE", [str(right.id)])

    return rows


def _build_clip_row(clip_id: str) -> dict:
    """Single-row clip metadata; downstream (populate_vector_map) only reads key.clip_id."""
    return {
        "key": {
            "session_id": clip_id,
            "clip_id": clip_id,
            "time_range": {"start_micros": 0, "end_micros": 0},
        },
        "clip": {
            "ground_truth_calibration": "",
            "ground_truth_egomotion": "",
        },
        "version": SCHEMA_VERSION,
    }


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
    tileset_json: str | Path | None = None,
) -> ConversionStats:
    """Parse ``osm_path`` and write a ClipGT parquet bundle to ``out_dir``.

    ``origin`` is a :class:`lanelet2.io.Origin` defining where local
    ``(x, y, z) = (0, 0, 0)`` lands in UTM. Output coordinates are metres in
    that local Cartesian frame (right-handed, Z-up — same as ClipGT).

    ``tileset_json``, when set, aligns output to a scene frame defined by that
    tileset's root ECEF transform. Points are then projected via
    ``GeocentricProjector`` (ECEF) and transformed by ``inv(T_ecef_scene)``
    instead of using ``origin``. This is required for downstream consumers
    (e.g. alpasim's route sanity check) that read a trajectory living in the
    same scene frame as the tileset.
    """
    import lanelet2  # imported lazily so the module is import-safe without ROS

    osm_path = Path(osm_path)
    out_dir = Path(out_dir)

    global _POINT_TRANSFORM
    if tileset_json is not None:
        _POINT_TRANSFORM = _load_ecef_scene_transform(Path(tileset_json))
        proj = lanelet2.projection.GeocentricProjector()
    else:
        _POINT_TRANSFORM = None
        proj = lanelet2.projection.UtmProjector(origin, True, False)
    try:
        map_, _errors = lanelet2.io.loadRobust(str(osm_path), proj)
        return _convert_impl(map_, osm_path, out_dir, clip_id)
    finally:
        _POINT_TRANSFORM = None


def _convert_impl(map_, osm_path: Path, out_dir: Path, clip_id: str | None) -> ConversionStats:

    clip_id = clip_id or f"lanelet2-{uuid.uuid4()}"

    lanes, crosswalk_lanelets = _extract_lanes(map_, clip_id)
    line_elems = _extract_linestring_elements(map_, clip_id)
    poly_elems = _extract_polygons(map_, clip_id)
    crosswalks = poly_elems["crosswalk"] + crosswalk_lanelets
    poles = _extract_poles(line_elems["traffic_light"], line_elems["traffic_sign"], clip_id)
    emitted_lane_ids = {row["key"]["map_id"] for row in lanes}
    associations = _extract_associations(map_, clip_id, emitted_lane_ids)
    stop_line_to_lanelets = _parse_stop_line_lanelet_pairs(osm_path)
    line_elems["wait_line"] = _extract_wait_lines(
        map_, clip_id, stop_line_to_lanelets, emitted_lane_ids
    )

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
    counts["association"] = _write(associations, schemas.ASSOCIATION, out_dir / "association.parquet")
    counts["clip"] = _write([_build_clip_row(clip_id)], schemas.CLIP, out_dir / "clip.parquet")

    return ConversionStats(out_dir=out_dir, counts=counts)


