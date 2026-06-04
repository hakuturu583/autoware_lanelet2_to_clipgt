"""Convert an Autoware Lanelet2 .osm map into a cosmos-transfer2.5
world-scenario parquet bundle.

Reference: ``docs/world_scenario_parquet.md`` in nvidia-cosmos/cosmos-transfer2.5.

Output layout (``{clip_id}.{element_type}.parquet`` files in a flat dir):

    <out_dir>/
      {clip_id}.calibration_estimate.parquet  (required, placeholder rig)
      {clip_id}.egomotion_estimate.parquet    (required, single pose at origin)
      {clip_id}.obstacle.parquet              (required, empty)
      {clip_id}.lane.parquet
      {clip_id}.lane_line.parquet
      {clip_id}.road_boundary.parquet
      {clip_id}.crosswalk.parquet
      {clip_id}.road_marking.parquet
      {clip_id}.pole.parquet
      {clip_id}.wait_line.parquet
      {clip_id}.traffic_light.parquet
      {clip_id}.traffic_sign.parquet

Coordinates are the local UTM-offset frame defined by ``origin``: X=Forward,
Y=Left, Z=Up (right-handed). Since we don't have ego trajectory data, the
synthetic egomotion is a single pose at (0, 0, 0) facing +X at t=0.
"""

from __future__ import annotations

import json
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

# Style/category lookup tables (lanelet2 → cosmos).
LANE_LINE_STYLE = {
    ("line_thin", "solid"): "SOLID_SINGLE",
    ("line_thin", "dashed"): "DASHED_SINGLE",
    ("line_thin", ""): "SOLID_SINGLE",
    ("line_thick", "solid"): "SOLID_DOUBLE",
    ("line_thick", "dashed"): "DASHED_SINGLE",
    ("line_thick", ""): "SOLID_DOUBLE",
}

ROAD_BOUNDARY_CATEGORY = {
    "road_border": "road_boundary",
    "road_shoulder": "tall_curb",
    "fence": "fence",
    "guard_rail": "barrier",
    "wall": "wall",
}

ROAD_MARKING_CATEGORY = {
    ("hatched_road_markings", ""): "ROI_POLYGON_KEEP_CLEAR",
    ("no_parking_area", ""): "ROI_POLYGON_KEEP_CLEAR",
    ("no_stopping_area", ""): "ROI_POLYGON_KEEP_CLEAR",
    ("pedestrian_marking", ""): "ROI_POLYGON_ROAD_MARKING_PED_XING",
}


def _pt(p) -> dict:
    return {"x": float(p.x), "y": float(p.y), "z": float(p.z)}


def _polyline(ls) -> list[dict]:
    return [_pt(p) for p in ls]


def _polygon_from_lanelet(ll) -> list[dict]:
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
        table = pa.table({field.name: [] for field in schema}, schema=schema)
    else:
        table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)
    return len(rows)


# --- Per-element extractors (same lanelet2 mapping as the clipgt target,
#     but emitting cosmos-flavoured payloads) ---


def _extract_lanes(map_, clip_id: str) -> tuple[list[dict], list[dict]]:
    lanes: list[dict] = []
    crosswalks: list[dict] = []
    for ll in map_.laneletLayer:
        subtype = _attr(ll, "subtype")
        speed_limit_raw = _attr(ll, "speed_limit")
        try:
            speed_limit = float(speed_limit_raw) if speed_limit_raw else 0.0
        except ValueError:
            speed_limit = 0.0
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
                    },
                    "version": SCHEMA_VERSION,
                }
            )
            continue
        if subtype in {"walkway", "pedestrian_lane"}:
            continue
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
                    "speed_limit": speed_limit,
                    "vehicle_types": ["CAR"] if subtype == "road" else [],
                },
                "version": SCHEMA_VERSION,
            }
        )
    return lanes, crosswalks


def _extract_linestring_elements(map_, clip_id: str) -> dict[str, list[dict]]:
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
            style = LANE_LINE_STYLE.get((ls_type, ls_subtype), "SOLID_SINGLE")
            color = _attr(ls, "color", "WHITE").upper()
            if color not in {"WHITE", "YELLOW"}:
                color = "WHITE"
            out["lane_line"].append(
                {
                    "key": _make_key(clip_id, ls.id),
                    "lane_line": {
                        "line_rail": pts,
                        "styles": [style] * len(pts),
                        "colors": [color] * len(pts),
                        "left_driving_direction": "UNKNOWN",
                        "right_driving_direction": "UNKNOWN",
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
                    },
                    "version": SCHEMA_VERSION,
                }
            )
        elif ls_type == "traffic_light":
            bbox = _linestring_to_oriented_bbox(
                pts,
                height=float(_attr(ls, "height", "0.5") or 0.5),
                default_thickness=0.15,
                category="traffic_light",
            )
            if bbox is not None:
                out["traffic_light"].append(
                    {
                        "key": _make_key(clip_id, ls.id),
                        "traffic_light": bbox,
                        "version": SCHEMA_VERSION,
                    }
                )
        elif ls_type == "traffic_sign":
            cat = ls_subtype.upper() if ls_subtype else "UNKNOWN"
            bbox = _linestring_to_oriented_bbox(
                pts,
                height=0.8,
                default_thickness=0.01,
                category=f"TRAFFIC_SIGN_{cat}",
            )
            if bbox is not None:
                out["traffic_sign"].append(
                    {
                        "key": _make_key(clip_id, ls.id),
                        "traffic_sign": bbox,
                        "version": SCHEMA_VERSION,
                    }
                )

    return out


def _linestring_to_oriented_bbox(
    pts: list[dict], *, height: float, default_thickness: float, category: str
) -> dict | None:
    if len(pts) < 2:
        return None
    p0, p1 = pts[0], pts[-1]
    dx = p1["x"] - p0["x"]
    dy = p1["y"] - p0["y"]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    return {
        "center": {
            "x": 0.5 * (p0["x"] + p1["x"]),
            "y": 0.5 * (p0["y"] + p1["y"]),
            "z": 0.5 * (p0["z"] + p1["z"]) + 0.5 * height,
        },
        "dimensions": {"x": length, "y": default_thickness, "z": height},
        "orientation": _yaw_to_quat(math.atan2(dy, dx)),
        "category": category,
    }


def _extract_polygons(map_, clip_id: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"crosswalk": [], "road_marking": []}
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
                    "crosswalk": {"category": "PEDESTRIAN", "location": pts},
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
                    },
                    "version": SCHEMA_VERSION,
                }
            )
    return out


def _extract_poles(
    traffic_lights: list[dict], traffic_signs: list[dict], clip_id: str
) -> list[dict]:
    rows: list[dict] = []
    for src, label in ((traffic_lights, "LIGHT"), (traffic_signs, "SIGN")):
        for r in src:
            payload = r["traffic_light" if label == "LIGHT" else "traffic_sign"]
            c = payload["center"]
            half_h = payload["dimensions"]["z"] * 0.5
            base = {"x": c["x"], "y": c["y"], "z": c["z"] - half_h - 0.1}
            top = {"x": c["x"], "y": c["y"], "z": c["z"] - half_h}
            rows.append(
                {
                    "key": _make_key(clip_id, f"pole-{r['key']['map_id']}"),
                    "pole": {"category": label, "location": [base, top]},
                    "version": SCHEMA_VERSION,
                }
            )
    return rows


# --- Required temporal placeholders ---


def _placeholder_egomotion(clip_id: str) -> list[dict]:
    return [
        {
            "key": {"clip_id": clip_id, "timestamp_micros": 0},
            "egomotion_estimate": {
                "name": "lanelet2:placeholder",
                "location": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "version": SCHEMA_VERSION,
        }
    ]


def _placeholder_calibration(clip_id: str) -> list[dict]:
    rig = {
        "rig": {
            "sensors": [
                {
                    "name": "camera:front:wide:120fov",
                    "nominalSensor2Rig_FLU": {
                        "t": [1.7, 0.0, 1.4],
                        "roll-pitch-yaw": [0.0, 0.0, 0.0],
                    },
                    "properties": {
                        "Model": "ftheta",
                        "width": "3848",
                        "height": "2168",
                        "cx": "1924.0",
                        "cy": "1084.0",
                        "polynomial": "0 5.38e-4 0 0 0 0",
                        "polynomial-type": "pixeldistance-to-angle",
                        "linear-c": "1.0",
                        "linear-d": "0.0",
                        "linear-e": "0.0",
                    },
                }
            ]
        }
    }
    return [
        {
            "key": {"clip_id": clip_id, "timestamp_micros": -1},
            "calibration_estimate": {
                "name": "lanelet2:placeholder",
                "rig_json": json.dumps(rig),
            },
            "version": SCHEMA_VERSION,
        }
    ]


@dataclass
class ConversionStats:
    out_dir: Path
    clip_id: str
    counts: dict[str, int]

    def __str__(self) -> str:  # pragma: no cover - debug helper
        lines = [f"Wrote cosmos bundle to {self.out_dir} (clip_id={self.clip_id})"]
        for name, n in sorted(self.counts.items()):
            lines.append(f"  {name:<24s} {n}")
        return "\n".join(lines)


def convert(
    osm_path: str | Path,
    out_dir: str | Path,
    origin,
    *,
    clip_id: str | None = None,
) -> ConversionStats:
    """Parse ``osm_path`` and write a cosmos world-scenario parquet bundle.

    ``origin`` is a :class:`lanelet2.io.Origin`; the projector outputs metres
    in a local Cartesian frame matching cosmos's (X=Forward, Y=Left, Z=Up)
    convention. ``clip_id`` is used both inside row keys and as the parquet
    file-name prefix (cosmos requires the AV2 ``{clip_id}.<elem>.parquet``
    layout); a UUID is generated when omitted.
    """
    import lanelet2

    osm_path = Path(osm_path)
    out_dir = Path(out_dir)
    clip_id = clip_id or f"lanelet2-{uuid.uuid4()}"

    proj = lanelet2.projection.UtmProjector(origin, True, False)
    map_, _errors = lanelet2.io.loadRobust(str(osm_path), proj)

    lanes, crosswalk_lanelets = _extract_lanes(map_, clip_id)
    line_elems = _extract_linestring_elements(map_, clip_id)
    poly_elems = _extract_polygons(map_, clip_id)
    crosswalks = poly_elems["crosswalk"] + crosswalk_lanelets
    poles = _extract_poles(
        line_elems["traffic_light"], line_elems["traffic_sign"], clip_id
    )

    def path(elem: str) -> Path:
        return out_dir / f"{clip_id}.{elem}.parquet"

    counts: dict[str, int] = {}
    counts["lane"] = _write(lanes, schemas.LANE, path("lane"))
    counts["lane_line"] = _write(line_elems["lane_line"], schemas.LANE_LINE, path("lane_line"))
    counts["road_boundary"] = _write(line_elems["road_boundary"], schemas.ROAD_BOUNDARY, path("road_boundary"))
    counts["wait_line"] = _write(line_elems["wait_line"], schemas.WAIT_LINE, path("wait_line"))
    counts["traffic_light"] = _write(line_elems["traffic_light"], schemas.TRAFFIC_LIGHT, path("traffic_light"))
    counts["traffic_sign"] = _write(line_elems["traffic_sign"], schemas.TRAFFIC_SIGN, path("traffic_sign"))
    counts["crosswalk"] = _write(crosswalks, schemas.CROSSWALK, path("crosswalk"))
    counts["road_marking"] = _write(poly_elems["road_marking"], schemas.ROAD_MARKING, path("road_marking"))
    counts["pole"] = _write(poles, schemas.POLE, path("pole"))
    counts["obstacle"] = _write([], schemas.OBSTACLE, path("obstacle"))
    counts["egomotion_estimate"] = _write(_placeholder_egomotion(clip_id), schemas.EGOMOTION_ESTIMATE, path("egomotion_estimate"))
    counts["calibration_estimate"] = _write(_placeholder_calibration(clip_id), schemas.CALIBRATION_ESTIMATE, path("calibration_estimate"))

    return ConversionStats(out_dir=out_dir, clip_id=clip_id, counts=counts)
