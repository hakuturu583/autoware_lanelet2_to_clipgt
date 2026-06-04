"""Parquet schemas for ClipGT map elements.

ClipGT row shape (per `docs/clipgt-format.md`):

    key:        struct (clip_id, label_class_id, map_id, map_id_version)
    <element>:  struct (payload, named after the file)
    version:    uint64

Geometry primitives:
    point  : struct{x, y, z: double}
    poly*  : list<point>
    quat   : struct{x, y, z, w: double}
    dim    : struct{x, y, z: double}
"""

from __future__ import annotations

import pyarrow as pa


POINT = pa.struct(
    [
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("z", pa.float64()),
    ]
)
QUAT = pa.struct(
    [
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("z", pa.float64()),
        ("w", pa.float64()),
    ]
)
DIM = POINT  # same struct shape
POLYLINE = pa.list_(POINT)


def _map_key() -> pa.StructType:
    return pa.struct(
        [
            ("clip_id", pa.string()),
            ("label_class_id", pa.string()),
            ("map_id", pa.string()),
            ("map_id_version", pa.string()),
        ]
    )


def _temporal_key() -> pa.StructType:
    return pa.struct(
        [
            ("clip_id", pa.string()),
            ("label_class_id", pa.string()),
            ("timestamp_micros", pa.int64()),
            ("label_id", pa.string()),
        ]
    )


def _payload(name: str, fields: list[tuple[str, pa.DataType]]) -> pa.Schema:
    return pa.schema(
        [
            ("key", _map_key()),
            (name, pa.struct(fields)),
            ("version", pa.uint64()),
        ]
    )


# --- Map (static) element schemas ---

LANE_LINE = _payload(
    "lane_line",
    [
        ("line_rail", POLYLINE),
        ("styles", pa.list_(pa.string())),
        ("colors", pa.list_(pa.string())),
        ("left_driving_direction", pa.list_(pa.string())),
        ("right_driving_direction", pa.list_(pa.string())),
        ("is_first_point_physical_end", pa.string()),
        ("is_last_point_physical_end", pa.string()),
        ("egomotion_label_class_id", pa.string()),
    ],
)

ROAD_BOUNDARY = _payload(
    "road_boundary",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
        ("left_driving_direction", pa.list_(pa.string())),
        ("right_driving_direction", pa.list_(pa.string())),
        ("is_first_point_physical_end", pa.string()),
        ("is_last_point_physical_end", pa.string()),
        ("egomotion_label_class_id", pa.string()),
    ],
)

LANE = _payload(
    "lane",
    [
        ("left_rail", POLYLINE),
        ("right_rail", POLYLINE),
        ("left_edge_styles", pa.list_(pa.string())),
        ("right_edge_styles", pa.list_(pa.string())),
        ("left_edge_colors", pa.list_(pa.string())),
        ("right_edge_colors", pa.list_(pa.string())),
        ("lane_direction", pa.string()),
        ("map_end", pa.string()),
        ("speed_limit", pa.string()),
        ("vehicle_types", pa.list_(pa.string())),
        ("use_types", pa.list_(pa.string())),
        ("suicide_lane_special_tag", pa.string()),
        ("egomotion_label_class_id", pa.string()),
    ],
)

WAIT_LINE = _payload(
    "wait_line",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
        ("is_implicit", pa.bool_()),
        ("intersection_subtype", pa.string()),
        ("egomotion_label_class_id", pa.string()),
    ],
)

CROSSWALK = _payload(
    "crosswalk",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
        ("egomotion_label_class_id", pa.string()),
    ],
)

ROAD_MARKING = _payload(
    "road_marking",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
        ("egomotion_label_class_id", pa.string()),
    ],
)

INTERSECTION_AREA = _payload(
    "intersection_area",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
        ("is_complete", pa.bool_()),
        ("egomotion_label_class_id", pa.string()),
    ],
)

ROAD_ISLAND = _payload(
    "road_island",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
        ("egomotion_label_class_id", pa.string()),
    ],
)

POLE = _payload(
    "pole",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
        ("egomotion_label_class_id", pa.string()),
    ],
)

TRAFFIC_LIGHT = _payload(
    "traffic_light",
    [
        ("center", POINT),
        ("dimensions", DIM),
        ("orientation", QUAT),
        ("category", pa.string()),
        ("egomotion_label_class_id", pa.string()),
    ],
)

TRAFFIC_SIGN = _payload(
    "traffic_sign",
    [
        ("center", POINT),
        ("dimensions", DIM),
        ("orientation", QUAT),
        ("category", pa.string()),
        ("egomotion_label_class_id", pa.string()),
    ],
)

EGOMOTION_ESTIMATE = pa.schema(
    [
        ("key", _temporal_key()),
        (
            "egomotion_estimate",
            pa.struct(
                [
                    ("name", pa.string()),
                    ("location", POINT),
                    ("orientation", QUAT),
                ]
            ),
        ),
        ("version", pa.uint64()),
    ]
)
