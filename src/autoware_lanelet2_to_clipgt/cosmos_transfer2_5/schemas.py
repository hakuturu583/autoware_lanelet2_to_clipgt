"""Parquet schemas for the cosmos-transfer2.5 world-scenario format.

Differences from the ClipGT schemas:

* ``lane_line.left_driving_direction`` / ``right_driving_direction`` are scalar
  strings (not lists).
* ``lane.speed_limit`` is a float (not a string), and the lane row drops
  ``map_end`` / ``use_types`` / ``suicide_lane_special_tag`` / the
  ``egomotion_label_class_id`` tag.
* Map-element rows drop ``egomotion_label_class_id``.
* Temporal keys carry only ``timestamp_micros`` (+ ``label_class_id`` /
  ``label_id`` where applicable).
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


def _obstacle_key() -> pa.StructType:
    return pa.struct(
        [
            ("clip_id", pa.string()),
            ("timestamp_micros", pa.int64()),
            ("label_class_id", pa.string()),
        ]
    )


def _ego_key() -> pa.StructType:
    return pa.struct(
        [
            ("clip_id", pa.string()),
            ("timestamp_micros", pa.int64()),
        ]
    )


def _calib_key() -> pa.StructType:
    return pa.struct(
        [
            ("clip_id", pa.string()),
            ("timestamp_micros", pa.int64()),
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


# --- Static (map) element schemas ---

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
        ("speed_limit", pa.float64()),
        ("vehicle_types", pa.list_(pa.string())),
    ],
)

LANE_LINE = _payload(
    "lane_line",
    [
        ("line_rail", POLYLINE),
        ("styles", pa.list_(pa.string())),
        ("colors", pa.list_(pa.string())),
        ("left_driving_direction", pa.string()),
        ("right_driving_direction", pa.string()),
    ],
)

ROAD_BOUNDARY = _payload(
    "road_boundary",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
    ],
)

CROSSWALK = _payload(
    "crosswalk",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
    ],
)

ROAD_MARKING = _payload(
    "road_marking",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
    ],
)

POLE = _payload(
    "pole",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
    ],
)

WAIT_LINE = _payload(
    "wait_line",
    [
        ("category", pa.string()),
        ("location", POLYLINE),
        ("is_implicit", pa.bool_()),
        ("intersection_subtype", pa.string()),
    ],
)

TRAFFIC_LIGHT = _payload(
    "traffic_light",
    [
        ("center", POINT),
        ("dimensions", POINT),
        ("orientation", QUAT),
        ("category", pa.string()),
    ],
)

TRAFFIC_SIGN = _payload(
    "traffic_sign",
    [
        ("center", POINT),
        ("dimensions", POINT),
        ("orientation", QUAT),
        ("category", pa.string()),
    ],
)


# --- Temporal element schemas ---

OBSTACLE = pa.schema(
    [
        ("key", _obstacle_key()),
        (
            "obstacle",
            pa.struct(
                [
                    ("trackline_id", pa.string()),
                    ("center", POINT),
                    ("size", POINT),
                    ("orientation", QUAT),
                    ("category", pa.string()),
                ]
            ),
        ),
        ("version", pa.uint64()),
    ]
)

EGOMOTION_ESTIMATE = pa.schema(
    [
        ("key", _ego_key()),
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

CALIBRATION_ESTIMATE = pa.schema(
    [
        ("key", _calib_key()),
        (
            "calibration_estimate",
            pa.struct(
                [
                    ("name", pa.string()),
                    ("rig_json", pa.string()),
                ]
            ),
        ),
        ("version", pa.uint64()),
    ]
)
