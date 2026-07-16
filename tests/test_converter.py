"""Regression tests for the Lanelet2 → ClipGT converter.

The consumer we care about is alpasim's ``artifact._load_clipgt_map``, which
calls ``trajdata.dataset_specific.mads.mads_utils.populate_vector_map``.
Rather than depend on the heavy trajdata stack in CI, these tests assert the
invariants that loader relies on:

* All required parquet files exist under the output directory.
* ``clip.parquet`` has a single row exposing ``key.clip_id``.
* ``association.parquet`` uses the (clip_id, label_class_id, map_id, kind) key
  and lists lanes only from the set we actually emitted.
* Every ``wait_line`` row is keyed ``{stop_line_id}-{lane_id}`` (populate_
  vector_map derives the lane via ``x.split("-")[1]``).
* Each lane's ``left_rail`` and ``right_rail`` polylines have identical
  lengths (populate_vector_map computes ``(left+right)/2`` per point).
"""
from __future__ import annotations

import math
from pathlib import Path

import lanelet2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from autoware_lanelet2_to_clipgt import converter, schemas

FIXTURE_MAP = Path(__file__).parent / "data" / "odaiba.osm"
ODAIBA_ORIGIN = lanelet2.io.Origin(35.6895, 139.6917, 0)

requires_fixture = pytest.mark.skipif(
    not FIXTURE_MAP.exists(),
    reason=f"fixture map not present at {FIXTURE_MAP} (see .gitignore — download separately)",
)


@pytest.fixture(scope="session")
def clipgt_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not FIXTURE_MAP.exists():
        pytest.skip(f"fixture map not present at {FIXTURE_MAP}")
    out_dir = tmp_path_factory.mktemp("clipgt")
    converter.convert(
        FIXTURE_MAP, out_dir, ODAIBA_ORIGIN, clip_id="test-clip"
    )
    return out_dir


def _write_tileset_json(tmp_path: Path, ecef_translation: tuple[float, float, float]) -> Path:
    """Write a minimal 3D Tiles tileset.json with an identity-rotation root transform."""
    import json

    tx, ty, tz = ecef_translation
    # Column-major 4x4: identity rotation + translation in the last row.
    transform = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        tx, ty, tz, 1.0,
    ]
    path = tmp_path / "tileset.json"
    path.write_text(json.dumps({"root": {"transform": transform}}))
    return path


# --- _resample_polyline ---------------------------------------------------


def test_resample_polyline_returns_requested_length():
    pts = [{"x": float(i), "y": 0.0, "z": 0.0} for i in range(5)]
    out = converter._resample_polyline(pts, 10)
    assert len(out) == 10


def test_resample_polyline_preserves_endpoints():
    pts = [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 10.0, "y": 0.0, "z": 0.0}]
    out = converter._resample_polyline(pts, 6)
    assert out[0] == pts[0]
    assert out[-1] == pts[-1]


def test_resample_polyline_preserves_content_when_lengths_match():
    pts = [{"x": float(i), "y": 0.0, "z": 0.0} for i in range(4)]
    out = converter._resample_polyline(pts, 4)
    assert out == pts
    # Returns a shallow copy so callers can't inadvertently mutate the input.
    assert out is not pts


def test_resample_polyline_uses_3d_arc_length():
    # Vertical rail: without z-aware arc length we'd treat these as coincident
    # (dx=dy=0) and fall into the degenerate path.
    pts = [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.0, "y": 0.0, "z": 10.0}]
    out = converter._resample_polyline(pts, 5)
    assert len(out) == 5
    assert out[0]["z"] == pytest.approx(0.0)
    assert out[-1]["z"] == pytest.approx(10.0)
    assert out[2]["z"] == pytest.approx(5.0)


def test_resample_polyline_handles_degenerate_zero_length_input():
    pts = [{"x": 1.0, "y": 2.0, "z": 3.0}] * 3
    out = converter._resample_polyline(pts, 5)
    assert len(out) == 5
    assert all(p == pts[0] for p in out)


# --- _parse_stop_line_lanelet_pairs ---------------------------------------


@requires_fixture
def test_parse_stop_line_lanelet_pairs_links_ways_to_lanelets():
    pairs = converter._parse_stop_line_lanelet_pairs(FIXTURE_MAP)
    assert pairs, "expected at least one stop_line ↔ lanelet mapping in odaiba"
    for stop_line_id, lanelets in pairs.items():
        assert stop_line_id.isdigit()
        assert lanelets, "every pair must have at least one lanelet"
        assert all(l.isdigit() for l in lanelets)


# --- End-to-end convert() -------------------------------------------------


EXPECTED_FILES = {
    "lane",
    "lane_line",
    "road_boundary",
    "wait_line",
    "traffic_light",
    "traffic_sign",
    "crosswalk",
    "road_marking",
    "intersection_area",
    "road_island",
    "pole",
    "egomotion_estimate",
    "association",
    "clip",
}


def test_convert_writes_every_expected_parquet(clipgt_bundle: Path):
    files = {p.stem for p in clipgt_bundle.glob("*.parquet")}
    assert EXPECTED_FILES <= files, f"missing: {EXPECTED_FILES - files}"


def test_clip_parquet_is_single_row_with_clip_id(clipgt_bundle: Path):
    df = pd.read_parquet(clipgt_bundle / "clip.parquet")
    assert len(df) == 1
    assert df.iloc[0]["key"]["clip_id"] == "test-clip"


def test_association_parquet_key_and_kinds(clipgt_bundle: Path):
    df = pd.read_parquet(clipgt_bundle / "association.parquet")
    assert len(df) > 0

    key_fields = set(df.iloc[0]["key"].keys())
    assert key_fields == {"clip_id", "label_class_id", "map_id", "kind"}

    kinds = {row["kind"] for row in df["key"]}
    # NEXT/PREVIOUS should always appear on a driveable map; LEFT/RIGHT are
    # optional but the two topological kinds are load-bearing.
    assert {"NEXT_LANE", "PREVIOUS_LANE"} <= kinds


def test_association_only_references_emitted_lanes(clipgt_bundle: Path):
    lane_ids = {
        row["map_id"]
        for row in pd.read_parquet(clipgt_bundle / "lane.parquet")["key"]
    }
    df = pd.read_parquet(clipgt_bundle / "association.parquet")
    for assoc in df["association"]:
        for lane_id in list(assoc["subjects"]) + list(assoc["objects"]):
            assert lane_id in lane_ids, f"association references unemitted lane {lane_id}"


def test_wait_line_key_encodes_lane_id(clipgt_bundle: Path):
    lane_ids = {
        row["map_id"]
        for row in pd.read_parquet(clipgt_bundle / "lane.parquet")["key"]
    }
    df = pd.read_parquet(clipgt_bundle / "wait_line.parquet")
    assert len(df) > 0
    for key in df["key"]:
        stop_line_id, _, lane_id = key["map_id"].partition("-")
        assert stop_line_id and lane_id, f"wait_line map_id missing '-': {key['map_id']}"
        assert lane_id in lane_ids


def test_lane_rails_have_matching_lengths(clipgt_bundle: Path):
    df = pd.read_parquet(clipgt_bundle / "lane.parquet")
    for lane in df["lane"]:
        assert len(lane["left_rail"]) == len(lane["right_rail"])


# --- Schema conformance ---------------------------------------------------


# --- tileset_json (scene-frame alignment) --------------------------------


def test_load_ecef_scene_transform_puts_tileset_origin_at_enu_zero(
    tmp_path: Path,
):
    # Use a real ECEF point (Odaiba, ~lat 35.63, lon 139.78) so the geodetic
    # inversion has meaningful lat/lon to work with.
    ecef = (-3963058.88, 3351535.88, 3694602.27)
    tileset = _write_tileset_json(tmp_path, ecef)
    T_enu_ecef = converter._load_ecef_scene_transform(tileset)
    origin_enu = T_enu_ecef @ np.array([*ecef, 1.0])
    assert origin_enu[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


def test_load_ecef_scene_transform_maps_ecef_radial_to_enu_up(
    tmp_path: Path,
):
    # A point 100 m further from Earth's center along the same radial as the
    # origin should map to almost (0, 0, +100) — Z is Up in the returned ENU
    # frame. (Radial is not exactly Up over a spheroid, but at 35°N the
    # deflection is well under a metre for a 100 m displacement.)
    ecef_origin = np.array([-3963058.88, 3351535.88, 3694602.27])
    radial = ecef_origin / np.linalg.norm(ecef_origin)
    ecef_above = ecef_origin + 100.0 * radial

    tileset = _write_tileset_json(tmp_path, tuple(ecef_origin))
    T = converter._load_ecef_scene_transform(tileset)
    p = T @ np.array([*ecef_above, 1.0])
    assert p[2] == pytest.approx(100.0, abs=2.0)   # z is Up
    assert abs(p[0]) < 2.0                          # x (East)  ~ 0
    assert abs(p[1]) < 2.0                          # y (North) ~ 0


@requires_fixture
def test_tileset_alignment_moves_output_close_to_origin(
    tmp_path_factory: pytest.TempPathFactory,
):
    # ECEF position for lat=35.6895, lon=139.6917 (the odaiba UtmProjector origin).
    # A tileset centered here should make the converter emit points near (0,0,0).
    # Compute ECEF via lanelet2's GeocentricProjector to avoid an extra dep.
    ecef_pt = lanelet2.projection.GeocentricProjector().forward(
        lanelet2.core.GPSPoint(35.6895, 139.6917, 0.0)
    )
    tileset = _write_tileset_json(
        tmp_path_factory.mktemp("tileset"), (ecef_pt.x, ecef_pt.y, ecef_pt.z)
    )

    out_dir = tmp_path_factory.mktemp("aligned")
    converter.convert(
        FIXTURE_MAP, out_dir, ODAIBA_ORIGIN, clip_id="aligned", tileset_json=tileset,
    )

    lane = pd.read_parquet(out_dir / "lane.parquet").iloc[0]["lane"]
    pt = lane["left_rail"][0]
    # Odaiba lanes span roughly ±5 km from that origin — well below the 100 km
    # bound we'd hit if inv(T_ecef_scene) weren't applied.
    assert abs(pt["x"]) < 100_000
    assert abs(pt["z"]) < 100_000


def test_convert_clears_point_transform_state(tmp_path_factory: pytest.TempPathFactory):
    # convert(tileset_json=...) sets a module-global; make sure a subsequent
    # call without tileset_json sees a clean slate.
    if not FIXTURE_MAP.exists():
        pytest.skip("fixture map missing")
    tileset = _write_tileset_json(tmp_path_factory.mktemp("t"), (0.0, 0.0, 0.0))
    converter.convert(
        FIXTURE_MAP, tmp_path_factory.mktemp("a"), ODAIBA_ORIGIN,
        clip_id="a", tileset_json=tileset,
    )
    assert converter._POINT_TRANSFORM is None
    converter.convert(
        FIXTURE_MAP, tmp_path_factory.mktemp("b"), ODAIBA_ORIGIN, clip_id="b",
    )
    assert converter._POINT_TRANSFORM is None


# --- Schema conformance ---------------------------------------------------


@pytest.mark.parametrize(
    "filename,schema",
    [
        ("association.parquet", schemas.ASSOCIATION),
        ("clip.parquet", schemas.CLIP),
        ("wait_line.parquet", schemas.WAIT_LINE),
        ("lane.parquet", schemas.LANE),
    ],
)
def test_parquet_matches_expected_schema(clipgt_bundle: Path, filename, schema):
    written = pq.read_table(clipgt_bundle / filename).schema
    for field in schema:
        assert field.name in written.names
        assert written.field(field.name).type == field.type
