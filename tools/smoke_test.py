#!/usr/bin/env python3
"""Converts a map built on the spot, through the installed package's own CLI.

A release can be broken in ways the unit tests never see: a wheel that omits the
hydra config tree, a dependency that resolves to something without `lanelet2` in
it, an entry point that does not start. Those only show up in an environment that
has the built artifact and nothing else, which is what the release workflow points
this at.

The map is built here rather than checked in. The repository's fixture map is not
version-controlled (see .gitignore), and a smoke test that needs a file someone
has to fetch is a smoke test that gets skipped -- so this writes a handful of
lanelets with the same library the converter reads them with.

    python tools/smoke_test.py            # both targets, in a temp directory
    python tools/smoke_test.py --keep DIR # leave the bundles behind to look at
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# The map is written and read back through the same projector, so the origin only
# has to be somewhere UTM is defined -- these are the coordinates the converter's
# own tests use.
ORIGIN_LAT, ORIGIN_LON = 35.6895, 139.6917

# Four road lanelets in a chain plus a crosswalk across the end of it: enough for a
# lane row per lanelet, a NEXT_LANE association per join, and a crosswalk row.
N_ROAD = 4
LANE_WIDTH = 3.5
SEGMENT = 10.0


def build_map(path: Path) -> None:
    import lanelet2
    from lanelet2.core import LaneletMap, LineString3d, Point3d, getId

    def bound(points: list[Point3d], **attrs: str) -> LineString3d:
        ls = LineString3d(getId(), list(points))
        for key, value in attrs.items():
            ls.attributes[key] = value
        return ls

    def strip(coords: list[tuple[float, float]]) -> LineString3d:
        return bound([Point3d(getId(), x, y, 0.0) for x, y in coords])

    map_ = LaneletMap()

    # The segments share their boundary points rather than merely touching: the
    # routing graph decides succession by point identity, so building each segment
    # its own endpoints would produce lanelets with no relations between them.
    left_pts = [Point3d(getId(), 0.0, i * SEGMENT, 0.0) for i in range(N_ROAD + 1)]
    right_pts = [Point3d(getId(), LANE_WIDTH, i * SEGMENT, 0.0) for i in range(N_ROAD + 1)]

    for i in range(N_ROAD):
        left = bound(left_pts[i : i + 2], type="line_thin", subtype="dashed")
        right = bound(right_pts[i : i + 2], type="line_thin", subtype="solid")
        lanelet = lanelet2.core.Lanelet(getId(), left, right)
        lanelet.attributes["subtype"] = "road"
        lanelet.attributes["location"] = "urban"
        lanelet.attributes["speed_limit"] = "40"
        map_.add(lanelet)

    # A crosswalk lanelet lying across the far end of the road.
    y = N_ROAD * SEGMENT + 2.0
    crossing = lanelet2.core.Lanelet(
        getId(),
        strip([(-1.0, y), (-1.0, y + 3.0)]),
        strip([(LANE_WIDTH + 1.0, y), (LANE_WIDTH + 1.0, y + 3.0)]),
    )
    crossing.attributes["subtype"] = "crosswalk"
    map_.add(crossing)

    projector = lanelet2.projection.UtmProjector(
        lanelet2.io.Origin(ORIGIN_LAT, ORIGIN_LON, 0.0), True, False
    )
    lanelet2.io.write(str(path), map_, projector)


def convert(osm: Path, out_dir: Path, target: str) -> None:
    subprocess.run(
        [
            sys.executable, "-m", "autoware_lanelet2_to_clipgt",
            f"target={target}",
            "map=example",
            "++map.mgrs_grid=null",
            f"++map.lat_lon={{latitude:{ORIGIN_LAT},longitude:{ORIGIN_LON},altitude:0.0}}",
            f"input_map_path={osm}",
            f"output_dir={out_dir}",
            "clip_id=smoke",
            f"hydra.run.dir={out_dir}/.hydra",
        ],
        check=True,
    )


def rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def check_clipgt(out_dir: Path) -> None:
    # populate_vector_map opens these by name, so their absence is the failure that
    # matters most -- an empty file is fine, a missing one is not.
    expected = {
        "association", "clip", "crosswalk", "egomotion_estimate", "intersection_area",
        "lane", "lane_line", "pole", "road_boundary", "road_island", "road_marking",
        "traffic_light", "traffic_sign", "wait_line",
    }
    found = {p.stem for p in out_dir.glob("*.parquet")}
    assert found == expected, f"clipgt bundle: missing {expected - found}, extra {found - expected}"

    lanes = rows(out_dir / "lane.parquet")
    assert len(lanes) == N_ROAD, f"expected {N_ROAD} lanes, got {len(lanes)}"
    assert all(
        len(r["lane"]["left_rail"]) == len(r["lane"]["right_rail"]) for r in lanes
    ), "left and right rails must have matching lengths"

    assert len(rows(out_dir / "crosswalk.parquet")) == 1, "expected the crosswalk lanelet"
    assert len(rows(out_dir / "clip.parquet")) == 1, "clip.parquet is a single row"

    # The chain gives N-1 joins, and each join is one NEXT and one PREVIOUS row.
    kinds = [r["key"]["kind"] for r in rows(out_dir / "association.parquet")]
    assert kinds.count("NEXT_LANE") == N_ROAD - 1, f"NEXT_LANE rows: {kinds.count('NEXT_LANE')}"
    assert kinds.count("PREVIOUS_LANE") == N_ROAD - 1, f"PREVIOUS_LANE rows: {kinds.count('PREVIOUS_LANE')}"


def check_cosmos(out_dir: Path) -> None:
    # cosmos wants the AV2 `{clip_id}.<element>.parquet` layout, which is the part
    # of this target a packaging mistake would quietly drop.
    expected = {
        "calibration_estimate", "crosswalk", "egomotion_estimate", "lane", "lane_line",
        "obstacle", "pole", "road_boundary", "road_marking", "traffic_light",
        "traffic_sign", "wait_line",
    }
    found = {p.name.split(".")[1] for p in out_dir.glob("smoke.*.parquet")}
    assert found == expected, f"cosmos bundle: missing {expected - found}, extra {found - expected}"
    assert len(rows(out_dir / "smoke.lane.parquet")) == N_ROAD


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=Path, help="write into this directory and keep it")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = args.keep or Path(tmp)
        work.mkdir(parents=True, exist_ok=True)

        osm = work / "smoke.osm"
        build_map(osm)
        print(f"built {osm} ({osm.stat().st_size} bytes)")

        convert(osm, work / "clipgt", "clipgt")
        check_clipgt(work / "clipgt")
        print("clipgt bundle ok")

        convert(osm, work / "cosmos", "cosmos_transfer2_5")
        check_cosmos(work / "cosmos")
        print("cosmos_transfer2_5 bundle ok")

    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
