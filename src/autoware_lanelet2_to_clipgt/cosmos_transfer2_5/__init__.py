"""Cosmos-Transfer2.5 world-scenario parquet target.

Reference: docs/world_scenario_parquet.md in nvidia-cosmos/cosmos-transfer2.5.
Same Apache Parquet container as ClipGT but with:

- AV2-style file naming: ``{clip_id}.{element_type}.parquet``
- Scalar (not list) ``left_driving_direction`` / ``right_driving_direction``
  on lane_line rows.
- ``lane.speed_limit`` typed as float, not string.
- ``obstacle`` / ``calibration_estimate`` / ``egomotion_estimate`` are required.
- Drops the ``egomotion_label_class_id`` field used by ClipGT.
- World frame anchored at the ego vehicle's starting pose
  (X=Forward, Y=Left, Z=Up).
"""

from .converter import ConversionStats, convert

__all__ = ["ConversionStats", "convert"]
