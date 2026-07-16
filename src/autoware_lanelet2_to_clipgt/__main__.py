"""Hydra-driven CLI: convert a Lanelet2 .osm map to a ClipGT parquet bundle.

Examples
--------
    # Use the bundled odaiba map config (origin = 54SUE + offset(92008.5, 45335.1))
    uv run python -m autoware_lanelet2_to_clipgt \\
        input_map_path=tests/data/odaiba.osm \\
        output_dir=out/

    # Pick a different map config
    uv run python -m autoware_lanelet2_to_clipgt map=example \\
        input_map_path=path/to/map.osm output_dir=out/

    # Override the origin inline
    uv run python -m autoware_lanelet2_to_clipgt \\
        input_map_path=tests/data/odaiba.osm output_dir=out/ \\
        map.mgrs_grid=54SUE map.offset.x=92008.5 map.offset.y=45335.1
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from . import converter as clipgt_converter
from .cosmos_transfer2_5 import converter as cosmos_converter
from .projection import origin_from_map_config

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    if cfg.get("input_map_path") is None:
        print("error: input_map_path is required", file=sys.stderr)
        return 1
    if cfg.get("output_dir") is None:
        print("error: output_dir is required", file=sys.stderr)
        return 1

    map_cfg = cfg.get("map")
    if map_cfg is None:
        print("error: no map config selected (pass map=<name>)", file=sys.stderr)
        return 1

    origin = origin_from_map_config(map_cfg)
    osm_path = Path(to_absolute_path(str(cfg.input_map_path)))
    out_dir = Path(to_absolute_path(str(cfg.output_dir)))

    logger.info("map config:\n%s", OmegaConf.to_yaml(map_cfg))
    pos = origin.position
    logger.info("origin: lat=%.8f lon=%.8f alt=%.3f", pos.lat, pos.lon, pos.alt)

    target_cfg = cfg.get("target") or {}
    fmt = (target_cfg.get("format") if hasattr(target_cfg, "get") else getattr(target_cfg, "format", None)) or "clipgt"
    logger.info("output format: %s", fmt)

    tileset_json = cfg.get("tileset_json")
    if tileset_json is not None:
        tileset_json = Path(to_absolute_path(str(tileset_json)))

    if fmt == "clipgt":
        stats = clipgt_converter.convert(
            osm_path,
            out_dir,
            origin,
            clip_id=cfg.get("clip_id"),
            tileset_json=tileset_json,
        )
    elif fmt == "cosmos_transfer2_5":
        stats = cosmos_converter.convert(osm_path, out_dir, origin, clip_id=cfg.get("clip_id"))
    else:
        print(f"error: unknown target.format '{fmt}'", file=sys.stderr)
        return 1
    print(stats)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
