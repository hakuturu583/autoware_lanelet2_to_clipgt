"""Coordinate origin helpers (MGRS / lat-lon → lanelet2.io.Origin).

Ported from ``autoware_lanelet2_to_opendrive/projection.py`` so that map
configuration files share the same MGRS grid + offset convention.

The :class:`lanelet2.projection.UtmProjector` always uses UTM internally; the
origin defines where in UTM the local x/y/z = 0 lands.
"""

from __future__ import annotations

import logging
import re

import lanelet2
import mgrs

logger = logging.getLogger(__name__)


def _normalize_mgrs_grid(mgrs_grid: str) -> str:
    """Zero-pad a partial MGRS grid string (e.g. ``54SUE``) to a 10-digit suffix."""
    processed = mgrs_grid.strip()
    match = re.match(r"^(\d+[A-Z][A-Z][A-Z])(.*)$", processed)
    if not match:
        return processed
    grid_square = match.group(1)
    coords = match.group(2)
    if len(coords) == 0:
        return grid_square + "0000000000"
    if len(coords) < 10:
        if len(coords) % 2 == 1:
            coords += "0"
        coords = coords.ljust(10, "0")
    return grid_square + coords


def mgrs_to_origin(mgrs_grid: str) -> lanelet2.io.Origin:
    """``54SUE...`` → :class:`lanelet2.io.Origin`."""
    try:
        processed = _normalize_mgrs_grid(mgrs_grid)
        lat, lon = mgrs.MGRS().toLatLon(processed)
        logger.debug("origin from mgrs=%s → (%.8f, %.8f)", processed, lat, lon)
        return lanelet2.io.Origin(lat, lon)
    except Exception as exc:  # pragma: no cover - thin wrapper
        raise ValueError(f"Invalid MGRS grid string '{mgrs_grid}': {exc}") from exc


def mgrs_offset_to_origin(
    mgrs_grid: str, offset_x: float, offset_y: float, offset_z: float = 0.0
) -> lanelet2.io.Origin:
    """``54SUE`` + (easting, northing[, alt]) → :class:`lanelet2.io.Origin`.

    Builds the 10-digit MGRS coordinate inside ``mgrs_grid``'s square from
    integer-truncated metre offsets, then converts to lat/lon. Matches the
    behaviour used in autoware_lanelet2_to_opendrive.
    """
    match = re.match(r"^(\d+[A-Z][A-Z][A-Z])", mgrs_grid.strip())
    if not match:
        raise ValueError(f"Invalid MGRS format: {mgrs_grid}")
    grid_square = match.group(1)
    easting = int(offset_x)
    northing = int(offset_y)
    full = f"{grid_square}{easting:05d}{northing:05d}"
    try:
        lat, lon = mgrs.MGRS().toLatLon(full)
    except Exception as exc:  # pragma: no cover - thin wrapper
        raise ValueError(
            f"Invalid MGRS grid '{mgrs_grid}' with offset ({offset_x}, {offset_y}): {exc}"
        ) from exc
    logger.debug(
        "origin from mgrs=%s offset=(%.3f, %.3f, %.3f) → (%.8f, %.8f)",
        grid_square, offset_x, offset_y, offset_z, lat, lon,
    )
    return lanelet2.io.Origin(lat, lon, offset_z)


def latlon_to_origin(
    latitude: float, longitude: float, altitude: float = 0.0
) -> lanelet2.io.Origin:
    return lanelet2.io.Origin(latitude, longitude, altitude)


def origin_from_map_config(cfg) -> lanelet2.io.Origin:
    """Resolve a :class:`lanelet2.io.Origin` from a hydra map config node.

    Accepts the same three shapes as ``autoware_lanelet2_to_opendrive``:

    1. ``lat_lon: { latitude, longitude, altitude? }``
    2. ``mgrs_grid: <code>`` + optional ``offset: { x, y, z? }``
    3. ``mgrs_grid: <full code>`` alone

    ``cfg`` may be an OmegaConf node, a plain ``dict``, or any mapping-like.
    """
    get = _make_getter(cfg)
    lat_lon = get("lat_lon")
    if lat_lon is not None:
        ll_get = _make_getter(lat_lon)
        return latlon_to_origin(
            float(ll_get("latitude")),
            float(ll_get("longitude")),
            float(ll_get("altitude") or 0.0),
        )

    mgrs_grid = get("mgrs_grid")
    if mgrs_grid is None:
        raise ValueError("map config must specify either lat_lon or mgrs_grid")
    offset = get("offset")
    if offset is None:
        return mgrs_to_origin(str(mgrs_grid))
    off_get = _make_getter(offset)
    return mgrs_offset_to_origin(
        str(mgrs_grid),
        float(off_get("x")),
        float(off_get("y")),
        float(off_get("z") or 0.0),
    )


def _make_getter(cfg):
    """Return ``getter(key) -> value | None`` working for dict / OmegaConf / attr objects."""
    if hasattr(cfg, "get"):
        return lambda key: cfg.get(key)
    return lambda key: getattr(cfg, key, None)
