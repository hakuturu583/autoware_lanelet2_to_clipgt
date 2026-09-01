# autoware_lanelet2_to_clipgt

Converts an [Autoware](https://autoware.org/) [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2)
`.osm` map into the parquet bundles that simulation and world-model tooling reads:

- **`clipgt`** — the ClipGT HDMap layout consumed by alpasim, through
  `trajdata.dataset_specific.mads.mads_utils.populate_vector_map`.
- **`cosmos_transfer2_5`** — an [NVIDIA cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
  world-scenario scene, in the AV2-style `{clip_id}.<element>.parquet` layout.

Lanelet2 comes from [`simple-lanelet2`](https://github.com/hakuturu583/simple_lanelet2), a
drop-in reimplementation published as a plain wheel — so this installs from an index
with no ROS, no Boost and no C++ toolchain.

```bash
pip install autoware-lanelet2-to-clipgt
```

## Converting a map

The CLI is [hydra](https://hydra.cc)-driven. A map needs an **origin**, which is where
local `(x, y, z) = (0, 0, 0)` lands: output coordinates are metres in that local
Cartesian frame, right-handed and Z-up.

```bash
# the bundled odaiba origin (MGRS 54SUE + offset)
python -m autoware_lanelet2_to_clipgt \
    map=odaiba input_map_path=map.osm output_dir=out/

# a cosmos-transfer2.5 scene instead
python -m autoware_lanelet2_to_clipgt \
    target=cosmos_transfer2_5 map=odaiba input_map_path=map.osm output_dir=out/

# an origin given inline
python -m autoware_lanelet2_to_clipgt \
    input_map_path=map.osm output_dir=out/ \
    map.mgrs_grid=54SUE map.offset.x=92008.5 map.offset.y=45335.1
```

An origin is written one of three ways in a map config under `conf/map/`
(`conf/map/example.yaml` has all three):

```yaml
mgrs_grid: 54SUE815501            # an MGRS code, on its own

mgrs_grid: 54SUE                  # a grid square plus metres within it
offset: { x: 81655.73, y: 50137.43, z: 42.49998 }

lat_lon: { latitude: 35.6895, longitude: 139.6917, altitude: 42.5 }
```

`tileset_json=path/to/tileset.json` aligns the output to the scene frame of a 3D Tiles
tileset instead: points are projected to ECEF and transformed by the inverse of the
tileset's root transform, then re-expressed as ENU at the origin. That is what lets the
map share a frame with a trajectory embedded in the same USDZ.

### Looking at the result

```bash
python -m autoware_lanelet2_to_clipgt.visualize out/                    # clipgt
python -m autoware_lanelet2_to_clipgt.cosmos_transfer2_5.visualize out/ # cosmos
```

Both draw a top-down preview from the parquet files. The clipgt one also accepts a
`.usdz`, which it extracts to a temp directory first.

## As a library

```python
import lanelet2
from autoware_lanelet2_to_clipgt import convert, origin_from_map_config

stats = convert("map.osm", "out/", lanelet2.io.Origin(35.6895, 139.6917, 0.0))
print(stats)
```

`origin_from_map_config` takes the same three shapes as the YAML above, from an
OmegaConf node or a plain dict.

## Development

```bash
uv sync
uv run pytest                  # most of the suite needs a fixture map (see below)
uv run python tools/smoke_test.py
```

The tests want a Lanelet2 map at `tests/data/odaiba.osm`, which is not
version-controlled; without it the cases that need one skip themselves. The smoke test
needs no fixture — it builds a small map with Lanelet2's own API, converts it through
the CLI and checks the bundles, which is also what the release workflow runs against a
freshly installed wheel.

Releases are cut by labelling a pull request `release:major`, `release:minor` or
`release:patch`; merging it bumps the version, tags it and publishes to PyPI. See
[`.github/workflows/release.yml`](.github/workflows/release.yml).

## License

[Apache License 2.0](LICENSE).
