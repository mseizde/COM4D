# Two-Ball Physics Dataset

Use `generate_physics_dataset.py` as the main entrypoint. It runs PyBullet
metadata generation, Blender RGB rendering, and COM4D preprocessing.

The preprocessing step writes `points.npy` files with:

- `object`: composed two-ball surface samples
- `parts`: explicit per-ball surface samples

The `parts` key is required for physics batches using COM4D spatio-temporal
mixing, because training needs a frame-major `[num_frames * num_objects]`
instance layout.

## Generate Dataset

```bash
cd /data/mseizde/com4d/COM4D

micromamba run -n com4d python datasets/synthetic/two_ball_test/generate_physics_dataset.py \
  --num-samples 100 \
  --num-frames 48 \
  --workers 4 \
  --preprocess-workers 4 \
  --device GPU \
  --gpu-ids 0,1,2,3 \
  --num-points 8192 \
  --json-output /data/mseizde/com4d/COM4D/dataset_json/physics.json \
  --processed-root /data/mseizde/com4d/datasets/processed/physics/two_ball \
  --overwrite
```

`generate_physics_dataset.py` now passes `--include-parts` to
`preprocess_physics_outputs.py` automatically.

## Repair Existing Processed Dataset

If a processed two-ball dataset already has frame GLBs but `points.npy` has an
empty `parts` list, repair it without regenerating or rerendering:

```bash
cd /data/mseizde/com4d/COM4D

micromamba run -n com4d python datasets/synthetic/two_ball_test/repair_two_ball_parts.py \
  --processed-root /mnt/mocap_b/work/com4d/datasets/processed/physics/two_ball \
  --num-points 8192 \
  --workers 32 \
  --overwrite
```

This preserves the existing composed `object` samples and RGB renders, then
adds `parts=[ball_0, ball_1]` from the existing per-frame GLBs.

## Main Files

- `generate_physics_dataset.py`: recommended end-to-end dataset generator.
- `run_physics_pipeline.py`: generate one raw sample folder.
- `generate_physics_metadata.py`: PyBullet trajectory and collision metadata.
- `render_physics_outputs.py`: Blender rendering for a raw sample.
- `preprocess_physics_outputs.py`: convert raw samples into COM4D training JSON and `points.npy`.
- `repair_two_ball_parts.py`: add explicit `parts` to an already processed dataset from existing GLBs.
- `two_ball_scene.blend`: Blender scene template.

## Output Used By Training

The default training config reads:

```text
/data/mseizde/com4d/COM4D/dataset_json/physics.json
```

and expects each referenced `points.npy` to contain two explicit `parts`.
