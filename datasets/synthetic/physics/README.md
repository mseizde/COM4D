# Synthetic Physics Dataset

Use `generate_physics_dataset.py` as the main entrypoint. It runs PyBullet metadata generation, Blender RGB rendering, and COM4D preprocessing for the supported physics scenarios.

Supported scenario labels include `two_ball_collision`, `ball_drop`, `rolling_occluder`, and `wall_impact`. The `two_ball_collision` name is kept as a scenario label because it describes that specific case.

## Generate Dataset

```bash
cd /data/mseizde/com4d/COM4D

micromamba run -n com4d python datasets/synthetic/physics/generate_physics_dataset.py \
  --num-samples 100 \
  --num-frames 48 \
  --workers 4 \
  --preprocess-workers 4 \
  --device GPU \
  --gpu-ids 0,1,2,3 \
  --num-points 8192 \
  --json-output /data/mseizde/com4d/COM4D/dataset_json/physics.json \
  --processed-root /data/mseizde/com4d/datasets/processed/physics/processed \
  --overwrite
```

`generate_physics_dataset.py` passes `--include-parts` to `preprocess_physics_outputs.py` automatically. The preprocessing step writes `points.npy` with composed `object` samples and explicit per-object `parts`, which is what the spatio-temporal physics batches expect.

## Main Files

- `generate_physics_dataset.py`: end-to-end dataset generator.
- `run_physics_pipeline.py`: generate one raw sample folder.
- `generate_physics_metadata.py`: PyBullet trajectory and collision metadata.
- `simulate_physics_scenario.py`: scenario definitions and simulation.
- `render_physics_outputs.py`: Blender rendering for a raw sample.
- `preprocess_physics_outputs.py`: convert raw samples into COM4D training JSON and `points.npy`.
- `physics_scene.blend`: Blender scene template.

## Output Used By Training

The default training config reads:

```text
/data/mseizde/com4d/COM4D/dataset_json/physics.json
```
