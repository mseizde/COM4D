# Ball Event Physics Dataset

This package is intended for single-ball bounce and rolling-occlusion synthetic
physics data. The preprocessor is already available as
`preprocess_ball_event_outputs.py`; it delegates to the generalized synthetic
preprocessor in `../two_ball_test/preprocess_physics_outputs.py`.

## Metadata Contract

Use `physics_metadata.json` with an `objects` dictionary. Dynamic objects are
turned into COM4D training targets and `parts`; static objects may be rendered
in RGB but are ignored by preprocessing.

Single-ball bounce:

```json
{
  "scenario": "bounce",
  "fps": 30,
  "objects": {
    "ball_0": {"type": "sphere", "dynamic": true, "radius": 0.25, "mass": 1.0}
  },
  "frames": [
    {
      "frame": 0,
      "ball_0": {
        "position": [0.0, 0.0, 1.4],
        "quaternion": [0.0, 0.0, 0.0, 1.0],
        "linear_velocity": [0.2, 0.0, -2.5],
        "angular_velocity": [0.0, 0.0, 0.0]
      },
      "contacts": {"ground": false}
    }
  ]
}
```

Rolling occlusion:

```json
{
  "scenario": "rolling_occlusion",
  "fps": 30,
  "objects": {
    "ball_0": {"type": "sphere", "dynamic": true, "radius": 0.25},
    "occluder_box": {
      "type": "box",
      "dynamic": false,
      "size": [0.7, 0.2, 0.9],
      "position": [0.0, -0.2, 0.45],
      "quaternion": [0.0, 0.0, 0.0, 1.0]
    }
  },
  "frames": []
}
```

## Preprocess

```bash
cd /data/mseizde/com4d/COM4D

micromamba run -n com4d python datasets/synthetic/ball_event_test/preprocess_ball_event_outputs.py \
  --input-root /data/mseizde/com4d/datasets/processed/physics/ball_event_raw \
  --output-root /data/mseizde/com4d/datasets/processed/physics/ball_event \
  --json-output /data/mseizde/com4d/COM4D/dataset_json/ball_event.json \
  --num-points 8192 \
  --include-parts \
  --overwrite
```

The generated `points.npy` contains:

```python
{
  "object": combined_dynamic_surface,
  "parts": [surface_for_each_dynamic_object]
}
```
