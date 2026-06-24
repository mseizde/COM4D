# COM4D Scripts Map

This directory contains small entrypoints around inference preprocessing,
evaluation, and debugging. Prefer the scripts listed here over ad-hoc direct
calls to lower-level helpers.

## Physics Evaluation

- `eval/run_physics_comparison.py`: run a base-vs-physics comparison on one synthetic physics case.
- `eval/run_physics_statistical_eval.py`: run multi-sample, multi-model physics/reconstruction evaluation.
- `eval/prepare_physics_inference_input.py`: convert a generated physics sample into an inference input folder.
- `eval/evaluate_physics.py`: compute physics metrics for generated outputs.
- `eval/evaluate_reconstruction.py`: reconstruction metrics.
- `eval/render_prediction_gifs.py`: backfill prediction GIFs; with GT camera/alignment metadata it also writes three-panel diagnostic, aligned-orbit, and faded-prediction + GT-wireframe artifacts.
- `eval/run_benchmark.py`: combine reconstruction and physics evaluation.

## Room/Layout Priors

These are experimental inference-time geometry/layout utilities:

- `inference/add_room_constraints.py`: attach room/floor/wall constraint inputs.
- `inference/optimize_layout_prior.py`: optimize object layout against priors.
- `inference/layout_prior_debug.py`: inspect/debug layout-prior inputs and outputs.
- `inference/eval_3dfront_geometry_priors.py`: evaluate room geometry prior signals.

Keep these separate from training. They are for inference-time layout priors and
debugging, not for the new physics spatio-temporal training path.

## External Geometry Preprocessors

These scripts bridge COM4D inputs to separate environments/repos:

- `inference/run_geometrycrafter_preprocess.py`
  - run with `micromamba run -n geometrycrafter`
  - writes GeometryCrafter depth/point/normal priors
- `inference/run_vggt_preprocess.py`
  - run with `micromamba run -n vggt`
  - writes VGGT camera/depth/track priors
- `inference/run_hyworld_preprocess.py`
  - run with `micromamba run -n hyworld2`
  - writes HY-World/WorldMirror priors
- `inference/export_vggt_previews.py`: preview/export helper for VGGT outputs.

These files intentionally stay under `scripts/inference/` because they prepare
inference inputs or priors. They should not be mixed into the training dataset
generation path.

## General Image/Mask Utilities

- `inference/extract_frames.py`
- `inference/pad_to_square.py`
- `inference/resize_images.py`
- `inference/center_by_mask_union.py`
- `inference/apply_mask_combinations.py`
- `inference/gsam.py`

These are generic preprocessing helpers used by inference workflows.
