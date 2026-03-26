# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LaneATT is an attention-guided lane detection model from the CVPR 2021 paper "Keep your Eyes on the Lane." It uses anchor-based proposals with an attention mechanism for real-time lane detection. Supports three datasets: TuSimple, CULane, and LLAMAS. Three backbone options: ResNet-18/34/122.

## Build & Run Commands

### Docker Container

All the below commands shoulb be executed in the container of 'laneatt-laneatt-1'.

### Setup
```bash
pip install -r requirements.txt
cd lib/nms && python setup.py install && cd -
```

### Training
```bash
python main.py train --exp_name <name> --cfg cfgs/laneatt_tusimple_resnet34.yml
# Resume interrupted training:
python main.py train --exp_name <name> --cfg cfgs/laneatt_tusimple_resnet34.yml --resume
```

### Evaluation
```bash
python main.py test --exp_name <name>                    # test last checkpoint
python main.py test --exp_name <name> --epoch 50         # test specific epoch
python main.py test --exp_name <name> --view all         # visualize predictions
python main.py test --exp_name <name> --view mistakes    # visualize only errors
```

### Demo / Inference
```bash
python demo.py --video path/to/video.mp4 --output output.mp4
```

### ONNX / TensorRT Export
```bash
python laneatt_to_onnx.py
python onnx_to_tensorrt.py   # run inside TensorRT Docker container
```

## Architecture

### Data Flow
```
Image (3×360×640)
  → ResNet backbone (feature extraction)
  → 1×1 conv (reduce to 64 channels)
  → Anchor feature pooling (sample features along anchor curves)
  → Anchor-to-anchor attention
  → Classification head (lane vs background, focal loss)
  → Regression head (72 x-offsets per anchor, smooth L1 loss)
  → CUDA NMS (filter overlapping proposals)
  → Lane decoding (proposals → Lane spline objects)
```

### Key Modules

- **`main.py`** — Entry point. Parses args, creates `Experiment` + `Config` + `Runner`, dispatches to train/test.
- **`lib/runner.py`** (`Runner`) — Training loop and evaluation logic. Manages dataloaders, optimizer, checkpointing.
- **`lib/models/laneatt.py`** (`LaneATT`) — Core model. Anchor generation, feature pooling, attention, classification/regression heads, NMS, loss computation, and proposal decoding all live here.
- **`lib/config.py`** (`Config`) — Loads YAML configs. Factory methods: `get_model()`, `get_dataset()`, `get_optimizer()`, `get_lr_scheduler()`.
- **`lib/experiment.py`** (`Experiment`) — Manages experiment directories under `experiments/<name>/`. Stores checkpoints (`models/model_XXXX.pt`), logs, configs, git state.
- **`lib/datasets/lane_dataset.py`** (`LaneDataset`) — Wraps dataset loaders with augmentation (imgaug) and converts annotations to model input format.
- **`lib/datasets/{tusimple,culane,llamas}.py`** — Dataset-specific annotation loaders implementing `LaneDatasetLoader`.
- **`lib/nms/`** — CUDA C++ extension for non-maximum suppression. Must be compiled before use.
- **`lib/focal_loss.py`** — Focal loss for handling class imbalance (few positive anchors vs many background).
- **`lib/lane.py`** (`Lane`) — Lane representation using `InterpolatedUnivariateSpline`; callable with y-coords to get x-coords.

### Anchor System
Anchors are generated from three edge groups (left: 72 origins × 6 angles, right: 72 × 6, bottom: 128 × 15), then filtered to top-K by frequency using precomputed masks (`data/*_anchors_freq.pt`). Each anchor is parameterized as `[score0, score1, start_y, start_x, length, x0..x71]`.

### Configuration
YAML configs in `cfgs/` control model parameters, training hyperparameters, dataset selection, augmentation, and loss settings. Config naming convention: `laneatt_{dataset}_{backbone}.yml`. Datasets expect data under `datasets/` directory.

### Experiment Structure
Training creates `experiments/<exp_name>/` containing:
- `models/model_XXXX.pt` — checkpoints per epoch
- `cfg.yaml` — config snapshot
- `code_state/` — git hash and diff at training start
- `train.log`, `results/` — logs and evaluation outputs

## Important Notes

- **CUDA required** — The NMS extension only supports GPU. CPU mode is explicitly unsupported.
- **`main.py` line 59** — After training, evaluation currently runs with `on_val=True` (evaluates on val set even in test mode). This appears intentional for the current development branch.
- The loss is weighted: `10 × focal_loss + regression_loss`.
- NMS thresholds differ between train (15px) and test (45px).
