# FreeReg AI Agent Instructions

## Project Overview
FreeReg is a **zero-shot** image-to-point cloud registration system (ICLR 2024) that leverages pretrained diffusion models (ControlNet/Stable Diffusion) and monocular depth estimators (ZoeDepth) WITHOUT task-specific training. The key insight: unify modality between RGB images and point clouds through intermediate "diffusion features" for robust cross-modality correspondence.

## Architecture & Pipeline

### Four-Stage Processing Pipeline
1. **ZoeDepth Generation** (`pipeline/gen_zoe.py`): Monocular depth estimation from RGB images using ZoeDepth models
2. **Feature Extraction** (`pipeline/gen_feat.py`): Extract two types of features:
   - **Diffusion Features (DF)**: Semantic features from ControlNet at layers [0,4,6], 16x downsampled
   - **Geometric Features (GF)**: FCGF features extracted on sparse voxelized depth maps (voxel_size=0.025m)
3. **Matching** (`pipeline/gen_match.py`): Nearest-neighbor matching + weighted feature fusion, then pose estimation
4. **Evaluation** (`pipeline/gen_eval.py`): Compute Inlier Ratio, Registration Recall metrics

### Key Components
- `tools/controlnet/`: ControlNet v1.1 depth-conditioned diffusion feature extractor
- `tools/zoe/`: ZoeDepth wrapper for metric depth estimation (indoor: ZoeD_M12_N, outdoor: ZoeD_M12_NK)
- `tools/fcgf/`: Fully Convolutional Geometric Features using MinkowskiEngine sparse convolutions
- `tools/pose/`: Registration solvers (PnP, RANSAC, Essential matrix)
- `dataops/metas.py`: Dataset metadata generation from file structure

## Critical Setup Requirements

### Model Checkpoints (Required)
Download pretrained models BEFORE running:
- **ControlNet** → `tools/controlnet/models/`: `v1-5-pruned.ckpt`, `control_v11f1p_sd15_depth_ft.pth` (finetuned version recommended)
- **ZoeDepth** → `tools/zoe/models/`: `ZoeD_M12_N.pt` (indoor), `ZoeD_M12_NK.pt` (outdoor)  
- **FCGF** → `tools/fcgf/models/`: `fcgf_indoor.pth`, `fcgf_outdoor.pth`

### Environment Setup
```bash
conda env create -f environment.yaml
conda activate freereg
# Fetch MiDaS for ZoeDepth (manual step required)
python -c "import torch; torch.hub.help('intel-isl/MiDaS', 'DPT_BEiT_L_384', force_reload=True)"
# Install MinkowskiEngine for FCGF
conda install openblas-devel -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git && cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

**Docker Alternative**: Use provided Dockerfile with CUDA 11.3, builds MinkowskiEngine automatically.

## Running FreeReg

### Quick Demo
```bash
python demo.py  # Registers data/demo/source_rgb.png to data/demo/target_pc.ply
```

### Benchmark Evaluation
```bash
python run.py --dataset [3dmatch|scannet|kitti] --type [d|g|dg]
# --type dg: Fused diffusion+geometric features (best performance)
# --type d:  Diffusion features only with PnP solver
# --type g:  Geometric features only with Kabsch solver
# --update_zoe/df/gf/pca: Force regenerate cached features
```

### Registration Type Selection (`--type`)
- `dg`: Uses BOTH `rgb_df+rgb_gf` and `dpt_df+dpt_gf`, Kabsch/RANSAC solver
- `d`: Diffusion features only (`rgb_df`, `dpt_df`), PnP solver
- `g`: Geometric features only (`rgb_gf`, `dpt_gf`), Kabsch solver

## Project-Specific Conventions

### Configuration System
- Uses OmegaConf YAML configs in `config/` (3dmatch.yaml, scannet.yaml, kitti.yaml)
- Key config sections: `meta` (data paths, intrinsics), `zoe`, `feat.cn` (ControlNet), `feat.fcgf`, `reg` (solvers)
- Access: `cfg = gen_config('config/3dmatch.yaml'); cfg.meta.rgb_size` → `[640, 480]`

### Feature Naming Convention
Format: `{modality}_{feature_type}` where:
- `modality`: `rgb` (RGB image), `dpt` (depth map)
- `feature_type`: `df` (diffusion features), `gf` (geometric features)
- Example: `rgb_df` = diffusion features from RGB, `dpt_gf` = geometric features from depth

### Metadata Structure
`gen_meta(cfg).run()` returns nested dict:
```python
metas = {
  'scene_name': {
    'frames': {frame_id: {'rgb_fn', 'dpt_fn', 'zoe_fn', 'to_fn', ...}},
    'pairs': [{'q_id', 'd_id', 'overlap', 'gt', 'to_fn'}, ...]
  }
}
```

### Feature Caching Strategy
- **ZoeDepth outputs**: Cached as `.zoe.npy` files (e.g., `frame_000.zoe.npy`)
- **Extracted features**: Cached in `tale_features/{dataset}/feat/*.feat.pth` as torch tensors
- **Matches**: Cached in `tale_features/{dataset}/match/*.trans.npz`
- Use `--update_zoe/df/gf/pca` flags to regenerate cached data

### Depth Handling
- **Scale**: Indoor datasets use 1000.0 (mm→m), follow `cfg.meta.dpt_scale`
- **Projection**: `dpt_3d_convert()` handles pc→depth and depth→pc with intrinsic matrices
- **Densification**: Optional sparse-to-dense completion via `cfg.meta.densify` flag

## Common Workflows

### Adding New Dataset
1. Create YAML config in `config/` with intrinsics, paths, ZoeDepth model type
2. Organize data: `data/{dataset}/scene/query/*.color.png`, `*.depth.png`, `*_pose.txt`
3. Update `dataops/metas.py` if file naming differs from 3DMatch format
4. Run pipeline: `python run.py --dataset {dataset} --type dg`

### Modifying Feature Fusion
Edit `pipeline/gen_match.py`:
- `weight=[w1,w2,...]`: Per-feature-type weights (normalized internally)
- `merge_conflict_check()`: Enforces RGB-D alignment when fusing rgb_* and dpt_* features

### Debugging Registration Failures
- Check `result.txt` for per-pair metrics (Inlier Ratio, Registration Recall)
- Visualize: Use `utils/drawer.py`'s `visualizor()` class for correspondence plots
- Common issues: Wrong intrinsic matrix in config, depth scale mismatch, missing model checkpoints

## Performance Notes
- **GPU Memory**: Full pipeline (~8GB VRAM). Set `load_model=False` in extractors if OOM
- **Speed**: ZoeDepth is bottleneck (~2s/frame). Use `--update_zoe` only when regenerating depth
- **Alternative**: Use [Free-FreeReg branch](https://github.com/WHU-USI3DV/FreeReg/tree/FFreeReg) for faster inference without diffusion features

## Anti-Patterns
- ❌ Don't mix indoor/outdoor models (ZoeD_M12_N for Kitti will fail)
- ❌ Don't use `processor_type='se3'` when source has no depth (use `'pnp'`)
- ❌ Don't modify cached features without setting `--update_*` flags
- ❌ Don't run `jupyter notebook` - this is a script-based project with no notebooks
