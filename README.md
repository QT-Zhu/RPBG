# RPBG: Robust Point-based Graphics

## Environment
```
conda create -n RPBG python=3.9 -y && conda activate RPBG
sh scripts/configure_env.sh
```

## Custom Data
We provide the scripts to process custom data without camera calibration and triangulation. The typical data structure is as follows.
```
|-- custom_root_path
    |-- camera.xml # agisoft format of camera intrinsics & extrinsics
    |-- scene-sparse.yaml # configuration file for sparse triangulation (SfM)
    |-- scene-dense.yaml # configuration file for dense triangulation (MVS)
    |-- images # raw images (not to be used in training)
    |-- sfm
        |-- sparse_pcd.ply # sparsely triangulated points
        |-- undis 
            |-- images # undistorted images (to be used in training)
    |-- mvs
        |-- dense_pcd.ply # densely triangulated points
```

### Data Preparation
First configure the path of your data in the script in `triangulation/prepare_inputs.sh`, as well as other settings if wanted, e.g., GPU indexes and distortion models, and execute it.
```
sh triangulation/prepare_inputs.sh
```
Then please fill the relevant information in `configs/paths.yaml` and create a custom config file similar to `configs/custom/sample.yaml`, and adopting the default set of hyper-parameters will just work fine.


### Training
To start training, please follow the scripts in `scripts`.
We give an example as follows.
```
sh scripts/train.sh configs/custom/sample.yaml
```

## Citation
```
@article{zhu2024rpbg,
  title={RPBG: Towards Robust Neural Point-based Graphics in the Wild},
  author={Zhu, Qingtian and Wei, Zizhuang and Zheng, Zhongtian and Zhan, Yifan and Yao, Zhuyu and Zhang, Jiawang and Wu, Kejian and Zheng, Yinqiang},
  journal={arXiv preprint arXiv:2405.05663},
  year={2024}
}
```

## Acknowledgements
We would like to thank the maintainers of the following repositories.
- [PCPR](https://github.com/wuminye/PCPR): for point cloud rasterization (z-buffering) by pure CUDA
- [NPBG](https://github.com/alievk/npbg): for the general point-based neural rendering pipeline & data convention
- [READ](https://github.com/JOP-Lee/READ): for more features implemented
- [Open3D](https://github.com/isl-org/Open3D): for visualization of point clouds on headless servers
- [COLMAP](https://colmap.github.io): for camera calibration and sparse triangulation
- [AA-RMVSNet](https://github.com/QT-Zhu/AA-RMVSNet): for dense triangulation