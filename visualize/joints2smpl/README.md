# joints2smpl
fit SMPL model using 3D joints

## Prerequisites
We have tested the code on Ubuntu 18.04/20.04 with CUDA 10.2/11.3

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do is to use the [anaconda](https://www.anaconda.com/).

You can create an anaconda environment called `fit3d` using
```
conda env create -f environment.yaml
conda activate fit3d
```

## Download SMPL models
Download [SMPL Female and Male](https://smpl.is.tue.mpg.de/) and [SMPL Netural](https://smplify.is.tue.mpg.de/), and rename the files and extract them to `<current directory>/smpl_models/smpl/`, eventually, the `<current directory>/smpl_models` folder should have the following structure:
   ```
   smpl_models
    └-- smpl
    	└-- SMPL_FEMALE.pkl
		└-- SMPL_MALE.pkl
		└-- SMPL_NEUTRAL.pkl
   ```   

## Demo
### Demo for sequences
python fit_seq.py --files test_motion2.npy

The results will locate in ./demo/demo_results/

## Citation
If you find this project useful for your research, please consider citing:
```
@article{zuo2021sparsefusion,
  title={Sparsefusion: Dynamic human avatar modeling from sparse rgbd images},
  author={Zuo, Xinxin and Wang, Sen and Zheng, Jiangbin and Yu, Weiwei and Gong, Minglun and Yang, Ruigang and Cheng, Li},
  journal={IEEE Transactions on Multimedia},
  volume={23},
  pages={1617--1629},
  year={2021}
}
```

## References
We indicate if a function or script is borrowed externally inside each file. Here are some great resources we 
benefit:

- Shape/Pose prior and some functions are borrowed from [VIBE](https://github.com/mkocabas/VIBE).
- SMPL models and layer is from [SMPL-X model](https://github.com/vchoutas/smplx).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
