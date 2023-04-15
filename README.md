# AF-SfMLearner

This is the official PyTorch implementation for training and testing depth estimation models using the method described in

> **Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue**
>
> [Shuwei Shao](https://scholar.google.com.hk/citations?hl=zh-CN&user=ecZHSVQAAAAJ), Zhongcai Pei, [Weihai Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=5PoZrcYAAAAJ), [Wentao Zhu](https://scholar.google.com.hk/citations?hl=zh-CN&user=2hjYfqIAAAAJ), Xingming Wu, Dianmin Sun and [Baochang Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=ImJz6MsAAAAJ)
>
> [accepted by Medical Image Analysis (arXiv pdf)](https://arxiv.org/pdf/2112.08122.pdf)

and 

> **Self-Supervised Learning for Monocular Depth Estimation on Minimally Invasive Surgery Scenes**
>
> [Shuwei Shao](https://scholar.google.com.hk/citations?hl=zh-CN&user=ecZHSVQAAAAJ), Zhongcai Pei, [Weihai Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=5PoZrcYAAAAJ), [Baochang Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=ImJz6MsAAAAJ), Xingming Wu, Dianmin Sun and [David Doermann](https://scholar.google.com.hk/citations?hl=zh-CN&user=RoGOW9AAAAAJ)
>
> [ICRA 2021 (pdf)](https://ieeexplore.ieee.org/abstract/document/9561508).

#### Overview

<p align="center">
<img src='imgs/overview.png' width=800/> 
</p>

## ✏️ 📄 Citation

If you find our work useful in your research please consider citing our paper:

```
@article{shao2022self,
  title={Self-Supervised monocular depth and ego-Motion estimation in endoscopy: Appearance flow to the rescue},
  author={Shao, Shuwei and Pei, Zhongcai and Chen, Weihai and Zhu, Wentao and Wu, Xingming and Sun, Dianmin and Zhang, Baochang},
  journal={Medical image analysis},
  volume={77},
  pages={102338},
  year={2022},
  publisher={Elsevier}
}
```
```
@inproceedings{shao2021self,
  title={Self-Supervised Learning for Monocular Depth Estimation on Minimally Invasive Surgery Scenes},
  author={Shao, Shuwei and Pei, Zhongcai and Chen, Weihai and Zhang, Baochang and Wu, Xingming and Sun, Dianmin and Doermann, David},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={7159--7165},
  year={2021},
  organization={IEEE}
}
```



## ⚙️ Setup

We ran our experiments with PyTorch 1.2.0, torchvision 0.4.0, CUDA 10.2, Python 3.7.3 and Ubuntu 18.04.



## 🖼️ Prediction for a single image or a folder of images

You can predict scaled disparity for a single image or a folder of images with:

```shell
CUDA_VISIBLE_DEVICES=0 python test_simple.py --model_path <your_model_path> --image_path <your_image_or_folder_path>
```



## 💾 Datasets

You can download the [Endovis or SCARED dataset](https://endovissub2019-scared.grand-challenge.org) by signing the challenge rules and emailing them to max.allan@intusurg.com, the [EndoSLAM dataset](https://data.mendeley.com/datasets/cd2rtzm23r/1), the [SERV-CT dataset](https://www.ucl.ac.uk/interventional-surgical-sciences/serv-ct), and the [Hamlyn dataset](http://hamlyn.doc.ic.ac.uk/vision/).

**Split**

The train/test/validation split for Endovis dataset used in our works is defined in the `splits/endovis` folder. 

**Data structure**

The directory of dataset structure is shown as follows:

```
/path/to/endovis_data/
  dataset1/
    keyframe1/
      image_02/
        data/
          0000000001.png
```



## ⏳ Endovis training

**Stage-wise fashion:**

Stage one:

```shell
CUDA_VISIBLE_DEVICES=0 python train_stage_one.py --data_path <your_data_path> --log_dir <path_to_save_model (optical flow)>
```

Stage two:

```shell
CUDA_VISIBLE_DEVICES=0 python train_stage_two.py --data_path <your_data_path> --log_dir <path_to_save_model (depth, pose, appearance flow, optical flow)> --load_weights_folder <path_to_the_trained_optical_flow_model_in_stage_one>
```

**End-to-end fashion:**

```shell
CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py --data_path <your_data_path> --log_dir <path_to_save_model (depth, pose, appearance flow, optical flow)>
```



## 📊 Endovis evaluation

To prepare the ground truth depth maps run:
```shell
CUDA_VISIBLE_DEVICES=0 python export_gt_depth.py --data_path endovis_data --split endovis
```
...assuming that you have placed the endovis dataset in the default location of `./endovis_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> --load_weights_folder ~/mono_model/mdp/models/weights_19 --eval_mono
```
#### Appearance Flow

<p align="center">
<img src='imgs/appearance_flow.png' width=800/> 
</p>

#### Depth Estimation

<p align="center">
<img src='imgs/depth.png' width=800/> 
</p>

#### Visual Odometry

<p align="center">
<img src='imgs/pose.png' width=800/> 
</p>

#### 3D Reconstruction

<p align="center">
<img src='imgs/reconstruction.png' width=800/> 
</p>


## 📦 Model zoo

| Model        | Abs Rel | Sq Rel | RMSE | RMSE log | Link |
| ------------ | ---------- | ------ | --------- | ---- | ---- |
| Stage-wise (ID 5 in Table 8) | 0.059 | 0.435 | 4.925 | 0.082 |[baidu](https://pan.baidu.com/s/1MT5RrbDl8Wh6otPihD0kEw) (code:n6lh); [google](https://drive.google.com/file/d/14VFlTHq6raQkdyCRBCQYV-mbFO4eOM5b/view?usp=sharing)|
| End-to-end (ID 3 in Table 8) | 0.059 | 0.470 | 5.062 | 0.083 |[baidu](https://pan.baidu.com/s/1JrcMBU0wKCbgEdiF2kzQ6A) (code:z4mo); [google](https://drive.google.com/file/d/1kf7LjQ6a2ACKr6nX5Uyee3of3bXn1xWB/view?usp=sharing)|
| ICRA  | 0.063 | 0.489 | 5.185 | 0.086 |[baidu](https://pan.baidu.com/s/11SogWGI7C7kUGTkABPTMOA) (code:wbm8); [google](https://drive.google.com/file/d/1klpUlkYtXZiRsjY6SdRHvNAKDoYc-zgo/view?usp=sharing)|

## Important Note

If you use the latest PyTorch version,

Note1: please try to add 'align_corners=True' to 'F.interpolate' and 'F.grid_sample' when you train the network, to get a good camera trajectory.

Note2: please revise color_avg=transforms.ColorJitter.get_params(self.brightness,self.contrast,self.saturation,self.hue) to color_avg=transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue).



## Contact

If you have any questions, please feel free to contact swshao@buaa.edu.cn.



## Acknowledgement

Our code is based on the implementation of [Monodepth2](https://github.com/nianticlabs/monodepth2). We thank Monodepth2's authors for their excellent work and repository.
