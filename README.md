# AF-SfMLearner

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue**
>
> [Shuwei Shao](https://scholar.google.com.hk/citations?hl=zh-CN&user=ecZHSVQAAAAJ), Zhongcai Pei, [Weihai Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=5PoZrcYAAAAJ), [Wentao Zhu](https://scholar.google.com.hk/citations?hl=zh-CN&user=2hjYfqIAAAAJ), Xingming Wu, Dianmin Sun and [Baochang Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=ImJz6MsAAAAJ).
>
> [Medical Image Analysis 2022 (arXiv pdf)](https://arxiv.org/pdf/2112.08122.pdf)

and 

> **Self-Supervised Learning for Monocular Depth Estimation on Minimally Invasive Surgery Scenes**
>
> [Shuwei Shao](https://scholar.google.com.hk/citations?hl=zh-CN&user=ecZHSVQAAAAJ), Zhongcai Pei, [Weihai Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=5PoZrcYAAAAJ), [Baochang Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=ImJz6MsAAAAJ), Xingming Wu, Dianmin Sun and [David Doermann](https://scholar.google.com.hk/citations?hl=zh-CN&user=RoGOW9AAAAAJ).
>
> [ICRA 2021 (pdf)](https://ieeexplore.ieee.org/abstract/document/9561508)



## ✏️ 📄 Citation
If you find our work useful in your research please consider citing our paper:

```
@article{shao2021self,
  title={Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue},
  author={Shao, Shuwei and Pei, Zhongcai and Chen, Weihai and Zhu, Wentao and Wu, Xingming and Sun, Dianmin and Zhang, Baochang},
  journal={arXiv preprint arXiv:2112.08122},
  year={2021}
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

