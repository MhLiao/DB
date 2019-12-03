# Introduction
This is a PyToch implementation of "Real-time Scene Text Detection with Differentiable Binarization". [This paper](https://arxiv.org/abs/1911.08947) presents a real-time arbitrary-shape scene text detector, achieving the state-of-the-art performance on standard benchmarks.

Part of the code is inherited from [MegReader](https://github.com/Megvii-CSG/MegReader).

## ToDo List

- [x] Release code
- [x] Document for Installation
- [x] Trained models
- [x] Document for testing and training
- [x] Evaluation
- [ ] Demo script
- [ ] More models on more datasets

## Installation

### Requirements:
- Python3
- PyTorch >= 1.2 
- GCC >= 4.9 (This is important for PyTorch)
- CUDA >= 9.0 (10.1 is recommended)


```bash
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name DB -y
  conda activate DB

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install -r requirements.txt

  # install PyTorch with cuda-10.1
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

  # clone repo
  git clone https://github.com/MhLiao/DB.git
  cd DB/

  # build deformable convolution opertor
  cd assets/ops/dcn/
  python setup.py build_ext --inplace

```

## Models
Download Trained models [Baidu Drive](https://pan.baidu.com/s/1vxcdpOswTK6MxJyPIJlBkA) (download code: p6u3), [Google Drive (ToDo)]() (My connection to Google is not stable.)
```
  pre-trained-model-synthtext   -- used to finetune models, not for evaluation
  td500_resnet18
  td500_resnet50
  totaltext_resnet18
  totaltext_resnet50
```

## Datasets
The root of the dataset directory can be ```DB/datasets/```.

Download the converted ground-truth and data list [Baidu Drive](https://pan.baidu.com/s/1BPYxcZnLXN87rQKmz9PFYA) (download code: mz0a), [Google Drive (ToDo)]() (My connection to Google is not stable.). The images of each dataset can be obtained from their official website.

## Testing
### Prepar dataset
An example of the path of test images: 
```
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt
```
The data root directory and the data list file can be defined in ```base_totaltext.yaml```

### Evaluate the performance
Note that we do not provide all the protocols for all benchmarks for simplification. The embedded evaluation protocol in the code is modified from the protocol of ICDAR 2015 dataset while support arbitrary-shape polygons. It almost produces the same results as the pascal evaluation protocol in Total-Text dataset. 

```python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.6```

```box_thresh``` can be used to balance the precision and recall, which may be different for different datasets to get a good F-measure. ```polygon``` is only used for arbitrary-shape text dataset. The size of the input images are defined in ```validate_data->processes->AugmentDetectionData``` in ```base_*.yaml```.

### Evaluate the speed 
Set ```adaptive``` to ```False``` in the yaml file to speedup the inference without decreasing the performance. The speed is evaluated by performing a testing image for 50 times to exclude extra IO time.

```python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.6 --speed```

Note that the speed is related to both to the GPU and the CPU since the model runs with the GPU and the post-processing algorithm runs with the CPU.

## Training
Check the paths of data_dir and data_list in the base_*.yaml file. For better performance, you can first per-train the model with SynthText and then fine-tune it with the specific real-world dataset.

```python train.py path-to-yaml-file --num_gpus 4```

You can also try distributed training (not fully tested)

```python -m torch.distributed.launch --nproc_per_node=4 train.py path-to-yaml-file --num_gpus 4```

## Improvements
Note that the current implementation is written by pure Python code except for the deformable convolution operator. Thus, the code can be further optimized by some optimization skills, such as [TensorRT](https://github.com/NVIDIA/TensorRT) for the model forward and efficient C++ code for the [post-processing function](https://github.com/MhLiao/DB/blob/d0d855df1c66b002297885a089a18d50a265fa30/structure/representers/seg_detector_representer.py#L26).

Another option to increase speed is to run the model forward and the post-processing algorithm in parallel through a producer-consumer strategy.

Contributions or pull requests are welcome.

## Citing the related works

Please cite the related works in your publications if it helps your research:

     @inproceedings{liao2020real,
      author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
      title={Real-time Scene Text Detection with Differentiable Binarization},
      booktitle={Proc. AAAI},
      year={2020}
    }


    

