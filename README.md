## News
* DBNet and DBNet++ are included in [MindOCR](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet), a MindSpore implementation.
* The ASF module in DBNet++([TPAMI](https://ieeexplore.ieee.org/abstract/document/9726868/), [arxiv](https://arxiv.org/abs/2202.10304)) is released.
* DB is included in [WeChat OCR engine](https://mp.weixin.qq.com/s/6IGXof3KWVnN8z1i2YOqJA)
* DB is included in [OpenCV](https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown)
* DB is included in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

# Introduction
This is a PyToch implementation of DBNet([arxiv](https://arxiv.org/abs/1911.08947)) and DBNet++([TPAMI](https://ieeexplore.ieee.org/abstract/document/9726868/), [arxiv](https://arxiv.org/abs/2202.10304)).  It presents a real-time arbitrary-shape scene text detector, achieving the state-of-the-art performance on standard benchmarks.

Part of the code is inherited from [MegReader](https://github.com/Megvii-CSG/MegReader).

## ToDo List

- [x] Release code
- [x] Document for Installation
- [x] Trained models
- [x] Document for testing and training
- [x] Evaluation
- [x] Demo script
- [x] Release DBNet++ code
- [x] Release DBNet++ models



## Installation

### Requirements:
- Python3
- PyTorch == 1.2 
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
  pip install -r requirement.txt

  # install PyTorch with cuda-10.1
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

  # clone repo
  git clone https://github.com/MhLiao/DB.git
  cd DB/

  # build deformable convolution opertor
  # make sure your cuda path of $CUDA_HOME is the same version as your cuda in PyTorch
  # make sure GCC >= 4.9
  # you need to delete the build directory before you re-build it.
  echo $CUDA_HOME
  cd assets/ops/dcn/
  python setup.py build_ext --inplace

```

## Models
New: DBNet++ trained models [Google Drive](https://drive.google.com/drive/folders/1buwe_b6ysoZFCJgHMHIr-yHd-hEivQRK?usp=sharing).

Download Trained models [Baidu Drive](https://pan.baidu.com/s/1o-itIZ5P_FH7rwSdpgLLww?pwd=x1er) (download code: x1er), [Google Drive](https://drive.google.com/open?id=1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG).
```
  pre-trained-model-synthtext   -- used to finetune models, not for evaluation
  td500_resnet18
  td500_resnet50
  totaltext_resnet18
  totaltext_resnet50
```

## Datasets
The root of the dataset directory can be ```DB/datasets/```.

Download the converted ground-truth and data list [Baidu Drive](https://pan.baidu.com/s/1VfHGYYWhxHot1RLyrfKOHg?pwd=0drc) (download code: 0drc), [Google Drive](https://drive.google.com/open?id=12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7). The images of each dataset can be obtained from their official website.

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

### Config file
**The YAML files with the name of ```base*.yaml``` should not be used as the training or testing config file directly.**

### Demo
Run the model inference with a single image. Here is an example:

```CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --image_path datasets/total_text/test_images/img10.jpg --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7 --visualize```

The results can be find in `demo_results`.

### Evaluate the performance
Note that we do not provide all the protocols for all benchmarks for simplification. The embedded evaluation protocol in the code is modified from the protocol of ICDAR 2015 dataset while support arbitrary-shape polygons. It almost produces the same results as the pascal evaluation protocol in Total-Text dataset. 

The `img651.jpg` in the test set of Total-Text contains exif info for a 90° rotation thus the gt does not match the image. You should read and re-write this image to get normal results. The converted image is also provided in the dataset links. 

The following command can re-implement the results in the paper:

```
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet50 --polygon --box_thresh 0.6

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet18_deform_thre.yaml --resume path-to-model-directory/td500_resnet18 --box_thresh 0.5

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet50_deform_thre.yaml --resume path-to-model-directory/td500_resnet50 --box_thresh 0.5

# short side 736, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet18_deform_thre.yaml --resume path-to-model-directory/ic15_resnet18 --box_thresh 0.55

# short side 736, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6

# short side 1152, which can be changed in base_ic15.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume path-to-model-directory/ic15_resnet50 --box_thresh 0.6
```

The results should be as follows:

|        Model       	| precision 	| recall 	| F-measure 	| precision (paper) 	| recall (paper) 	| F-measure (paper) 	|
|:------------------:	|:---------:	|:------:	|:---------:	|:-----------------:	|:--------------:	|:-----------------:	|
| totaltext-resnet18 	|    88.9   	|  77.6  	|    82.9   	|        88.3       	|      77.9      	|        82.8       	|
| totaltext-resnet50 	|    88.0   	|  81.5  	|    84.6   	|        87.1       	|      82.5      	|        84.7       	|
|   td500-resnet18   	|    86.5   	|  79.4  	|    82.8   	|        90.4       	|      76.3      	|        82.8       	|
|   td500-resnet50   	|    91.1   	|  80.8  	|    85.6   	|        91.5       	|      79.2      	|        84.9       	|
| ic15-resnet18 (736) |    87.7   	|  77.5  	|    82.3   	|        86.8       	|      78.4     	|        82.3       	|
| ic15-resnet50 (736) |    91.3   	|  80.3  	|    85.4   	|        88.2       	|      82.7      	|        85.4       	|
| ic15-resnet50 (1152)|    90.7   	|  84.0  	|    87.2   	|        91.8      	  |      83.2      	|        87.3       	|


```box_thresh``` can be used to balance the precision and recall, which may be different for different datasets to get a good F-measure. ```polygon``` is only used for arbitrary-shape text dataset. The size of the input images are defined in ```validate_data->processes->AugmentDetectionData``` in ```base_*.yaml```.

### Evaluate the speed 
Set ```adaptive``` to ```False``` in the yaml file to speedup the inference without decreasing the performance. The speed is evaluated by performing a testing image for 50 times to exclude extra IO time.

```CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7 --speed```

Note that the speed is related to both to the GPU and the CPU since the model runs with the GPU and the post-processing algorithm runs with the CPU.

## Training
Check the paths of data_dir and data_list in the base_*.yaml file. For better performance, you can first per-train the model with SynthText and then fine-tune it with the specific real-world dataset.

```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4```

You can also try distributed training (**Note that the distributed mode is not fully tested. I am not sure whether it can achieves the same performance as non-distributed training.**)

```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py path-to-yaml-file --num_gpus 4```

## Improvements
Note that the current implementation is written by pure Python code except for the deformable convolution operator. Thus, the code can be further optimized by some optimization skills, such as [TensorRT](https://github.com/NVIDIA/TensorRT) for the model forward and efficient C++ code for the [post-processing function](https://github.com/MhLiao/DB/blob/d0d855df1c66b002297885a089a18d50a265fa30/structure/representers/seg_detector_representer.py#L26).

Another option to increase speed is to run the model forward and the post-processing algorithm in parallel through a producer-consumer strategy.

Contributions or pull requests are welcome.

## Third-party implementations
* Keras implementation: [xuannianz/DifferentiableBinarization](https://github.com/xuannianz/DifferentiableBinarization)
* DB is included in [OpenCV](https://github.com/opencv/opencv/blob/master/doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown)
* DB is included in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## Citing the related works

Please cite the related works in your publications if it helps your research:

     @inproceedings{liao2020real,
      author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
      title={Real-time Scene Text Detection with Differentiable Binarization},
      booktitle={Proc. AAAI},
      year={2020}
    }

    @article{liao2022real,
      title={Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion},
      author={Liao, Minghui and Zou, Zhisheng and Wan, Zhaoyi and Yao, Cong and Bai, Xiang},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2022},
      publisher={IEEE}
    }


    

