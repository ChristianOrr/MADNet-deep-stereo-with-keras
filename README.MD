# TensorFlow 2 / Keras Implementation of MADNet
## Important Update:
This code is currently being migrated over to Keras 3. Check back soon for updates.

## Introduction
MADNet is a deep stereo depth estimation model. Its key defining features are:
 1. It has a light-weight architecture which means it has low latency.
 2. It supports self-supervised training, so it can be conveniently adapted in the field with no training data. 
 3. It's a stereo depth model, which means it's capable of high accuracy.

This code is an implementation of MADNet in TensorFlow 2 / Keras. MADNet was created using the Keras Functional API and the Subclassed API. The subclassed method was abandoned due to having poor optimization. Due to great support from Keras the Functional method has following features:
 1. Good optimization. 
 2. High level Keras methods (.fit, .predict and .evaluate).
 3. Little boilerplate code.
 4. Decent support from external packages (like Weights and Biases). 
 5. Callbacks.

### MADNet Functional API Capabilities
- Training using groundtruth and without groundtruth (self-supervised) with the .fit method. 
- Inferencing without adaptation and with full MAD adaptation with the .predict method. 
- MAD inferencing (adapting between 1-5 modules) has support with some limitations. It doesn't support graph tracing, so eager mode must be on when using .predict in this mode. The MAD mode options are "random" (randomly select the modules) and "sequential" (select modules in order). 
- MAD++ isnt currently implemented. 

### Supplementary Material
- Pretrained model weights and training info: [madnet_keras](https://huggingface.co/ChristianOrr/madnet_keras)
- Subclassed MADNet implementation: [subclassed-madnet-keras](https://github.com/ChristianOrr/subclassed-madnet-keras)
- Tensorflow 1 MADNet implementation: [Real-time-self-adaptive-deep-stereo](https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo).
- Research paper for MADNet and MAD: [Real-time self-adaptive deep stereo](https://arxiv.org/abs/1810.05424). 
- MAD++ adaptation paper: [Continual Adaptation for Deep Stereo](https://arxiv.org/pdf/2007.05233.pdf).
- Stereo Camera Calibration: [stereo-camera-calibration](https://github.com/ChristianOrr/stereo-camera-calibration).

![image](images/MADNet.png)

**Abstract**:

Deep convolutional neural networks trained end-to-end are the undisputed state-of-the-art methods to regress dense disparity maps directly from stereo pairs. However, such methods suffer from notable accuracy drops when exposed to scenarios significantly different from those seen in the training phase (e.g.real vs synthetic images, indoor vs outdoor, etc). As it is unlikely to be able to gather enough samples to achieve effective training/ tuning in any target domain, we propose to perform unsupervised and continuous online adaptation of a deep stereo network in order to preserve its accuracy independently of the sensed environment. However, such a strategy can be extremely demanding regarding computational resources and thus not enabling real-time performance. Therefore, we address this side effect by introducing a new lightweight, yet effective, deep stereo architecture Modularly ADaptive Network (MADNet) and by developing Modular ADaptation (MAD), an algorithm to train independently only sub-portions of our model. By deploying MADNet together with MAD we propose the first ever realtime self-adaptive deep stereo system.


## Requirements
This software has been developed and tested with python 3.9 and tensorflow 2.8. All required packages can be installed using pip and requirements.txt

```
pip3 install -r requirements.txt
```

## Pretrained Weights
Pretrained weights from flyingthings-3d and kitti datasets are available in the huggingface repo: [madnet_keras](https://huggingface.co/ChristianOrr/madnet_keras). Note that it's not necessary to manually download the weights. The weights can be automatically downloaded and loaded into the model by specifying "kitti" or "synthetic" for the --weights_path parameter while training or inferencing.

## Data Preparation
For the training and inferencing scripts to function the data needs to be prepared in the following way. For supervised training the left rectified, right rectified and groundtruth is needed. Optionally for validation, the rectified left, rectified right and groundtruth can also be used during supervised training. Note that Bad3 and EPE metrics can only be provided if groundtruth is available. Each file needs to be placed in its own folder with the same filename, since the left, right and groundtruth is matched based on the order they are placed in the folder. For unsupervised training / inferencing the same method is used but only left and right rectified images are needed. If you would like to load the images using a different method (like from a CSV file) you will need to update the StereoDatasetCreator class with your own code in the preprocessing.py script.  

Note: The stereo images need to be rectified for both supervised and unsupervised training and inferencing. If you don't have rectified stereo images or don't know what that is, don't worry, I have created a notebook to show you how to calibrate and rectify your stereo images  [here](https://colab.research.google.com/github/ChristianOrr/stereo-camera-calibration/blob/main/stereo_calibration.ipynb).

## MADNet Training
Two training options are available, supervised training (with groundtruth disparity) and unsupervised training (no groundtruth disparity). Its recommended to either use pretrained weights or perform supervised training on the model before performing unsupervised training on your dataset. Alternatively fine-tuning on the model can be performed while inferencing. Examples for both training methods are listed below:
1. Supervised Training
   ```bash
   python3 train.py \
   --train_left_dir /path/to/left/train_images \
   --train_right_dir /path/to/right/train_images \
   --train_disp_dir /path/to/right/train_images \
   -o /path/to/output/trained_model \
   --height 480 \
   --width 640 \
   --batch_size 1 \
   --num_epochs 500 \
   --epoch_steps 1000 \
   --save_freq 1000 \
   --log_tensorboard \
   --val_left_dir /path/to/left/val_images \
   --val_right_dir /path/to/right/val_images \
   --val_disp_dir /path/to/right/val_images \
   --weights_path kitti 
   ```
2. Un-Supervised Training
   ```bash
   python3 train.py \
   --train_left_dir /path/to/left/train_images \
   --train_right_dir /path/to/right/train_images \
   -o /path/to/output/trained_model \
   --height 480 \
   --width 640 \
   --batch_size 1 \
   --num_epochs 500 \
   --epoch_steps 1000 \
   --save_freq 1000 \
   --log_tensorboard \
   --weights_path kitti 
   ```
## Weights and Biases 
### Training    
Logging training to wandb is also an option. The training parameters are the same as above, but an additional --dataset_name parameter is available for logging the dataset. To log training to wandb just run the command above but use the train_wandb.py script instead. You will be prompted to login to your wandb account, after you've logged in the training run will automaticallly start uploading the training results to wandb.   

### Sweeps
Hyperparameter sweeps are also supported using wandb. To perform a sweep you need to first create a sweep config file. See sweep_config_example.yaml for an example of how the sweep config must look. Once you've created your sweep config, initialize the sweep with the command below:
```commandline
wandb sweep sweep_config.yaml
```
This command will print out a sweep ID, which includes the entity name and project name. Copy that to use in the next step! In a shell on your own machine, run the wandb agent command:
```commandline
wandb agent <USERNAME/PROJECTNAME/SWEEPID>
```
The sweep should start immediately.

For further details on sweeps see the wandb site: [sweeps quickstart](https://docs.wandb.ai/guides/sweeps/quickstart)

## Inferencing / Online Adaptation
Inferencing can be performed using full MAD (adapt all modules), MAD (adapt 1-5 modules), or inferencing only. Examples of performing these three options are shown below:

1. Full MAD
   ```bash
   python3 inferencing.py \
   --left_dir /path/to/left/rectified_images \
   --right_dir /path/to/right/rectified_images \
   --num_adapt 6 \
   -o /path/to/save/adapted_model \
   --weights_path /path/to/pretrained/model \
   --height 480 \
   --width 640 \
   --batch_size 1
   ```
2. MAD
   ```bash
   python3 inferencing.py \
   --left_dir /path/to/left/rectified_images \
   --right_dir /path/to/right/rectified_images \
   --num_adapt 1 \
   --mad_mode random
   -o /path/to/save/adapted_model \
   --weights_path /path/to/pretrained/model \
   --height 480 \
   --width 640 \
   --batch_size 1     
   ```
2. Inferencing Only
   ```bash
   python3 inferencing.py \
   --left_dir /path/to/left/rectified_images \
   --right_dir /path/to/right/rectified_images \
   --num_adapt 0 \
   --weights_path /path/to/pretrained/model \
   --height 480 \
   --width 640 \
   --batch_size 1     
   ```
## Adaptation Results
### Disparity Maps
The image below shows the predictions of the different methods at each step. The depth map colors are such that red is closest, green is medium distance and blue far.

Starting from the left to right column:
1. Left rectified RGB images.
2. Disparity predictions from the pretrained Kitti model (no adaptation).
3. Disparity predictions from pretrained synthetic weights with full MAD (adapting all 6 modules).
4. Disparity predictions from pretrained synthetic weights with MAD (adapting 1 module randomly).
5. Disparity predictions from pretrained synthetic weights (no adaptation).


![image](images/adaptation_results.png)

### Inferencing speed
The table and graph below shows the frames per second for the different inferencing modes on dataset sizes of 100, 1,000 and 10,000 images, using an RTX 3050 Ti laptop GPU. The results were obtained using the evaluation.py script on kitti images (480x640) with no logging or saving images. The results show the model perform significantly faster on larger dataset sizes. The MAD adaptation modes only start to show a sizeable benefit over Full MAD at the 10,000 dataset size. Sequential MAD performs consistently worse than both Full MAD and random MAD at all dataset sizes, making this method redundant unless further optimizations can be found.

Note that graph execution is enabled by default using all methods. This provides a speed increase at the cost of having a very slow start while the graph is being traced. The results show that the graph tracing overhead gets diminished as the dataset size increases.

|                  | FPS 100 | FPS 1,000 | FPS 10,000 |
|------------------|---------|-----------|------------|
| No Adaptation    | 11.1    | 31.8      | 37.8       |
| Full MAD         | 2.6     | 10.3      | 14.2       |
| MAD 1 Random     | 2.4     | 11.1      | 18.6       |
| MAD 2 Random     | 2.4     | 10.7      | 18.4       |
| MAD 3 Random     | 2.3     | 10.8      | 16.9       |
| MAD 4 Random     | 2.2     | 11.0      | 15.1       |
| MAD 5 Random     | 2.1     | 9.9       | 14.4       |
| MAD 1 Sequential | 2.1     | 9.6       | 14.5       |
| MAD 2 Sequential | 2       | 8.1       | 11.6       |
| MAD 3 Sequential | 1.9     | 6.7       | 8.9        |
| MAD 4 Sequential | 1.8     | 5.7       | 7.2        |
| MAD 5 Sequential | 1.8     | 5.1       | 6.2        |

![image](images/frame_rate_line_plot.png)

## BibTeX entry and citation info

```bibtex
@InProceedings{Tonioni_2019_CVPR,
    author = {Tonioni, Alessio and Tosi, Fabio and Poggi, Matteo and Mattoccia, Stefano and Di Stefano, Luigi},
    title = {Real-time self-adaptive deep stereo},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}    
}
```

```bibtex
@article{Poggi2021continual,
    author={Poggi, Matteo and Tonioni, Alessio and Tosi, Fabio
            and Mattoccia, Stefano and Di Stefano, Luigi},
    title={Continual Adaptation for Deep Stereo},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
    year={2021}
}
```

```bibtex
@InProceedings{MIFDB16,
  author    = "N. Mayer and E. Ilg and P. Hausser and P. Fischer and D. Cremers and A. Dosovitskiy and T. Brox",
  title     = "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation",
  booktitle = "IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)",
  year      = "2016",
  note      = "arXiv:1512.02134",
  url       = "http://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16"
}
```

```bibtex
@INPROCEEDINGS{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}
```

```bibtex
@INPROCEEDINGS{Menze2015CVPR,
  author = {Moritz Menze and Andreas Geiger},
  title = {Object Scene Flow for Autonomous Vehicles},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2015}
}
```

