# QARC (video Quality Aware Rate Control)

## Introduction
Real-time video streaming is now one of the main applications in all network environments. Due to the fluctuation of throughput under various network conditions, how to choose a proper bitrate adaptively has become an upcoming and interesting issue. To tackle this problem, most proposed rate control methods work for providing high video bitrates instead of video qualities. Nevertheless, we notice that there exists a trade-off between sending bitrate and video quality, which motivates us to focus on how to reach a balance between them. We then propose QARC (video Quality Aware Rate Control), a rate control algorithm that aims to obtain a higher perceptual video quality with possible lower sending rate and transmission latency. In detail, QARC uses deep reinforcement learning(DRL) algorithm to train a neural network to select future bitrates based on previously observed network status and past video frames. To overcome the *state explosion problem*, we design a neural network to predict future perceptual video quality as a vector for taking the place of the raw picture in the DRL's inputs. 

![overview](overview.png)

## Install Requirements

The current codebase runs on Python 2.7. This work requires tensorflow, tflearn, scipy, matplotlib, opencv2 and tensorboard. Please install these package before using QARC. 

```
pip install tensorflow==1.15 tflearn==0.3.2 scipy==1.2.2 matplotlib opencv-python tensorboard
```

## Video datasets

In `videodatasets/`, we describe how to generate video datasets.

1. Clone the vmaf git repository
    ```
    git clone https://github.com/Netflix/vmaf
    ```
    and checkout version 1.3.9 (available on the `v1.3.9` tag).
2. Copy the `videodatasets/` folder to the `vmaf/` folder.
3. Download several video clips into `videodatasets/mov/` folders, and the video MUST be encoded in h.264 format.
4. Run `trans.py` for transcoding `mp4` video format to `flv`.
5. Run `main.py` for generating video datasets including logs and video frames.
    > Note: To avoid any potential errors, make sure to install ffmpeg, find your path to `ffmpeg` by running `which ffmpeg` in the command line, and set `FFMPEG_PATH` in `vmaf/python/src/vmaf/externals.py`, e.g. add a line like
    ```
    FFMPEG_PATH="[path to exec]/ffmpeg"
    ````

6. Run `process-vmaf.py` to generate a h5py file for training.
   > Note: This script expects frame image files from the previous step to be in a subdirectory named `img/<original_video_filename>/`, e.g., if the original file was `1.mp4`, then its frame image files would be under `img/1.mp4/`.

## QARC-basic
The traditional QARC method is composed of two modules: VQPN and VQRL.

### VQPN (Video Quality Prediction Network)
To run and evaluate VQPN, you should follow these steps.

1. Copy the `*.h5` training dataset and test dataset from `videodatasets/` to `QARC/VQPN`
   
2. There are two nerual network architecture candidates, GRU model (as shown in the paper) and CNN model (also reaches a high performance but a little bit lower than the GRU model performs), in different folders. Type `cd gru/ & python main.py` for training GRU model or `cd cnn/ & python main.py` for training CNN model. We use early stopping method that automaticlly tear down the training process if overfitting occurs.
   
3. We provide several baselines for evaluating the performance of VQPN in `baseline/` folder.

### VQRL (Video Quality Reinforcement Learning)
To run and test VQRL, one should follow these steps:

1. Copy the pretrain VQPN model to `QARC/VQRL/model/`, we recommend GRU model for higher performances.
   
2. Collect network traces and copy the traces to `QARC/VQRL/train_sim_traces/`, the format of network traces is described like this: `timestamp\tbandwidth\n`, and you can also use cooked_traces from Pensieve as network trace dataset.
   
3. Run the training process via `python main.py`

The usage of advanced-QARC is quite the same as QARC.

## Cite

If you find this work useful to you, please cite

```
@inproceedings{huang2018qarc,
  title={Qarc: Video quality aware rate control for real-time video streaming based on deep reinforcement learning},
  author={Huang, Tianchi and Zhang, Rui-Xiao and Zhou, Chao and Sun, Lifeng},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={1208--1216},
  year={2018}
}
```
