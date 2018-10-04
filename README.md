# QARC (video Quality Aware Rate Control)

## Introduction
Real-time video streaming is now one of the main applications in all network environments. Due to the fluctuation of throughput under various network conditions, how to choose a proper bitrate adaptively has become an upcoming and interesting issue. To tackle this problem, most proposed rate control methods work for providing high video bitrates instead of video qualities. Nevertheless, we notice that there exists a trade-off between sending bitrate and video quality, which motivates us to focus on how to reach a balance between them. We then propose QARC (video Quality Aware Rate Control), a rate control algorithm that aims to obtain a higher perceptual video quality with possible lower sending rate and transmission latency. In detail, QARC uses deep reinforcement learning(DRL) algorithm to train a neural network to select future bitrates based on previously observed network status and past video frames. To overcome the ``state explosion problem'', we design a neural network to predict future perceptual video quality as a vector for taking the place of the raw picture in the DRL's inputs. 

![overview](overview.png)

## Video datasets
In 'videodatasets/', we describe how to generate video datasets.

1. Clone vmaf repo. for computing vmaf score: git clone https://github.com/Netflix/vmaf
2. Copy the 'videodatasets' folder to the 'vmaf' folder.
3. Download several video clips into 'videodatasets/mov' folders, and the video MUST be encoded as h.264 format.
4. Run 'trans.py' for transcoding 'mp4' video format to 'flv'.
5. Run 'main.py' for generating video datasets including logs and video frames.
6. Type 'process-vmaf.py' to generate h5py file for training.

## QARC-basic
The traditional QARC method is composed of two modules: VQPN and VQRL.
### VQPN

## advanced-QARC

