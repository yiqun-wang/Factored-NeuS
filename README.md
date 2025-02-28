# Factored-NeuS: Reconstructing Surfaces, Illumination, and Materials of Possibly Glossy Objects

*Yue Fan, Ivan Skorokhodov, Oleg Voynov, Savva Ignatyev, Evgeny Burnaev, Peter Wonka, Yiqun Wang*
**[[Paper Website]](https://yiqun-wang.github.io/Factored-NeuS)**

To appear in CVPR 2025

## Abstract
We develop a method that recovers the surface, materials, and illumination of a scene from its posed multi-view images.
In contrast to prior work, it does not require any additional data and can handle glossy objects or bright lighting.
It is a progressive inverse rendering approach, which consists of three stages.
In the first stage, we reconstruct the scene radiance and signed distance function (SDF) with a novel regularization strategy for specular reflections.
We propose to explain a pixel color using both surface and volume rendering jointly, which allows for handling complex view-dependent lighting effects for surface reconstruction.
In the second stage, we distill light visibility and indirect illumination from the learned SDF and radiance field using learnable mapping functions.
Finally, we design a method for estimating the ratio of incoming direct light reflected in a specular manner and use it to reconstruct the materials and direct illumination.
Experimental results demonstrate that the proposed method outperforms the current state-of-the-art in recovering surfaces, materials, and lighting without relying on any additional data.

## Usage

**Environment**

```shell
pip install - r requirements.txt
```

**Training and Evaluation**

```shell
bash sh_dtu.sh
```

You can modify the parameters in Python execution command in the following format.
```shell
python exp_runner.py --mode train --conf ./confs/wmask.conf --case data_DTU/dtu_scan97 --type dtu --gpu 0 --surface_weight 0.1
```
