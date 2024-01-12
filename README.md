# On the representation and methodology for wide and short range head pose estimation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/on-the-representation-and-methodology-for/head-pose-estimation-on-aflw2000)](https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000?p=on-the-representation-and-methodology-for)

We provide Python code in order to replicate the head pose experiments in our paper https://doi.org/10.1016/j.patcog.2024.110263

If you use this code for your own research, you must reference our journal paper:

```
On the representation and methodology for wide and short range head pose estimation
Alejandro Cobo, Roberto Valle, José M. Buenaposada, Luis Baumela.
Pattern Recognition, PR 2024.
https://doi.org/10.1016/j.patcog.2024.110263
```

#### Requisites
- images_framework https://github.com/pcr-upm/images_framework
- pytorch (v1.13.0)
- sciPy

#### Installation
This repository must be located inside the following directory:
```
images_framework
    └── alignment
        └── opal23_headpose
```
#### Usage
```
usage: opal23_headpose_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Alignment --database DATABASE
```

* Use the --database option to select the database model.
```
usage: Opal23Headpose [--gpu GPU]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).

```
usage: Opal23Headpose [--rotation-mode {euler,quaternion,6d,6d_opal}]
```

* Use the --rotation-mode option to specify the internal pose parameterization of the network.
```
> python images_framework/alignment/opal23_headpose/test/opal23_headpose_test.py --input-data images_framework/alignment/opal23_headpose/test/example.tif --database 300wlp --gpu 0 --rotation-mode euler --save-image
```
