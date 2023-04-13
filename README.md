# Head pose using OPAL (2023)

#### Requisites
- images_framework https://github.com/bobetocalo/images_framework

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
> python images_framework/alignment/opal23_headpose/test/opal23_headpose_test.py --input-data images_framework/alignment/opal23_headpose/test/example.tif --database AFLW --gpu 0 --save-image
```
