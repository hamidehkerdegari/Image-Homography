# Image-Homography
- Finding A Planar Homography Between Image Pairs
- version: 1.1
- author: Hamideh Kerdegari
- copyright: Copyright 2018
- license: GPL
- email: hamideh.kerdegari@gmail.com
- status: development

## Description
This repository contains a Python script for finding a planar homography between image pairs.
This Python script receives two images and stores the output in the outFolderPath.

## System requirements:
- Python2/3
- OpenCV

### Requaired python packages

Install "argparse", "numpy":

```bash
sudo pip install argparse numpy
```

## Usage:
You can run this script from terminal.

```bash
python homography.py [-h] image1Path image2Path outFolderPath
```
### Example:
```sh
$ python homography.py image1.jpg image2.jpg results
```
