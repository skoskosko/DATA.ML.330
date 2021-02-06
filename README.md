# README.md

This repository contains project for TUNI DATA.ML.330 

this is stolen from https://github.com/pytorch/vision/blob/master/references/detection/utils.py and https://towardsdatascience.com/building-your-own-object-detector-pytorch-vs-tensorflow-and-how-to-even-get-started-1d314691d4ae
And probably some of it could be included directly with pip packages. This is a work in progress.

And some files have duplicate functions and such. This hopefully will be cleaned at some point

## Omitted files and directories

Some files are omitted from this repository but are required for successful usage.

### Folders

- /Dry
- /Dry/images
- /output
- /output/slices
- /output/report


### Files

-/Dry/images/{2500 images of plywood named like `dry_n.png` where n is number of image from 0 to 2578}

## Usage

All the commands are ran inside devcontainer.

### Using devcontainer

To use the devcontainer in which everything is installed you need to install ms-vscode-remote.remote-containers plugin to visual studio code. 
If you dont want to, check .devcontainer/Dockerfile for dependencies

Just start devcontainer. You can check https://code.visualstudio.com/docs/remote/containers out if you have problems

### Preparing data

``` sh
python3 src/prepare.py
```

This should generate labels.csv inside Dry folder

### Training model


``` sh
python3 src/train.py
```
This should create a folder named slice_model with model inside it

### Testing model

``` sh
python3 src/test.py
```

This should print some output for you to wonder

### Generating nice pictures

``` sh
python3 src/make_output.py
```

Should create files in /output/report folder

## Knwon limitations

showimage is not working, because current display is not linked inside devcontainer