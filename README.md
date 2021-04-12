# NTOU CS Computer Vision Homework Collection
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&style=flat-square&logoColor=white)](https://github.com/pre-commit/pre-commit)
![bugs](https://img.shields.io/badge/README-Great_IMO-green?&style=flat-square&logo=Read%20The%20Docs&logoColor=white)
![bugs](https://img.shields.io/badge/Bugs-A_lot-orange?&style=flat-square&logo=stackoverflow&logoColor=white) 
![bugs](https://img.shields.io/badge/CI-NOT_NOW-red?&style=flat-square&logo=CircleCI&logoColor=white)

## Requirements
- pipenv
- pyenv (optional)
- Python 3rd-party packages
  - opencv-python
  - numpy
  - matplotlib
  - pillow
  - tensorflow

## Before anything
- This repo is developed based on Python 3.9.1 on pipenv
- Run `pipenv install` to install dependencies
- Put all medias into `data/` (Recommended)

---

## HW1 - Video Filter (WIP)
### Overview
- Contians 4 frame filters (2 todo)
  - RGB -> BGR
  - HEATMAP

### Dive into the code
  - Entry `hw1_filter/__main__.py`
  - to run it, `pipenv run hw1 --source <path-to-input-video>`