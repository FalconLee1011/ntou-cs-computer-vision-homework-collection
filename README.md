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
  - pre-commit

## Before anything
- This repo is developed based on Python 3.9.1 on pipenv
- Run `pipenv install` to install dependencies
- Put all medias into `data/` (Recommended)

---

## HW1 - Video Filter
### Overview
- Contians 4 frame filters
  - RGB -> BGR (CV2)
  - Heatmap (CV2)
  - Flip upside down (TF)
  - High Pass

### Dive into the code
  - Entry `hw1_filter/__main__.py`
  - To run it, `pipenv run hw1 --source <path-to-input-video>`

### Result
- `pipenv run hw1 --source ./data/homework_1_test_video.mp4`
  <img src="docs/hw1_result.png"/>