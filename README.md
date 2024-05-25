# Polarimetric Light Transport Analysis for Specular Inter-reflection

[**Project page**](https://elerac.github.io/projects/PolarimetricInterreflection/) | [**Paper**](https://arxiv.org/abs/2312.04140)

[Ryota Maeda](https://elerac.github.io/) and [Shinsaku Hiura](https://vislab.jp/hiura/index-e.html), **Polarimetric Light Transport Analysis for Specular Inter-reflection**, IEEE Transactions on Computational Imaging.

![teaser](https://elerac.github.io/projects/PolarimetricInterreflection/teaser_wide.svg)

## Usage

### Requirements

- NumPy
- OpenCV
- [Polanalyser (v2.0.1)](https://github.com/elerac/polanalyser)


### Data

Download the data from [**Google Drive**](https://drive.google.com/drive/folders/1JjRqc4nO469e1E2jbg8wCewpZ6rPg5Ry?usp=sharing) and extract it to the `data` directory.

```sh
.
├── data
│   ├── bunny
|   |   ├── image01.exr
|   |   ├── image01.json
|   |   ├── image02.exr
|   |   ├── image02.json
|   |   ├── ...
│   ├── bowl
│   ├── dragon
│   ├── mac
│   ├── ...
├── decompose_components.py
├── README.md
└── ...
```

### Run

```sh
python3 decompose_components.py data/bunny
```
