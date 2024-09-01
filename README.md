# Mesh Density Adaptation for Template-based Shape Reconstruction
\[[ArXiv](http://arxiv.org/abs/2307.16205)\] \[[Video](https://youtu.be/L-WNBUNyP-Y)\] \[[Paper](https://doi.org/10.1145/3588432.3591498)\]

Official code for Yucheol Jung*, Hyomin Kim*, Gyeongha Hwang, Seung-Hwan Baek, Seungyong Lee,
"Mesh Density Adaptation for Template-based Shape Reconstruction", SIGGRAPH 2023.
(Jung and Kim shares equal contribution)

![image](images/teaser.png)

This repository contains the code for the density adaptation module and scripts for the experiments introduced in the paper.

## Setup

* Clone this repository including the submodules
```bash
git clone --recursive https://github.com/ycjungSubhuman/density-adaptation
```

#### Docker
* Launch a docker environment using the image `min00001/adadense`
```bash
docker run --gpus all -v $PWD:/workspace -it min00001/adadense /bin/bash
cd /workspace
conda activate lapf
```

#### Docker (Build from dockerfile)

```bash
cd docker
docker build -t gltorch .
cd ..
docker run --gpus all -v $PWD:/workspace -it gltorch /bin/bash
cd /workspace
python generate_mass.py
```

#### Non-docker

1. Install pytorch and nvdiffrast
1. `pip install -r docker/requirements.txt`

### Inverse Rendering

1. Download the scene files from https://github.com/rgl-epfl/large-steps-pytorch and save the `scene` directory under `ext/large-steps`
2. `python generate_mass.py`


### Non-rigid registration

1. Download the 3DCaricShop data from https://qiuyuda.github.io/3DCaricShop/ and save the contents of `processedData` under `./3dcaricshop/original_data/processedData/`
2. `python fitting_sphere.py`

### Non-rigid registration (with landmarks)

1. Download the 3DCaricShop data from https://qiuyuda.github.io/3DCaricShop/ and save the contents of `processedData` under `./3dcaricshop/original_data/processedData/`
2. `python fitting_sphere_lmk.py`

- Note 1: The landmark for sphere in `./3dcaricshop/data/` is produced via landmark averaging described in the paper.
- Note 2: So, in this case, the baseline `large_steps` also benefits from our density adaptation because the density adaption is already "embedded" in the landmark annotation.
- Note 3: This script may not completely reproduce the reported numerical errors in the paper. Visually they should produce similar results.

## Citation

If you want to cite this code, you may refer to this bibtex entry
```bibtex
@inproceedings{jung2023density,
  author = {Jung, Yucheol and Kim, Hyomin and Hwang, Gyeongha and Baek, Seung-Hwan and Lee, Seungyong},
  title = {Mesh Density Adaptation for Template-Based Shape Reconstruction},
  year = {2023},
  isbn = {9798400701597},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3588432.3591498}, doi = {10.1145/3588432.3591498},
  abstract = {In 3D shape reconstruction based on template mesh deformation, a regularization, such as smoothness energy,
  is employed to guide the reconstruction into a desirable direction. In this paper, we highlightan often overlooked property
  in the regularization: the vertex density in the mesh. Without careful control on the density, the reconstruction may suffer
  from under-sampling of vertices near shape details. We propose a novel mesh density adaptation method to resolve the
  under-sampling problem. Our mesh density adaptation energy increases the density of vertices near complex structures via deformation
  to help reconstruction of shape details. We demonstrate the usability and performance of mesh density adaptation with two tasks,
  inverse rendering and non-rigid surface registration. Our method produces more accurate reconstruction results compared to the cases
  without mesh density adaptation. Our code is available at https://github.com/ycjungSubhuman/density-adaptation.},
  booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
  articleno = {53}, numpages = {10},
  keywords = {diffusion re-parameterization, Laplacian regularization, non-rigid registration, Inverse rendering},
  location = {Los Angeles, CA, USA},
  series = {SIGGRAPH '23}
}
```

## Acknowledgement
This code builds upon https://github.com/rgl-epfl/large-steps-pytorch . We thank the authors for sharing their code.
