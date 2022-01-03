# Renderer-with-pytorch3d
It is a rendering code with pytorch3d which can be used when you want to convert pyrend, trimesh or opendr to pytorch3d renderer.

# Purpose of this repo

When you clone projects about 3D vision, many projects use pyrend, trimesh or opendr to render the result(human mesh).
But pyrend and opendr have many dependencies with the latest python, pytorch and cuda version,
which means training speed is limited to low cuda version.

So I wrote renderer code with pytorch3d that supports the latest pytorch version.
I hope you can save rendering time by using this code.

# How to use

## Installation instructions
I suggest to use conda virtual env to use pytorch3d renderer with python >= 3.7, pytorch >= 1.6.0
You can download compatible pytorch3d files here (https://anaconda.org/pytorch3d/pytorch3d/files)
You can use both CUDA == 10.1 and 11.1 or other compatible versions that pytorch3d provide.
Except Pytorch3d related libraries, I suggest you to download other libraries with requirement.txt of your projects

```
git clone https://github.com/hyeonLewis/Renderer-with-pytorch3d
conda install ***.tar.bz2 #Pytorch3d tar file you download
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge #Check version with pytorch3d 
```
