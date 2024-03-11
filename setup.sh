#!/usr/bin/env bash

conda create -n cs224n_dfp python=3.8
conda activate cs224n_dfp

conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install tqdm==4.58.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install scikit-learn==0.24.1
pip install tokenizers==0.10.1
pip install explainaboard_client==0.0.7
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
