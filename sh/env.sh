export PATH=/data/mseizde/bin:$PATH
export MAMBA_ROOT_PREFIX=/data/mseizde/micromamba
eval "$(/data/mseizde/bin/micromamba shell hook --shell bash)"

micromamba create -n com4d python=3.11.13 -y
micromamba activate com4d

python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
python -m pip install -r requirements.txt

micromamba install -c conda-forge libegl libglu pyopengl -y

python -m pip install -U pip setuptools wheel
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
