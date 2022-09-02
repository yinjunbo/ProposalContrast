## Installation
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint)'s codebase.

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.1 or higher
- CUDA 10.0 or higher
- CMake 3.13.2 or higher
- [APEX](https://github.com/nvidia/apex)
- [SpConv](https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39) 

#### Notes
- Our ProposalContrast support both [SpConv v1.2]((https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39)) and [SpConv v2](https://github.com/traveller59/spconv).
- Orther versions of SpConv may consume more GPU memory.

##### Our project has beed tested in the following environment:

- OS: Ubuntu 18.04
- Python: 3.6.13
- PyTorch: 1.10.1
- CUDA: 11.1

### Basic Installation 

```bash
# basic python libraries
conda create --name proposalcontrast python=3.6
conda activate proposalcontrast
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/yinjunbo/ProposalContrast.git
cd ProposalContrast
pip install -r requirements.txt

# add ProposalContrast to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERPOINT"
```

### Advanced Installation 

#### Cuda Extensions

```bash
# set the cuda path(change the path to your own cuda location) 
export PATH=/usr/local/cuda-11.1/bin:$PATH
export CUDA_PATH=/usr/local/cuda-11.1
export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
bash setup.sh 
```

#### APEX

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5633f6  
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If your use torch>1.8, please remember to modify the code in ```~/envs/proposalcontrast/lib/python3.6/site-packages/apex/amp/_amp_state.py```:
```
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
```
#### SpConv v1.2
```bash
sudo apt-get install libboost-all-dev
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *  
```

#### SpConv v2
```bash
sudo apt-get install libboost-all-dev
pip install spconv-cu111
```

#### Please prepare the dataset following [WAYMO](WAYMO.md) and start the self-supervised learning of 3D models accroding to [RUN_MODEL](RUN_MODEL.md).