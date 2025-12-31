#!/bin/bash
ENV_NAME="phi_nc_env"
YML_FILE="phi_nc_env.yml"

# 1. Remove old env
if conda info --envs | grep -q "$ENV_NAME"; then
    conda env remove -n $ENV_NAME
fi

# 2. Create env from YAML (comment out pip section manually first)
conda env create -f $YML_FILE

# 3. Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 4. Install torch + torchvision
if command -v nvidia-smi &> /dev/null; then
    pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch==2.5.1 torchvision
fi

# 5. Install remaining pip packages
pip install -r requirements_fixed.txt
