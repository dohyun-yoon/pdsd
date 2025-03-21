#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -L jobenv=singularity
#PJM -j

module load singularity

singularity exec \
  --bind $HOME \
  --nv $HOME/pytorch_2.0.1-cuda11.7-cudnn8-runtime.sif \
  bash -c "
  source ${HOME}/.venv/bin/activate
  now=$(date "+%Y%m%d%H%M")
  echo Running in: $HOSTNAME
  echo version: ${now}
  CUDA_LAUNCH_BLOCKING=1 python3 train.py
  "