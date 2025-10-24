#!/bin/bash
cd frontend_py && source .venv/bin/activate && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib && \
python main.py