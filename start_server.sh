#!/bin/bash
# cd server && source .venv/bin/activate && python main.py --pipeline controlnetSDTurbot2i --taesd --sfast # --use-upscaler #--onediff #--sfast #--compel #--torch-compile #--sfast # --torch_compile # --sfast

# cd server && source .venv/bin/activate && python main.py --pipeline img2imgStreamDiffusion --taesd --default-curation-index 11 #--use-upscaler #--onediff #--sfast #--compel #--torch-compile #--sfast

cd server && source .venv/bin/activate && python main.py --pipeline img2imgStreamDiffusionXL --taesd --use-upscaler --default-curation-index 21 "$@"  #--torch-compile #--onediff #--sfast #--compel #--torch-compile #--sfast
