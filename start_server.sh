#!/bin/bash
# cd server && source .venv/bin/activate && python main.py --pipeline controlnetSDTurbot2i --taesd --sfast #--onediff #--sfast #--compel #--torch-compile #--sfast # --torch_compile # --sfast
#cd server && source .venv/bin/activate && python main.py --pipeline img2imgStreamDiffusion --taesd --use-upscaler #--onediff #--sfast #--compel #--torch-compile #--sfast # --torch_compile # --sfast
# cd server && source .venv/bin/activate && python main.py --pipeline txt2imgStreamDiffusion  --taesd --use-upscaler #--onediff #--sfast #--compel #--torch-compile #--sfast # --torch_compile # --sfast

cd server && source .venv/bin/activate && python main.py --pipeline img2imgStreamDiffusionXL --taesd #--onediff #--sfast #--compel #--torch-compile #--sfast # --torch_compile # --sfast
#cd server && source .venv/bin/activate && python main.py --pipeline txt2imgStreamDiffusionXL --taesd #--onediff #--sfast #--compel #--torch-compile #--sfast # --torch_compile # --sfast
