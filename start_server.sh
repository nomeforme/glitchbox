# export HF_HOME="/media/monsterdrive/models"
# uv run server/main.py --pipeline controlnetSDTurbot2i # --sfast
# uv run server/main.py --pipeline img2img
# uv run server/main.py --pipeline img2imgStreamDiffusion --taesd --tensorrt # --sfast
# python server/main.py --pipeline controlnetSDTurboi2i 
uv run server/main.py --pipeline controlnetSDTurbo