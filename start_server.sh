# export HF_HOME="/media/monsterdrive/models"
uv run server/main.py --pipeline controlnetSDTurbot2i --taesd # --torch_compile # --sfast
# uv run server/main.py --pipeline img2img
# uv run server/main.py --pipeline img2imgStreamDiffusion --taesd --tensorrt # --sfast
# uv run server/main.py --pipeline controlnetSDTurbo
# uv run server/main.py --pipeline img2imgSDTurbo