# export HF_HOME="/media/monsterdrive/models"
python main.py --pipeline controlnetSDTurbot2i --taesd --sfast #--onediff #--sfast #--compel #--torch-compile #--sfast # --torch_compile # --sfast
# uv run main.py --pipeline predict # --torch_compile # --sfast

# uv run main.py --pipeline controlnetTRTSDTurbot2i # --torch_compile # --sfast
# uv run main.py --pipeline img2img
# uv run main.py --pipeline img2imgStreamDiffusion --taesd --tensorrt # --sfast
# uv run main.py --pipeline controlnetSDTurbo
# uv run main.py --pipeline img2imgSDTurbo

python main.py --pipeline controlnetSDTurbot2i --taesd --sfast