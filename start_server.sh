# export HF_HOME="."
python3 server/main.py --pipeline controlnetSDTurbot2i --taesd --sfast #--onediff #--sfast #--compel #--torch-compile #--sfast # --torch_compile # --sfast
# uv run server/main.py --pipeline predict # --torch_compile # --sfast

# uv run server/main.py --pipeline controlnetTRTSDTurbot2i # --torch_compile # --sfast
# uv run server/main.py --pipeline img2img
# uv run server/main.py --pipeline img2imgStreamDiffusion --taesd --tensorrt # --sfast
# uv run server/main.py --pipeline controlnetSDTurbo
# uv run server/main.py --pipeline img2imgSDTurbo
