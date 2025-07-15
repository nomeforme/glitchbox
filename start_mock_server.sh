#!/bin/bash
cd server && source .venv/bin/activate && python main.py --pipeline controlnetSDTurbot2i_mock --taesd --sfast #--onediff #--sfast #--compel #--torch-compile #--sfast # --torch_compile # --sfast
