#!/usr/bin/env bash
# NOT executed/validated on this macOS host. No CUDA required for POS modules.
set -euo pipefail
conda env create -f environment-linux.yml
printf '%s\n' 'Activate: conda activate rppg-privacy'
git clone https://github.com/ubicomplab/rPPG-Toolbox.git vendor/rPPG-Toolbox
git -C vendor/rPPG-Toolbox rev-parse HEAD > vendor/toolbox_revision.txt
# For the FULL toolbox use its own environment, not this minimal environment:
# cd vendor/rPPG-Toolbox && bash setup.sh conda
# Inspect upstream setup first; its pinned Python/Torch dependencies may change.
