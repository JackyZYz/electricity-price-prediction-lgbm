#!/bin/bash
set -e

# cd 到项目根目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."

# 激活虚拟环境（如需要可取消注释）
# source .venv/bin/activate

python -m src.pipeline.train_pipeline
