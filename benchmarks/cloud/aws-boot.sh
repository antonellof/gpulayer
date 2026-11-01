#!/bin/bash
# First-boot install for AWS GPU instances (user-data, written by aws.sh `up` with @REF@ set to
# the current commit): uv, Python 3.12, PyTorch, Lightning and that commit of GPUlayer, in
# /home/ubuntu/venv, with the checkout in /home/ubuntu/gpulayer. Writes ~/.gpulayer-ready when
# done; the log is ~/gpulayer-setup.log. No secrets are passed here.
cat > /home/ubuntu/setup.sh <<'SETUP'
set -euxo pipefail
cd "$HOME"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone -q https://github.com/antonellof/gpulayer.git
git -C gpulayer checkout -q @REF@
uv venv -q --python 3.12 venv
uv pip install -q --python venv/bin/python -e "./gpulayer[dev,examples]" lightning
venv/bin/python -c "import torch, lightning, gpulayer; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'lightning', lightning.__version__, 'gpulayer', gpulayer.__version__)"
touch "$HOME/.gpulayer-ready"
SETUP
chown ubuntu:ubuntu /home/ubuntu/setup.sh
su - ubuntu -c 'bash setup.sh > gpulayer-setup.log 2>&1'
