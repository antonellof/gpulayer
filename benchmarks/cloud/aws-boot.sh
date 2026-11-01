#!/bin/bash
# First-boot install for AWS GPU instances (user-data, written by aws.sh `up` with @REF@ set to
# the current commit): uv, Python 3.12, PyTorch, Lightning and that commit of GPUlayer, in
# /home/ubuntu/venv, with the checkout in /home/ubuntu/gpulayer. Writes ~/.gpulayer-ready when
# done; the log is ~/gpulayer-setup.log. No secrets are passed here.
# The AMI's /etc/profile.d/dlami.sh points LD_LIBRARY_PATH at the system CUDA, whose cuDNN
# clashes with the PyTorch wheel's (convolutions fail with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED).
# Clear it for login shells (after dlami.sh) and in .bashrc right after the AMI's own line that
# sources dlami.sh for non-interactive shells (ssh commands).
echo 'unset LD_LIBRARY_PATH' > /etc/profile.d/zz-gpulayer.sh
sed -i '/profile.d\/dlami.sh/a unset LD_LIBRARY_PATH' /home/ubuntu/.bashrc
cat > /home/ubuntu/setup.sh <<'SETUP'
set -euxo pipefail
cd "$HOME"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone -q https://github.com/antonellof/gpulayer.git
git -C gpulayer checkout -q @REF@
uv venv -q --python 3.12 venv
uv pip install -q --python venv/bin/python -e "./gpulayer[dev,examples]" lightning
env -u LD_LIBRARY_PATH venv/bin/python -c "import torch; c = torch.nn.Conv2d(3, 8, 3).cuda(); c(torch.randn(1, 3, 8, 8, device='cuda'))"
venv/bin/python -c "import torch, lightning, gpulayer; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'lightning', lightning.__version__, 'gpulayer', gpulayer.__version__)"
touch "$HOME/.gpulayer-ready"
SETUP
chown ubuntu:ubuntu /home/ubuntu/setup.sh
su - ubuntu -c 'bash setup.sh > gpulayer-setup.log 2>&1'
