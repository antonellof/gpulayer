#!/usr/bin/env bash
# Run GPUlayer's GPU validation on AWS EC2 (the counterpart of gcp.sh for NVIDIA machines).
#
#   export AWS_REGION=us-east-1                 # optional, default us-east-1
#   benchmarks/cloud/aws.sh up gl-gpu4 [g4dn.12xlarge]   # 4x T4; g4dn.xlarge for 1x T4
#   benchmarks/cloud/aws.sh push gl-gpu4                 # copy this checkout
#   benchmarks/cloud/aws.sh validate gl-gpu4             # scripts/validate_gpu.sh (runs
#                                                        # validate_multigpu.py with 2+ GPUs)
#   benchmarks/cloud/aws.sh fetch gl-gpu4 benchmarks/results/<date>-aws/
#   benchmarks/cloud/aws.sh down gl-gpu4 [more...]       # terminate
#
# Instances use the AWS Deep Learning Base OSS NVIDIA driver AMI (Ubuntu 22.04), the default
# VPC, a key pair and a security group named gpulayer-bench (SSH from this machine's public IP,
# all traffic between members, for NCCL across machines). Instances are tagged
# purpose=gpulayer-bench; `down` terminates them. Results stay on the instance until `fetch`.
set -euo pipefail

REGION=${AWS_REGION:-us-east-1}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
KEY=${GL_AWS_KEY:-$HOME/.ssh/gpulayer-bench}
A=(--region "$REGION")

ami() { aws ssm get-parameter "${A[@]}" --query Parameter.Value --output text \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id; }
ip_of() { aws ec2 describe-instances "${A[@]}" --filters "Name=tag:Name,Values=$1" \
  "Name=instance-state-name,Values=running" --query "Reservations[].Instances[].PublicIpAddress" --output text; }
id_of() { aws ec2 describe-instances "${A[@]}" --filters "Name=tag:Name,Values=$1" \
  "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query "Reservations[].Instances[].InstanceId" --output text; }
on() { local vm=$1; shift; ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
  "ubuntu@$(ip_of "$vm")" "$@"; }

setup() {  # key pair and security group, created once
  if [ ! -f "$KEY" ]; then ssh-keygen -q -t ed25519 -N '' -f "$KEY" -C gpulayer-bench; fi
  aws ec2 describe-key-pairs "${A[@]}" --key-names gpulayer-bench >/dev/null 2>&1 ||
    aws ec2 import-key-pair "${A[@]}" --key-name gpulayer-bench \
      --public-key-material "fileb://$KEY.pub" >/dev/null
  local vpc sg me
  vpc=$(aws ec2 describe-vpcs "${A[@]}" --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text)
  sg=$(aws ec2 describe-security-groups "${A[@]}" --filters Name=group-name,Values=gpulayer-bench \
    "Name=vpc-id,Values=$vpc" --query "SecurityGroups[0].GroupId" --output text)
  if [ "$sg" = None ]; then
    sg=$(aws ec2 create-security-group "${A[@]}" --group-name gpulayer-bench --vpc-id "$vpc" \
      --description "GPUlayer benchmarks" --query GroupId --output text)
    aws ec2 authorize-security-group-ingress "${A[@]}" --group-id "$sg" \
      --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$sg}]" >/dev/null
  fi
  me=$(curl -s https://checkip.amazonaws.com)
  aws ec2 authorize-security-group-ingress "${A[@]}" --group-id "$sg" --protocol tcp --port 22 \
    --cidr "$me/32" >/dev/null 2>&1 || true
  echo "$sg"
}

cmd=${1:-}; shift || true
case "$cmd" in
  up)
    name=$1 type=${2:-g4dn.12xlarge}
    sg=$(setup)
    aws ec2 run-instances "${A[@]}" --image-id "$(ami)" --instance-type "$type" \
      --key-name gpulayer-bench --security-group-ids "$sg" \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name},{Key=purpose,Value=gpulayer-bench}]" \
      --query "Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone]" --output text
    aws ec2 wait instance-running "${A[@]}" --instance-ids "$(id_of "$name")"
    for _ in $(seq 1 30); do on "$name" nvidia-smi -L 2>/dev/null && exit 0; sleep 10; done
    echo "no ssh or driver on $name" >&2; exit 1 ;;
  push)
    vm=$1 tgz=$(mktemp -t gpulayer-src).tgz
    (cd "$ROOT" && git ls-files | grep -v '^site/\|^docs/assets/dashboard/' ; echo site/src/lib/version.ts) \
      | (cd "$ROOT" && COPYFILE_DISABLE=1 tar czf "$tgz" -T -)
    scp -i "$KEY" -o StrictHostKeyChecking=accept-new "$tgz" "ubuntu@$(ip_of "$vm"):gpulayer-src.tgz"
    on "$vm" 'rm -rf gpulayer && mkdir gpulayer && tar xzf gpulayer-src.tgz -C gpulayer 2>/dev/null; ls gpulayer | head -3' ;;
  validate)
    on "$1" 'cd gpulayer && n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l);
      [ "$n" = 0 ] || { echo "GPU busy; not starting a timed run"; exit 1; };
      bash scripts/validate_gpu.sh > validate_gpu.log 2>&1; grep -E "^CHECK|SUMMARY|ALL STEPS|FAILED" validate_gpu.log | tail -30' ;;
  fetch)
    vm=$1 dest=$2; mkdir -p "$dest"
    scp -i "$KEY" "ubuntu@$(ip_of "$vm"):gpulayer/validate_gpu.log" "$dest/" || true
    on "$vm" 'nvidia-smi; nvidia-smi topo -m' > "$dest/nvidia-smi.txt" || true
    on "$vm" 'cd gpulayer && $HOME/.local/bin/uv pip freeze --python .venv-validate/bin/python' > "$dest/pip-freeze.txt" || true
    for f in "$dest"/*; do [ -f "$f" ] && sed -i.bak "s#/home/[A-Za-z0-9_.-]*#/home/user#g" "$f" && rm -f "$f.bak"; done
    ls -la "$dest" ;;
  down)
    for vm in "$@"; do
      id=$(id_of "$vm"); [ -n "$id" ] && aws ec2 terminate-instances "${A[@]}" --instance-ids $id \
        --query "TerminatingInstances[].[InstanceId,CurrentState.Name]" --output text
    done ;;
  *)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
