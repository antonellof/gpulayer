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
#   GL_AWS_SPOT=1 benchmarks/cloud/aws.sh up ...         # Spot instead of on-demand
#   GL_AWS_AZ=us-east-1b benchmarks/cloud/aws.sh up ...  # one zone, for NCCL across machines
#   benchmarks/cloud/aws.sh allow 203.0.113.7 [8470]     # let one outside IP reach a GPUlayer
#                                                        # server on these instances (TLS port)
#   benchmarks/cloud/aws.sh revoke 203.0.113.7 [8470]
#   benchmarks/cloud/aws.sh ssh gl-gpu4 'nvidia-smi'      # run a command; also: ip, private-ip
#
# Instances use the AWS Deep Learning Base OSS NVIDIA driver AMI (Ubuntu 22.04), a VPC of their
# own named gpulayer-bench (one subnet per zone, internet gateway; the default VPC is not used
# or changed), a key pair and a security group named gpulayer-bench (SSH from this machine's
# public IP, all traffic between members, for NCCL across machines). On first boot
# (aws-boot.sh, user-data) each instance installs uv, Python 3.12, PyTorch, Lightning and the
# current commit of GPUlayer (pushed to GitHub) into ~/venv; `ready NAME...` waits for it.
# GL_AWS_BOOT=0 skips that. `down-network` deletes the VPC when no instance uses it. Instances are tagged
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

AZ=${GL_AWS_AZ:-${REGION}c}

tagged() { aws ec2 "describe-$1" "${A[@]}" --filters "Name=tag:Name,Values=$2" \
  --query "$3" --output text; }

network() {  # a VPC of its own (not the default VPC): subnet per zone, internet gateway, route
  local vpc igw rt sub n
  vpc=$(tagged vpcs gpulayer-bench "Vpcs[0].VpcId")
  if [ "$vpc" = None ]; then
    vpc=$(aws ec2 create-vpc "${A[@]}" --cidr-block 10.77.0.0/16 --query Vpc.VpcId --output text \
      --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=gpulayer-bench},{Key=purpose,Value=gpulayer-bench}]")
    aws ec2 modify-vpc-attribute "${A[@]}" --vpc-id "$vpc" --enable-dns-hostnames >/dev/null
    igw=$(aws ec2 create-internet-gateway "${A[@]}" --query InternetGateway.InternetGatewayId --output text \
      --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=gpulayer-bench}]")
    aws ec2 attach-internet-gateway "${A[@]}" --vpc-id "$vpc" --internet-gateway-id "$igw"
    rt=$(aws ec2 create-route-table "${A[@]}" --vpc-id "$vpc" --query RouteTable.RouteTableId --output text \
      --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=gpulayer-bench}]")
    aws ec2 create-route "${A[@]}" --route-table-id "$rt" --destination-cidr-block 0.0.0.0/0 \
      --gateway-id "$igw" >/dev/null
  fi
  sub=$(tagged subnets "gpulayer-bench-$AZ" "Subnets[0].SubnetId")
  if [ "$sub" = None ]; then
    n=$(( $(printf '%d' "'${AZ: -1}") - 96 ))  # a=1, b=2, ...
    sub=$(aws ec2 create-subnet "${A[@]}" --vpc-id "$vpc" --availability-zone "$AZ" \
      --cidr-block "10.77.$n.0/24" --query Subnet.SubnetId --output text \
      --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=gpulayer-bench-$AZ}]")
    aws ec2 modify-subnet-attribute "${A[@]}" --subnet-id "$sub" --map-public-ip-on-launch
    rt=$(tagged route-tables gpulayer-bench "RouteTables[0].RouteTableId")
    aws ec2 associate-route-table "${A[@]}" --route-table-id "$rt" --subnet-id "$sub" >/dev/null
  fi
  echo "$vpc $sub"
}

setup() {  # key pair, network and security group, created once; prints "sg subnet"
  if [ ! -f "$KEY" ]; then ssh-keygen -q -t ed25519 -N '' -f "$KEY" -C gpulayer-bench; fi
  aws ec2 describe-key-pairs "${A[@]}" --key-names gpulayer-bench >/dev/null 2>&1 ||
    aws ec2 import-key-pair "${A[@]}" --key-name gpulayer-bench \
      --public-key-material "fileb://$KEY.pub" >/dev/null
  local vpc sub sg me
  read -r vpc sub < <(network)
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
  echo "$sg $sub"
}

cmd=${1:-}; shift || true
case "$cmd" in
  up)
    name=$1 type=${2:-g4dn.12xlarge}
    read -r sg sub < <(setup)
    market=()
    boot=()
    if [ "${GL_AWS_BOOT:-1}" = 1 ]; then
      # Install uv, Python, PyTorch, Lightning and this commit of GPUlayer on first boot.
      ref=$(git -C "$ROOT" rev-parse HEAD)
      bootfile=$(mktemp)
      sed "s/@REF@/$ref/" "$ROOT/benchmarks/cloud/aws-boot.sh" > "$bootfile"
      boot=(--user-data "file://$bootfile")
    fi
    if [ "${GL_AWS_SPOT:-}" = 1 ]; then
      market+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
    fi
    aws ec2 run-instances "${A[@]}" --image-id "$(ami)" --instance-type "$type" "${market[@]}" \
      --key-name gpulayer-bench --security-group-ids "$sg" --subnet-id "$sub" "${boot[@]}" \
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
  ssh)
    vm=$1; shift; on "$vm" "$@" ;;
  ready)  # wait until the boot install finished on each instance
    for vm in "$@"; do
      for _ in $(seq 1 90); do on "$vm" test -f .gpulayer-ready 2>/dev/null && break; sleep 10; done
      on "$vm" 'test -f .gpulayer-ready && tail -1 gpulayer-setup.log' || echo "$vm: not ready"
    done ;;
  down-network)  # delete the VPC created by `network` (after `down` of every instance)
    vpc=$(tagged vpcs gpulayer-bench "Vpcs[0].VpcId")
    [ "$vpc" = None ] && exit 0
    for sub in $(aws ec2 describe-subnets "${A[@]}" --filters "Name=vpc-id,Values=$vpc" --query "Subnets[].SubnetId" --output text); do
      aws ec2 delete-subnet "${A[@]}" --subnet-id "$sub"; done
    for sg in $(aws ec2 describe-security-groups "${A[@]}" --filters "Name=vpc-id,Values=$vpc" Name=group-name,Values=gpulayer-bench --query "SecurityGroups[].GroupId" --output text); do
      aws ec2 delete-security-group "${A[@]}" --group-id "$sg"; done
    rt=$(tagged route-tables gpulayer-bench "RouteTables[0].RouteTableId")
    [ "$rt" != None ] && aws ec2 delete-route-table "${A[@]}" --route-table-id "$rt"
    igw=$(tagged internet-gateways gpulayer-bench "InternetGateways[0].InternetGatewayId")
    if [ "$igw" != None ]; then
      aws ec2 detach-internet-gateway "${A[@]}" --internet-gateway-id "$igw" --vpc-id "$vpc"
      aws ec2 delete-internet-gateway "${A[@]}" --internet-gateway-id "$igw"
    fi
    aws ec2 delete-vpc "${A[@]}" --vpc-id "$vpc" && echo "deleted $vpc" ;;
  ip)
    ip_of "$1" ;;
  private-ip)
    aws ec2 describe-instances "${A[@]}" --filters "Name=tag:Name,Values=$1" \
      "Name=instance-state-name,Values=running" --query "Reservations[].Instances[].PrivateIpAddress" --output text ;;
  allow|revoke)
    ip=$1 port=${2:-8470}
    sg=$(aws ec2 describe-security-groups "${A[@]}" --filters Name=group-name,Values=gpulayer-bench \
      --query "SecurityGroups[0].GroupId" --output text)
    verb=authorize; [ "$cmd" = revoke ] && verb=revoke
    aws ec2 "$verb-security-group-ingress" "${A[@]}" --group-id "$sg" --protocol tcp \
      --port "$port" --cidr "$ip/32" >/dev/null && echo "$cmd $ip:$port" ;;
  down)
    for vm in "$@"; do
      id=$(id_of "$vm"); [ -n "$id" ] && aws ec2 terminate-instances "${A[@]}" --instance-ids $id \
        --query "TerminatingInstances[].[InstanceId,CurrentState.Name]" --output text
    done ;;
  *)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
