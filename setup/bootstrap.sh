#!/usr/bin/env bash
set -euo pipefail
root="$HOME/robot-transfer/bowl"
mkdir -p "$root/logs" "$root/tools" "$root/models"
exec > >(tee -a "$root/logs/bootstrap.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$root/logs/bootstrap.exit"' EXIT
date -u
sudo -n apt-get update
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg git-lfs python3.12-venv
python3 -m venv "$root/tools/uv"
"$root/tools/uv/bin/python" -m pip install 'uv==0.13.4'
"$root/tools/uv/bin/uv" --version
for spec in \
  'Isaac-GR00T https://github.com/NVIDIA/Isaac-GR00T.git dac54b313160c201c450f97df0fd74ebd379c892' \
  'RoboLab https://github.com/NVlabs/RoboLab.git b6cfcf6ccd2113c4cd1216d35b33024a7d5691da'; do
  read -r name url revision <<< "$spec"
  if [ ! -d "$root/$name/.git" ]; then
    mkdir -p "$root/$name"
    git -C "$root/$name" init
    git -C "$root/$name" remote add origin "$url"
    git -C "$root/$name" fetch --depth 1 origin "$revision"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$root/$name" checkout --detach FETCH_HEAD
  fi
  test "$(git -C "$root/$name" rev-parse HEAD)" = "$revision"
  git -C "$root/$name" status --short
done
date -u
echo BOOTSTRAP_COMPLETE
