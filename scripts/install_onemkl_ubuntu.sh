#!/usr/bin/env bash
set -euo pipefail

# Install Intel oneMKL development package on Ubuntu/Pop!_OS.
# Requires sudo privileges.
#
# This script configures Intel's oneAPI apt repository and installs oneMKL.
# Use --dry-run to print commands without executing privileged operations.

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as a normal user with sudo rights, not as root."
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required but not found."
  exit 1
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    eval "$*"
  fi
}

tmp_dir="$(mktemp -d)"
cleanup() { rm -rf "${tmp_dir}"; }
trap cleanup EXIT

key_url="https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB"
keyring_path="/usr/share/keyrings/intel-oneapi-archive-keyring.gpg"
source_file="/etc/apt/sources.list.d/intel-oneapi.list"
repo_line="deb [signed-by=${keyring_path}] https://apt.repos.intel.com/oneapi all main"

echo "Fetching Intel oneAPI apt key..."
curl -fL -o "${tmp_dir}/intel-oneapi-key.pub" "${key_url}"

echo "Configuring signed apt source..."
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] sudo install -d -m 0755 /usr/share/keyrings"
  echo "[dry-run] gpg --dearmor < ${tmp_dir}/intel-oneapi-key.pub | sudo tee ${keyring_path} >/dev/null"
  echo "[dry-run] echo '${repo_line}' | sudo tee ${source_file} >/dev/null"
else
  sudo install -d -m 0755 /usr/share/keyrings
  gpg --dearmor < "${tmp_dir}/intel-oneapi-key.pub" | sudo tee "${keyring_path}" >/dev/null
  echo "${repo_line}" | sudo tee "${source_file}" >/dev/null
fi

echo "Updating apt metadata..."
run_cmd "sudo apt-get update -y"

echo "Installing oneMKL development package..."
run_cmd "sudo apt-get install -y intel-oneapi-mkl-devel"

echo
echo "Installation complete."
echo "Now run:"
echo "  source /opt/intel/oneapi/setvars.sh"
echo "  cd /home/jason/Desktop/osqp_related/osqp"
echo "  cmake -S . -B build_mkl_probe -DOSQP_ALGEBRA_BACKEND=mkl"
