#!/usr/bin/env bash

# Install the diff.lab, diff.lab_assets, and diff.lab_tasks packages in editable mode

# Exit if any command fails
set -e

# Isaac Lab path (default in docker). Optionally export ISAACLAB_PATH before running.
ISAACLAB_PATH=${ISAACLAB_PATH:-/workspace/isaaclab}
# Python command; override via PYTHON_BIN="</path/to/python>" or "...</isaaclab.sh -p"
PYTHON_BIN=${PYTHON_BIN:-${ISAACLAB_PATH}/isaaclab.sh -p}
# Split into array to allow commands with flags (e.g., isaaclab.sh -p)
IFS=' ' read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

# Pip flags: disable build isolation & online indexes by default for offline installs.
PIP_FLAGS=${PIP_FLAGS:---no-build-isolation --no-index}
PIP_FIND_LINKS=""
PIP_PREBUNDLE_DIR=$(find "${ISAACLAB_PATH}" -maxdepth 5 -type d -name "pip_prebundle" -print -quit 2>/dev/null || true)
if [[ -n "${PIP_PREBUNDLE_DIR}" ]]; then
    PIP_FLAGS="${PIP_FLAGS} --find-links ${PIP_PREBUNDLE_DIR}"
fi
IFS=' ' read -r -a PIP_ARGS <<< "${PIP_FLAGS}"

# Base path to the extensions
BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/extensions"

#
echo "Installing all packages from ${BASE_PATH}..."

# Install each package in editable mode
echo "Using python: ${PYTHON_CMD[*]}"
echo "Using pip flags: ${PIP_FLAGS}"
echo "Installing diff.lab..."
"${PYTHON_CMD[@]}" -m pip install "${PIP_ARGS[@]}" -e "${BASE_PATH}/diff.lab"

echo "Installing diff.lab_assets..."
"${PYTHON_CMD[@]}" -m pip install "${PIP_ARGS[@]}" -e "${BASE_PATH}/diff.lab_assets"

echo "Installing diff.lab_tasks..."
"${PYTHON_CMD[@]}" -m pip install "${PIP_ARGS[@]}" -e "${BASE_PATH}/diff.lab_tasks[all]"

# echo "Installing diff.lab_apps..."
# pip install -e "${BASE_PATH}/diff.lab_apps"

echo "All packages have been installed in editable mode."
