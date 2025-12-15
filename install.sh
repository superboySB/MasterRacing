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

# Base path to the extensions
BASE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/extensions"

#
echo "Installing all packages from ${BASE_PATH}..."

# Install each package in editable mode
echo "Using python: ${PYTHON_CMD[*]}"
echo "Installing diff.lab..."
"${PYTHON_CMD[@]}" -m pip install -e "${BASE_PATH}/diff.lab"

echo "Installing diff.lab_assets..."
"${PYTHON_CMD[@]}" -m pip install -e "${BASE_PATH}/diff.lab_assets"

echo "Installing diff.lab_tasks..."
"${PYTHON_CMD[@]}" -m pip install -e "${BASE_PATH}/diff.lab_tasks[all]"

# echo "Installing diff.lab_apps..."
# pip install -e "${BASE_PATH}/diff.lab_apps"

echo "All packages have been installed in editable mode."
