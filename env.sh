#!/bin/bash

# This script automatically adds the python directory to PYTHONPATH
# Usage: source python-env.sh

# Get the directory where this script is located
# Check ZSH / BASH
if [[ -n "$BASH_VERSION" ]]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [[ -n "$ZSH_VERSION" ]]; then
    SCRIPT_PATH="$0"
else
    SCRIPT_PATH="$0"
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

# Add the python directory to PYTHONPATH
PYTHON_DIR="${SCRIPT_DIR}/python"

# Check if PYTHONPATH already contains our directory
if [[ ":$PYTHONPATH:" != *":${PYTHON_DIR}:"* ]]; then
    # Add to PYTHONPATH without trailing colon
    if [[ -z "$PYTHONPATH" ]]; then
        export PYTHONPATH="${PYTHON_DIR}"
    else
        export PYTHONPATH="${PYTHON_DIR}:${PYTHONPATH}"
    fi
    echo "PYTHONPATH updated to include: ${PYTHON_DIR}"
else
    echo "PYTHONPATH already includes: ${PYTHON_DIR}"
fi

echo "Current PYTHONPATH: $PYTHONPATH"
