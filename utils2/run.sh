#!/bin/bash
# VB5K Boundary Marker 실행 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv/bin/python"

if [ ! -f "$VENV" ]; then
    echo "ERROR: venv not found at $VENV"
    exit 1
fi

cd "$SCRIPT_DIR"
"$VENV" boundary_marker_gui.py
