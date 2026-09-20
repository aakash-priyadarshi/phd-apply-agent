#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
echo 'Setup complete. Copy .env.example to .env, then run: .venv/bin/python -m streamlit run streamlit_app.py'
