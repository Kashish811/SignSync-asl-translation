set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.env"
cd "$SS_HOME"
[ -x "$VENV/bin/python" ] && export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
