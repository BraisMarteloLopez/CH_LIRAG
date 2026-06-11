#!/usr/bin/env bash
# Crea el venv persistente en /home/jovyan/lirag_venv e instala las deps.
#
# Por que venv en NFS y no conda base:
#   /opt/conda esta en overlay (no persiste entre reinicios del pod);
#   /home/jovyan es NFS (persiste).
#
# Requiere: estar en la raiz del repo (donde existe requirements.txt).

set -euo pipefail

VENV=/home/jovyan/lirag_venv

if [ ! -f requirements.txt ]; then
    echo "ERROR: ejecuta desde la raiz del repo (donde esta requirements.txt)"
    exit 1
fi

echo "creando venv en $VENV ..."
python3 -m venv "$VENV"

# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install -q --upgrade pip

# Pin la version probada con OpenWebUI (smoke test verde).
# requirements.txt admite >=0.3.0,<1.0; lo pineamos a 0.3.19 para reproducibilidad.
pip install -q 'langchain-nvidia-ai-endpoints==0.3.19'

# Stack del motor (resto de deps)
pip install -q -r requirements.txt

# Extras: python-igraph (KG, no esta en requirements core) +
# langchain-openai/openai (fallback OpenAI-compat por si se quiere usar)
pip install -q 'python-igraph' 'langchain-openai' 'openai'

echo
echo "venv listo: $VENV"
echo "activar:    source $VENV/bin/activate"
