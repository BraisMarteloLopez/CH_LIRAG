#!/usr/bin/env bash
# Arranque diario del entorno JupyterLab.
#
# El pod se redespliega cada dia: los PROCESOS y /opt/conda se pierden;
# /home/jovyan (NFS) persiste. Este script es idempotente — lanzarlo al
# abrir el entorno deja todo operativo o reporta que falta.
#
# Uso:  bash /home/jovyan/lirag/scripts/jupyter/00_daily_start.sh

set -uo pipefail  # sin -e: queremos ejecutar TODOS los checks y reportar

REPO=/home/jovyan/lirag
VENV=/home/jovyan/lirag_venv
FAIL=0

say() { printf "  %-22s %s\n" "$1" "$2"; }

echo "== daily start: $(date '+%Y-%m-%d %H:%M') =="

# --- 1) GPU (informativo; el motor no la usa directamente) ---
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    say "GPU" "OK ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))"
else
    say "GPU" "no visible (no bloqueante)"
fi

# --- 2) venv persistente (puede romperse si el pod cambia de imagen/python) ---
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import igraph, chromadb, boto3" >/dev/null 2>&1; then
    say "venv" "OK ($VENV)"
else
    say "venv" "ROTO o ausente -> bash $REPO/scripts/jupyter/01_setup_venv.sh"
    FAIL=1
fi

# --- 3) MinIO (proceso muere en cada redespliegue; datos persisten en NFS) ---
health() {
    curl -sS --max-time 4 -o /dev/null -w '%{http_code}' \
        http://127.0.0.1:9000/minio/health/live 2>/dev/null
}
if [ "$(health)" = "200" ]; then
    say "MinIO" "ya corriendo"
else
    echo "  MinIO caido; relanzando..."
    bash "$REPO/scripts/jupyter/02_minio_run.sh" || true
    if [ "$(health)" = "200" ]; then
        say "MinIO" "arrancado"
    else
        say "MinIO" "FALLO -> tail -30 /home/jovyan/minio_logs/minio.log"
        FAIL=1
    fi
fi

# --- 4) OpenWebUI alcanzable (sin api_key: 401/403 = alcanzable) ---
code=$(curl -sS --max-time 6 -o /dev/null -w '%{http_code}' \
    https://open-webui.ia.labia.tics/api/models 2>/dev/null)
case "$code" in
    200|401|403) say "OpenWebUI" "alcanzable (HTTP $code)" ;;
    *)           say "OpenWebUI" "NO alcanzable (HTTP ${code:-sin respuesta})"; FAIL=1 ;;
esac

# --- 5) .env presente ---
if [ -f "$REPO/sandbox_mteb/.env" ]; then
    say ".env" "presente"
else
    say ".env" "FALTA -> cp $REPO/sandbox_mteb/env.example.jupyter $REPO/sandbox_mteb/.env (+ api_key)"
    FAIL=1
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "todo listo. Para trabajar:"
    echo "  source $VENV/bin/activate && cd $REPO"
else
    echo "hay checks fallidos (ver arriba). Corrige y relanza este script."
    exit 1
fi
