#!/usr/bin/env bash
# Lanza MinIO en background con datos persistentes en NFS.
#
# Prerequisito: binario en /home/jovyan/bin/minio (subido via la UI de JupyterLab,
# ya que dl.min.io esta bloqueado desde el pod). Ver README.md paso 3.
#
# Tras un reinicio del pod, los datos persisten en NFS pero el proceso muere
# (no hay tmux/screen). Hay que relanzar este script.

set -euo pipefail

BIN=/home/jovyan/bin/minio
DATA=/home/jovyan/minio_data
LOGS=/home/jovyan/minio_logs

if [ ! -x "$BIN" ]; then
    echo "ERROR: no encuentro el binario MinIO en $BIN"
    echo
    echo "  Pasos:"
    echo "  1) Descarga https://dl.min.io/server/minio/release/linux-amd64/minio"
    echo "     en tu maquina."
    echo "  2) Subelo a /home/jovyan/ via la UI de JupyterLab."
    echo "  3) mkdir -p /home/jovyan/bin"
    echo "     mv /home/jovyan/minio $BIN"
    echo "     chmod +x $BIN"
    exit 1
fi

mkdir -p "$DATA" "$LOGS"

# Credenciales root del servidor (coinciden con env.example.jupyter).
# Si quieres rotarlas, cambia aqui Y en sandbox_mteb/.env.
export MINIO_ROOT_USER=${MINIO_ROOT_USER:-minioadmin}
export MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-minioadmin}

# Puerto libre?
if (echo > /dev/tcp/127.0.0.1/9000) 2>/dev/null; then
    echo "AVISO: el puerto 9000 ya esta ocupado."
    echo "  Procesos activos:"
    pgrep -af minio || echo "  (no es MinIO; otro servicio en 9000)"
    exit 1
fi

echo "arrancando MinIO ..."
nohup "$BIN" server "$DATA" \
    --address :9000 --console-address :9001 \
    > "$LOGS/minio.log" 2>&1 &
disown

sleep 2

if curl -sS --max-time 5 -o /dev/null -w '%{http_code}' \
        http://127.0.0.1:9000/minio/health/live | grep -q 200; then
    echo "MinIO arriba:"
    echo "  API     http://127.0.0.1:9000"
    echo "  consola http://127.0.0.1:9001"
    echo "  data    $DATA"
    echo "  log     $LOGS/minio.log"
else
    echo "ERROR: MinIO no responde. Ultimas lineas del log:"
    tail -30 "$LOGS/minio.log"
    exit 1
fi
