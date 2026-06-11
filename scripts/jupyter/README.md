# Despliegue alternativo: JupyterLab + OpenWebUI

Entorno alternativo al Linux con NIM: pod JupyterLab con disco persistente
(NFS bajo `/home/jovyan`) y B200 con MIG (3g.90gb visible para PyTorch).
El motor habla con OpenWebUI vía `ChatNVIDIA` / `NVIDIAEmbeddings` apuntados a
`https://open-webui.ia.labia.tics/api`, **sin cambios de código** (verificado
por smoke test con `langchain-nvidia-ai-endpoints==0.3.19`, chat sync + async +
embeddings + fallback `langchain-openai`).

Ver `CLAUDE.md` → "Despliegue alternativo: JupyterLab + OpenWebUI" para el
contexto completo.

## Prerequisitos verificados

- Pod JupyterLab con `/home/jovyan` montado en NFS (1 TB, persiste entre
  reinicios del pod). `/opt/conda` está en overlay, **no persiste**.
- B200 MIG 3g.90gb (89 GB), CUDA 13.2, visible desde `torch`.
- OpenWebUI en `https://open-webui.ia.labia.tics/api` con api_key Bearer.
- Modelo chat: `coding-qwen3-coder-next` (32K ctx, sin thinking-mode, ~1s por
  extracción de tripletas sobre chunks de ~5K chars).
- Modelo embeddings: `demo-bge-m3` (1024 dims, ~60 ms).
- PyPI vía `nexus.ia.labia.tics` (proxy interno, ya configurado en el pod).
- GitHub bloqueado por proxy: traer el repo vía GitLab interno (mirror) o
  tarball por la UI.
- `dl.min.io` bloqueado: el binario de MinIO se sube **una vez** por la UI.
- `tmux`/`screen` no disponibles: procesos en background con `nohup`. Datos
  persisten en NFS pero **los procesos mueren al reiniciar el pod** (relanzar
  MinIO manualmente).

## Orden de pasos

1. **Repo en `/home/jovyan/lirag`** — `git clone` desde GitLab interno o
   tarball por la UI.
2. **`bash scripts/jupyter/01_setup_venv.sh`** — crea venv persistente en
   `/home/jovyan/lirag_venv` con `requirements.txt` + `python-igraph` +
   `langchain-openai` (fallback). Activar con
   `source /home/jovyan/lirag_venv/bin/activate`.
3. **Binario de MinIO** — desde tu máquina descarga
   `https://dl.min.io/server/minio/release/linux-amd64/minio` (≈ 110 MB) y
   súbelo por la UI de JupyterLab a `/home/jovyan/minio`. Luego:
   ```
   mkdir -p /home/jovyan/bin
   mv /home/jovyan/minio /home/jovyan/bin/minio
   chmod +x /home/jovyan/bin/minio
   ```
4. **`bash scripts/jupyter/02_minio_run.sh`** — lanza MinIO en background
   sobre `/home/jovyan/minio_data` (datos persisten en NFS).
5. **`python scripts/jupyter/03_create_bucket.py`** — crea el bucket
   `lakehouse` (idempotente).
6. **`.env`** — `cp sandbox_mteb/env.example.jupyter sandbox_mteb/.env` y pega
   la `NVIDIA_API_KEY` (no commitear: `.env` está en `.gitignore`).
7. **Colocar la colección de LI_AD** en `lakehouse/admin/collections/{collection_id}/`.
   Como el MinIO de producción no es alcanzable desde Jupyter, hay que
   exportar la colección a tarball en tu Linux, subirla por la UI y replicarla
   al MinIO local con `boto3`. Script pendiente.
8. **`probe.py`** contra la colección colocada (snippet en `FAST_NOTES.md` §D).
9. **Indexación**: mini-script que llama a `load_collection` +
   `LightRAGRetriever.index_documents`. Pendiente.

## Lo que NO está automatizado y por qué

- **Descarga del binario MinIO**: `dl.min.io` bloqueado. Upload UI manual,
  una sola vez (queda en NFS).
- **Transferencia de la colección**: MinIO de producción no alcanzable desde
  Jupyter. Tarball + upload UI hasta que el flujo se automatice.
- **Recuperación tras reinicio del pod**: relanzar `02_minio_run.sh`
  manualmente. Los datos persisten, el proceso no.
