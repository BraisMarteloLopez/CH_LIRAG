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

## Arranque diario (el pod se redespliega cada dia)

El pod JupyterLab se recrea a diario: los **procesos** y `/opt/conda` se
pierden; `/home/jovyan` (NFS) persiste. Al abrir el entorno, lanzar:

```
bash /home/jovyan/lirag/scripts/jupyter/00_daily_start.sh
```

Es idempotente: verifica GPU, venv, MinIO (lo relanza si esta caido),
OpenWebUI alcanzable y `.env` presente. Si todo esta verde, imprime el
comando de activacion del venv. Si algo falta, dice exactamente que script
ejecutar para arreglarlo.

## Setup inicial (una sola vez)

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

## Traer e indexar una coleccion

7. **Exportar del MinIO de produccion** (corre en el ENTORNO LINUX, no en
   Jupyter — desde el pod ese MinIO no es ruteable):
   ```
   python scripts/jupyter/04_export_collection.py <collection_id>
   ```
   Produce `collection_export/<collection_id>.tgz`. Subirlo a `/home/jovyan/`
   por la UI de JupyterLab.
8. **Importar al MinIO local** (en Jupyter, venv activo):
   ```
   python scripts/jupyter/05_import_collection.py /home/jovyan/<collection_id>.tgz
   ```
   Sube las parts primero y `collection.json` el ultimo (manifest-as-commit,
   misma semantica que LI_AD).
9. **Indexar / construir el KG** (en Jupyter, venv activo, desde la raiz del
   repo):
   ```
   python scripts/jupyter/06_index_collection.py <collection_id> [--query "..."]
   ```
   Carga via `load_collection` (valida el contrato), construye el KG +
   3 VDBs, imprime stats (KGStats + extractor) y, con `--query`, hace un
   retrieval de sanidad. No borra indices al salir: el KG cache
   (`KG_CACHE_DIR`) queda reutilizable.

## Lo que NO está automatizado y por qué

- **Descarga del binario MinIO**: `dl.min.io` bloqueado. Upload UI manual,
  una sola vez (queda en NFS).
- **Transferencia de la colección**: MinIO de producción no alcanzable desde
  Jupyter → export (Linux) + upload UI + import (Jupyter), scripts 04/05.
- **Arranque tras redespliegue del pod**: `00_daily_start.sh` manual al abrir
  el entorno (no hay hook de arranque del pod a nuestro alcance).

> Estos scripts se validan solo en el entorno del usuario (claude_code no
> tiene acceso a Jupyter/MinIO/OpenWebUI). Si un script falla, pegar la
> salida completa en la sesion para diagnostico.
