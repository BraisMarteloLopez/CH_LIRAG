"""Exporta una coleccion del MinIO de PRODUCCION a un tarball.

CORRE EN EL ENTORNO LINUX (el que alcanza el MinIO de produccion), NO en
Jupyter: desde el pod ese MinIO no es ruteable (verificado). El tarball
resultante se sube a Jupyter por la UI y se importa con
`05_import_collection.py`.

Sigue el contrato (INGESTION_CONTRACT.md): entra por collection.json y
descarga exactamente las parts que el manifest enumera — no hace glob.

Uso (desde el entorno Linux, con boto3 instalado):
    python scripts/jupyter/04_export_collection.py col_XXXX_yyyy \
        [--endpoint http://172.30.79.110:9000] [--access-key ...] \
        [--secret-key ...] [--bucket lakehouse] [--prefix admin/collections] \
        [--out ./collection_export]

Defaults de conexion: los del MinIO de produccion en env.example; tambien
respeta MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY del entorno.
"""
import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import boto3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("collection_id")
    ap.add_argument("--endpoint", default=os.environ.get("MINIO_ENDPOINT", "http://172.30.79.110:9000"))
    ap.add_argument("--access-key", default=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"))
    ap.add_argument("--secret-key", default=os.environ.get("MINIO_SECRET_KEY", "minio123"))
    ap.add_argument("--bucket", default="lakehouse")
    ap.add_argument("--prefix", default="admin/collections")
    ap.add_argument("--out", default="./collection_export")
    args = ap.parse_args()

    cid = args.collection_id
    s3 = boto3.client(
        "s3", endpoint_url=args.endpoint,
        aws_access_key_id=args.access_key, aws_secret_access_key=args.secret_key,
    )

    out_root = Path(args.out)
    col_dir = out_root / cid
    col_dir.mkdir(parents=True, exist_ok=True)

    # 1) Manifest (punto de entrada del contrato)
    manifest_key = f"{args.prefix}/{cid}/collection.json"
    try:
        body = s3.get_object(Bucket=args.bucket, Key=manifest_key)["Body"].read()
    except Exception as e:
        print(f"ERROR: no se pudo leer el manifest '{manifest_key}': {e}")
        return 1
    manifest = json.loads(body)
    (col_dir / "collection.json").write_bytes(body)
    parts = manifest.get("parts", [])
    print(f"manifest OK: collection={cid} generation={manifest.get('generation')} "
          f"num_chunks={manifest.get('num_chunks')} parts={len(parts)}")
    if not parts:
        print("ERROR: manifest sin 'parts' — nada que exportar")
        return 1

    # 2) Parts enumeradas por el manifest
    total = 0
    for part in parts:
        rel = part["path"]  # p.ej. chunks/{stem}.parquet
        dest = col_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        key = f"{args.prefix}/{cid}/{rel}"
        s3.download_file(args.bucket, key, str(dest))
        total += 1
        print(f"  [{total}/{len(parts)}] {rel} ({dest.stat().st_size} bytes)")

    # 3) Tarball
    tgz = out_root / f"{cid}.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(col_dir, arcname=cid)
    print(f"\nexport completo: {tgz} ({tgz.stat().st_size // 1024} KiB)")
    print("siguiente paso: subir el .tgz a /home/jovyan/ via la UI de JupyterLab "
          "y ejecutar 05_import_collection.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
