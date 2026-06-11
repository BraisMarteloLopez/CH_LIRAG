"""Importa un tarball de coleccion (creado por 04_export_collection.py) al
MinIO LOCAL de Jupyter.

CORRE EN JUPYTER, con el venv activo y MinIO local arriba (02_minio_run.sh).

Respeta la semantica manifest-as-commit del contrato: sube primero las
parts y `collection.json` EL ULTIMO, igual que hara LI_AD en su ingesta —
un lector nunca ve manifest sin parts completas.

Uso:
    python scripts/jupyter/05_import_collection.py /home/jovyan/col_XXXX_yyyy.tgz \
        [--endpoint http://127.0.0.1:9000] [--access-key minioadmin] \
        [--secret-key minioadmin] [--bucket lakehouse] [--prefix admin/collections]
"""
import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tgz", help="ruta al tarball {collection_id}.tgz")
    ap.add_argument("--endpoint", default="http://127.0.0.1:9000")
    ap.add_argument("--access-key", default="minioadmin")
    ap.add_argument("--secret-key", default="minioadmin")
    ap.add_argument("--bucket", default="lakehouse")
    ap.add_argument("--prefix", default="admin/collections")
    args = ap.parse_args()

    tgz = Path(args.tgz)
    if not tgz.exists():
        print(f"ERROR: no existe {tgz}")
        return 1

    s3 = boto3.client(
        "s3", endpoint_url=args.endpoint,
        aws_access_key_id=args.access_key, aws_secret_access_key=args.secret_key,
    )

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        tmp_path = Path(tmp)

        # localizar {collection_id}/collection.json
        manifests = list(tmp_path.glob("*/collection.json"))
        if len(manifests) != 1:
            print(f"ERROR: esperaba exactamente 1 */collection.json en el tarball, "
                  f"encontre {len(manifests)}")
            return 1
        col_dir = manifests[0].parent
        cid = col_dir.name

        files = sorted(p for p in col_dir.rglob("*") if p.is_file())
        parts = [p for p in files if p.name != "collection.json"]
        manifest = col_dir / "collection.json"

        # 1) parts primero
        for i, p in enumerate(parts, 1):
            rel = p.relative_to(col_dir)
            key = f"{args.prefix}/{cid}/{rel}"
            s3.upload_file(str(p), args.bucket, key)
            print(f"  [{i}/{len(parts)}] {key}")

        # 2) manifest EL ULTIMO (commit)
        key = f"{args.prefix}/{cid}/collection.json"
        s3.upload_file(str(manifest), args.bucket, key)
        print(f"  [commit] {key}")

    print(f"\nimport completo: coleccion '{cid}' en "
          f"{args.endpoint}/{args.bucket}/{args.prefix}/{cid}/")
    print(f"siguiente paso: python scripts/jupyter/06_index_collection.py {cid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
