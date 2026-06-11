"""Crea el bucket `lakehouse` en el MinIO local (idempotente).

Coincide con `MINIO_BUCKET_NAME=lakehouse` de `sandbox_mteb/env.example.jupyter`
y replica el nombre del bucket de produccion para que la ruta sea identica:
`lakehouse/admin/collections/{collection_id}/`.

Ejecutar tras `02_minio_run.sh` y desde el venv (`source /home/jovyan/lirag_venv/bin/activate`).
"""
import sys
import boto3
from botocore.exceptions import ClientError

ENDPOINT = "http://127.0.0.1:9000"
ACCESS = "minioadmin"
SECRET = "minioadmin"
BUCKET = "lakehouse"

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS,
    aws_secret_access_key=SECRET,
    region_name="us-east-1",
)

try:
    s3.create_bucket(Bucket=BUCKET)
    print(f"bucket '{BUCKET}' creado")
except ClientError as e:
    code = e.response.get("Error", {}).get("Code", "")
    if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
        print(f"bucket '{BUCKET}' ya existia (OK)")
    else:
        print(f"ERROR creando bucket: {code}: {e}")
        sys.exit(1)

print("buckets:", [b["Name"] for b in s3.list_buckets()["Buckets"]])
