from minio import Minio
import os

def get_client():
    return Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS"],
        secret_key=os.environ["MINIO_SECRET"],
        secure=False
    )
