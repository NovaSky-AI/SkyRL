import os
from urllib.parse import urlparse

from enum import Enum
import torch
import torch.distributed

import io

ONEGB = 1024 * 1024 * 1024


class Cloud(Enum):
    AWS = "aws"
    GCP = "gcp"


def uploadDirectoryToS3(path, bucketname, prefix):
    import boto3

    s3 = boto3.client("s3")
    for root, dirs, files in os.walk(path):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), path)
            s3_key = os.path.join(prefix, relative_path)
            s3.upload_file(os.path.join(root, file), bucketname, s3_key)


def upload_file_to_s3(path, bucketname, prefix):
    import boto3

    s3 = boto3.client("s3")
    s3.upload_file(path, bucketname, prefix)


def upload_dir_to_anyscale(local_path, remote_key):
    save_bucket, remote_prefix, cloud = _get_anyscale_bucket_and_file_key(remote_key)
    if cloud == Cloud.AWS:
        uploadDirectoryToS3(local_path, save_bucket, remote_prefix)
    elif cloud == Cloud.GCP:
        upload_directory_to_gcs(local_path, save_bucket, remote_prefix)
    else:
        raise NotImplementedError


def upload_file_to_anyscale(local_path, remote_key):
    save_bucket, remote_prefix, cloud = _get_anyscale_bucket_and_file_key(remote_key)
    if cloud == Cloud.AWS:
        upload_file_to_s3(local_path, save_bucket, remote_prefix)
    elif cloud == Cloud.GCP:
        upload_file_to_gcs(local_path, save_bucket, remote_prefix)
    else:
        raise NotImplementedError


def _get_anyscale_bucket_and_file_key(path):
    parsed_url = urlparse(os.environ["ANYSCALE_ARTIFACT_STORAGE"])
    if parsed_url.scheme == "s3":
        cloud = Cloud.AWS
    else:
        cloud = Cloud.GCP
    save_bucket, prefix = parsed_url.netloc, parsed_url.path
    prefix = prefix.lstrip("/")
    save_bucket = save_bucket.rstrip("/")
    path = os.path.join(prefix, path)
    return save_bucket, path, cloud


def write_to_s3(obj, path: str):
    import boto3

    save_bucket, path, _ = _get_anyscale_bucket_and_file_key(path)
    s3 = boto3.client("s3")
    cpu_buffer = io.BytesIO()
    torch.save(obj, cpu_buffer)  # save to cpu
    cpu_buffer.seek(0)
    s3.upload_fileobj(cpu_buffer, save_bucket, path)
    cpu_buffer.close()


# Upload a single file to Google Cloud Storage
def upload_file_to_gcs(local_file_path, bucket_name, destination_blob_path):
    import os

    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD"] = str(
        100 * 1024 * 1024
    )  # 100 MiB threshold
    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_PARTS"] = "10"  # 10 parts in parallel
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path, chunk_size=ONEGB)

    blob.upload_from_filename(local_file_path)

    print(f"File {local_file_path} uploaded to gs://{bucket_name}/{destination_blob_path}")


# Upload an entire directory to Google Cloud Storage
def upload_directory_to_gcs(local_directory, bucket_name, destination_prefix=""):
    import os

    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD"] = str(
        100 * 1024 * 1024
    )  # 100 MiB threshold
    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_PARTS"] = "10"  # 10 parts in parallel
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)

            # Determine the blob path in GCS
            relative_path = os.path.relpath(local_path, local_directory)
            blob_path = os.path.join(destination_prefix, relative_path).replace(
                "\\", "/"
            )  # Ensure proper path separators

            # Upload the file
            blob = bucket.blob(blob_path, chunk_size=ONEGB)
            blob.upload_from_filename(local_path)

            print(f"File {local_path} uploaded to gs://{bucket_name}/{blob_path}")

    print("Directory upload complete")
