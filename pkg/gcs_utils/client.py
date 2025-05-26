import os
from dotenv import load_dotenv
from google.cloud import storage

# Load environment variables
load_dotenv()

# Fetch from .env
BUCKET_NAME = os.getenv("GRAPHCAST_BUCKET_NAME")
PROJECT_ID = os.getenv("GRAPHCAST_PROJECT_ID")

if not BUCKET_NAME or not PROJECT_ID:
    raise EnvironmentError("Missing required environment variables.")

def get_bucket():
    """
    Returns a GCS bucket object using the configured project and bucket name.

    Returns:
        google.cloud.storage.Bucket: The GCS bucket object.
    """
    client = storage.Client(project=PROJECT_ID)
    return client.bucket(BUCKET_NAME)

def upload_file(local_path: str, blob_path: str):
    """
    Uploads a local file to the configured GCS bucket.

    Args:
        local_path (str): Path to the local file to upload.
        blob_path (str): Destination path within the bucket (e.g., "folder/file.nc").
    """
    bucket = get_bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"☁️ Uploaded: {local_path} → gs://{BUCKET_NAME}/{blob_path}")

def download_file(blob_path: str, local_path: str):
    """
    Downloads a file from the configured GCS bucket to a local path.

    Args:
        blob_path (str): Path to the file within the bucket.
        local_path (str): Local destination path to save the file.
    """
    bucket = get_bucket()
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"⬇️ Downloaded: gs://{BUCKET_NAME}/{blob_path} → {local_path}")

def list_files(prefix: str = ""):
    """
    Lists all file paths in the bucket that start with the given prefix.

    Args:
        prefix (str, optional): Folder path or prefix to filter files (e.g., "data/input/"). Defaults to "".

    Returns:
        list[str]: A list of blob (file) paths matching the prefix.
    """
    bucket = get_bucket()
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)]
