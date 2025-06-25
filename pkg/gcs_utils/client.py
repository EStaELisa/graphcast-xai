import os
from dotenv import load_dotenv
from google.cloud import storage

# Load environment variables
load_dotenv()

def get_bucket():
    """
    Returns a GCS bucket object using the configured project and bucket name.
    """
    bucket_name = os.getenv("GRAPHCAST_BUCKET_NAME")
    project_id = os.getenv("GRAPHCAST_PROJECT_ID")

    if not bucket_name or not project_id:
        raise EnvironmentError("Missing required environment variables.")

    client = storage.Client(project=project_id)
    return client.bucket(bucket_name)


def upload_file(local_path: str, blob_path: str):
    """
    Uploads a local file to the configured GCS bucket.

    Parameters
    ----------
    local_path: str
        Local file path to upload.
    blob_path: str
        Path in the GCS bucket where the file will be stored.
    """
    bucket = get_bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    bucket_name = bucket.name
    print(f"☁️ Uploaded: {local_path} → gs://{bucket_name}/{blob_path}")

def download_file(blob_path: str, local_path: str):
    """
    Downloads a file from the configured GCS bucket to a local path.

    Parameters
    ----------
    blob_path: str
        Path in the GCS bucket to download from.
    local_path: str
        Local file path where the downloaded file will be saved.
    """
    bucket = get_bucket()
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    bucket_name = bucket.name
    print(f"⬇️ Downloaded: gs://{bucket_name}/{blob_path} → {local_path}")

def list_files(prefix: str = ""):
    """
    Lists all file paths in the bucket that start with the given prefix.

    Parameters
    ----------
    prefix: str
        Optional prefix to filter the files listed in the bucket.   
    
    Returns
    -------
    list
        List of file paths in the GCS bucket that match the prefix.
    """
    bucket = get_bucket()
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)]
