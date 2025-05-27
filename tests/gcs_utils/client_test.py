import os
import pytest
from unittest import mock
from pkg.gcs_utils.client import get_bucket, upload_file, download_file, list_files

# Set test values for environment
os.environ["GRAPHCAST_BUCKET_NAME"] = "test-bucket"
os.environ["GRAPHCAST_PROJECT_ID"] = "test-project"

@pytest.fixture
def mock_bucket():
    mock_blob = mock.MagicMock()
    mock_bucket = mock.MagicMock()
    mock_bucket.blob.return_value = mock_blob
    return mock_bucket, mock_blob

@pytest.fixture
def mock_storage_client(mock_bucket):
    mock_bucket_obj, _ = mock_bucket
    mock_client = mock.MagicMock()
    mock_client.bucket.return_value = mock_bucket_obj
    with mock.patch("pkg.gcs_utils.client.storage.Client", return_value=mock_client):
        yield mock_client

def test_get_bucket_returns_bucket(mock_storage_client):
    from pkg.gcs_utils.client import get_bucket
    bucket = get_bucket()
    assert bucket is mock_storage_client.bucket.return_value
    mock_storage_client.bucket.assert_called_once_with("test-bucket")

def test_upload_file_calls_blob_and_upload(mock_storage_client, mock_bucket):
    _, mock_blob = mock_bucket
    upload_file("local.txt", "remote/path.txt")
    mock_blob.upload_from_filename.assert_called_once_with("local.txt")

def test_download_file_calls_blob_and_download(mock_storage_client, mock_bucket):
    _, mock_blob = mock_bucket
    download_file("remote/path.txt", "local.txt")
    mock_blob.download_to_filename.assert_called_once_with("local.txt")

def test_list_files_returns_blob_names(mock_storage_client):
    mock_blob1 = mock.Mock(name="blob1")
    mock_blob1.name = "file1.txt"
    mock_blob2 = mock.Mock(name="blob2")
    mock_blob2.name = "file2.txt"

    bucket = mock_storage_client.bucket.return_value
    bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

    result = list_files("some/prefix/")
    bucket.list_blobs.assert_called_once_with(prefix="some/prefix/")
    assert result == ["file1.txt", "file2.txt"]
