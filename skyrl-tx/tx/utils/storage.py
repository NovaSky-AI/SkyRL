from contextlib import contextmanager
import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from typing import Generator
from cloudpathlib import AnyPath


@contextmanager
def pack_and_upload(dest: AnyPath) -> Generator[Path, None, None]:
    """Give the caller a temp directory that gets uploaded as a tar.gz archive on exit.

    Args:
        dest: Destination path for the tar.gz file
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        yield tmp_path

        # Create tar archive of temp directory contents
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            for p in tmp_path.iterdir():
                tar.add(p, arcname=p.name)
        tar_buffer.seek(0)

        # Write the tar file (handles both local and cloud storage)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            f.write(tar_buffer.read())


@contextmanager
def download_and_unpack(source: AnyPath) -> Generator[Path, None, None]:
    """Download and extract a tar.gz archive and give the content to the caller in a temp directory.

    Args:
        source: Source path for the tar.gz file
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Download and extract tar archive (handles both local and cloud storage)
        with source.open("rb") as f:
            with tarfile.open(fileobj=f, mode="r:gz") as tar:
                tar.extractall(tmp_path, filter="data")

        yield tmp_path


def download_file(source: AnyPath) -> io.BytesIO:
    """Download a file from storage and return it as a BytesIO object.

    Args:
        source: Source path for the file (local or cloud)

    Returns:
        BytesIO object containing the file contents
    """
    buffer = io.BytesIO()
    with source.open("rb") as f:
        buffer.write(f.read())
    buffer.seek(0)
    return buffer
