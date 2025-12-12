from contextlib import contextmanager
import io
import gzip
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

        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("wb") as f:

            # Use compresslevel=0 to prioritize speed, as checkpoint files don't compress well.
            with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=0) as gz_stream:
                with tarfile.open(fileobj=gz_stream, mode="w:") as tar:
                    tar.add(tmp_path, arcname="")


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
