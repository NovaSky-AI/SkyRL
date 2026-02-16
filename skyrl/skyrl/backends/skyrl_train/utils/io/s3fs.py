from datetime import datetime, timezone, timedelta
import fsspec

# Optional AWS deps (present when s3fs is installed)
try:
    import botocore.session as _botocore_session
    from botocore.exceptions import ClientError

    _HAS_BOTOCORE = True
except Exception:
    _HAS_BOTOCORE = False

    class ClientError(Exception):  # fallback type
        pass


_S3_FS = None  # type: ignore


def get_s3_fs():
    """Return a cached S3 filesystem instance, creating it once."""
    global _S3_FS
    if _S3_FS is None:
        _S3_FS = fsspec.filesystem("s3")
    return _S3_FS


def s3_expiry_time():
    """Return botocore credential expiry (datetime in UTC) or None."""
    if not _HAS_BOTOCORE:
        return None
    try:
        sess = _botocore_session.get_session()
        creds = sess.get_credentials()
        if not creds:
            return None
        return getattr(creds, "expiry_time", None) or getattr(creds, "_expiry_time", None)
    except Exception:
        return None


def s3_refresh_if_expiring(fs) -> None:
    """
    Simple refresh:
    - If expiry exists and is within 300s (or past), refresh with fs.connect(refresh=True).
    - Otherwise, do nothing.
    """
    exp = s3_expiry_time()
    if not exp:
        return
    now = datetime.now(timezone.utc)
    if now >= exp - timedelta(seconds=300):
        try:
            fs.connect(refresh=True)  # rebuild session
        except Exception:
            pass


def call_with_s3_retry(fs, fn, *args, **kwargs):
    """
    Wrapper for calling an S3 method. If it fails with ExpiredToken, force refresh once and retry.
    """
    try:
        return fn(*args, **kwargs)
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code")
        if code in {"ExpiredToken", "ExpiredTokenException", "RequestExpired"} and hasattr(fs, "connect"):
            try:
                fs.connect(refresh=True)
            except Exception:
                pass
            return fn(*args, **kwargs)
        raise
