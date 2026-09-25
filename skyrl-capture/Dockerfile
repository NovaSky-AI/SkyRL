FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so source edits do not invalidate the layer.
# `viewer` is copied because the wheel force-includes it; without it here the
# build fails outright rather than producing an image with no viewer.
COPY pyproject.toml README.md ./
COPY src ./src
COPY viewer ./viewer
RUN pip install --no-cache-dir ".[fast,s3]"

# Mount /data/record if the results should outlive the container -- without it,
# nothing survives the process. Exports are written beside the record.
#
# PROXY_HOST is 0.0.0.0 because the default, 127.0.0.1, is loopback *inside the
# container*: `-p 8080:8080` would forward to a port nothing is accepting on.
# PUBLIC_URL is what a trajectory's base URL is built from, so override it when
# callers reach this container by any other name.
ENV PROXY_HOST=0.0.0.0

EXPOSE 8080
ENTRYPOINT ["skyrl-capture"]
CMD ["serve"]
