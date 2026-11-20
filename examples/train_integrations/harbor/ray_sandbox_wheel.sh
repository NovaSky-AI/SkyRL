# Sourced by the SANDBOX=ray run scripts. SkyRL's pinned Ray lacks the sandbox fixes until 2.61, so when
# RAY_WHEEL is unset this sets it to the wheel of Ray master 78e4b1f713 for this node's architecture and
# the job's Python (UV_PYTHON when it is a version such as 3.13, else the repo's .python-version).
# The node must run the same Ray commit (docker/Dockerfile.ray-sandbox). RAY_WHEEL= (empty) turns it off.
RAY_SANDBOX_COMMIT=78e4b1f71314fa46ca9bd2907d6aab4ec536909c
if [[ -z "${RAY_WHEEL+set}" ]]; then
  if [[ "${UV_PYTHON:-}" =~ ^3\.[0-9]+$ ]]; then
    _ray_py="$UV_PYTHON"
  else
    _ray_py="$(cat "$(dirname "${BASH_SOURCE[0]}")/../../../.python-version")"
  fi
  case "$(uname -m)" in
    x86_64) _ray_platform=manylinux2014_x86_64 ;;
    aarch64) _ray_platform=manylinux2014_aarch64 ;;
    *)
      echo "no Ray master wheel for $(uname -m): set RAY_WHEEL to a Ray $RAY_SANDBOX_COMMIT wheel" >&2
      exit 1
      ;;
  esac
  _ray_tag="cp${_ray_py//./}"
  RAY_WHEEL="https://s3-us-west-2.amazonaws.com/ray-wheels/master/$RAY_SANDBOX_COMMIT/ray-3.0.0.dev0-$_ray_tag-$_ray_tag-$_ray_platform.whl"
  unset _ray_py _ray_platform _ray_tag
fi
