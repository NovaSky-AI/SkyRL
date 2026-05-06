#!/usr/bin/env bash
set -euo pipefail

# Remove Harbor/R2EGym docker-compose resources left by this user's local
# benchmark data tree. This is intentionally stricter than "name=r2egym-"
# alone because rootful Docker is shared on the head node.

DATA_ROOT="${DATA_ROOT:-$HOME/data/harbor}"
DRY_RUN="${DRY_RUN:-0}"

docker info >/dev/null

container_ids=()
while IFS= read -r cid; do
  [ -n "$cid" ] || continue
  name="$(docker inspect "$cid" --format '{{.Name}}' 2>/dev/null || true)"
  workdir="$(docker inspect "$cid" --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' 2>/dev/null || true)"
  case "$name" in
    /r2egym-*)
      case "$workdir" in
        "$DATA_ROOT"/*/environment|"$DATA_ROOT"/*/*/environment)
          container_ids+=("$cid")
          ;;
      esac
      ;;
  esac
done < <(docker ps -aq --filter name=r2egym-)

if [ "${#container_ids[@]}" -gt 0 ]; then
  echo "Removing ${#container_ids[@]} Harbor/R2EGym containers under DATA_ROOT=$DATA_ROOT"
  if [ "$DRY_RUN" != 1 ]; then
    docker rm -f "${container_ids[@]}" >/dev/null 2>&1 || true
  fi
else
  echo "No Harbor/R2EGym containers matched DATA_ROOT=$DATA_ROOT"
fi

network_ids=()
while IFS= read -r nid; do
  [ -n "$nid" ] || continue
  project="$(docker network inspect "$nid" --format '{{index .Labels "com.docker.compose.project"}}' 2>/dev/null || true)"
  case "$project" in
    r2egym-*) network_ids+=("$nid") ;;
  esac
done < <(docker network ls -q --filter name=r2egym-)

if [ "${#network_ids[@]}" -gt 0 ]; then
  echo "Removing ${#network_ids[@]} Harbor/R2EGym compose networks"
  if [ "$DRY_RUN" != 1 ]; then
    docker network rm "${network_ids[@]}" >/dev/null 2>&1 || true
  fi
fi

volume_names=()
while IFS= read -r volume; do
  [ -n "$volume" ] || continue
  project="$(docker volume inspect "$volume" --format '{{index .Labels "com.docker.compose.project"}}' 2>/dev/null || true)"
  case "$project" in
    r2egym-*) volume_names+=("$volume") ;;
  esac
done < <(docker volume ls -q --filter name=r2egym-)

if [ "${#volume_names[@]}" -gt 0 ]; then
  echo "Removing ${#volume_names[@]} Harbor/R2EGym compose volumes"
  if [ "$DRY_RUN" != 1 ]; then
    docker volume rm -f "${volume_names[@]}" >/dev/null 2>&1 || true
  fi
fi
