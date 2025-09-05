#!/usr/bin/env bash

# Convert an environment spec (org/name@version) into uv flags.
# Usage: verifiers_env_to_uv_flags "will/wordle@0.1.4"
verifiers_env_to_uv_flags() {
  local spec="$1"
  local org="${spec%%/*}"
  local rest="${spec#*/}"
  local name="${rest%%@*}"
  local version=""
  if [[ "$rest" == *"@"* ]]; then
    version="${rest##*@}"
  fi
  local index_url="https://hub.primeintellect.ai/${org}/simple/"
  local with_req="$name"
  if [[ -n "$version" ]]; then
    with_req="${name}==${version}"
  fi
  echo "--with ${with_req} --extra-index-url ${index_url}"
} 