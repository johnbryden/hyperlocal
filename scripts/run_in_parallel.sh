#!/usr/bin/env bash
set -euo pipefail

PYTHON="$(pyenv which python)"
export PYTHON

mkdir -p junk

# Items may contain spaces
items=(
    "Peckham"
    "Gorton and Denton"
    "Kensington and Bayswater"
    "Bolsover"
    "Makerfield"
)

# The command you want to run per item:
run_one() {
  local x="$1"
  # replace this with your real command; keep "$x" quoted
  printf 'processing: %s\n' "$x"
  # e.g.: mytool --arg "$x"
}

export -f run_one

# Feed items to parallel as newline-separated input
printf '%s\n' "${items[@]}" |
  parallel --linebuffer '
    x={}
    # make a safe filename from x (spaces->underscore, remove nasty chars)
    safe=$(printf "%s" "$x" | tr " /" "__" | tr -cd "A-Za-z0-9._-")
    $PYTHON -u -m app.run_process --location "$x" >"junk/${safe}.log" 2>&1
  '
