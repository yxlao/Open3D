#!/bin/bash

declare -a profilers=("RunSystem")

profiler_path="../../../cmake-build-release/bin"

declare -a configs=("livingroom1" "livingroom2" "office1" "office2")
config_path="../ReconstructionSystem/config"

for profiler in "${profilers[@]}"
do
    for config in "${configs[@]}"
    do
        "$profiler_path/$profiler" "$config_path/$config.json" | tee "$config.txt"
    done
done
