#!/bin/bash

declare -a profilers=("ProfileIntegration")
#("ProfileFGR" "ProfileColoredICP" "ProfileRGBDOdometry")

profiler_path="../../../cmake-build-release/bin/examples"

declare -a configs=("fr2_desktop" "fr3_household"
                    "lounge" "copyroom" "livingroom1")
config_path="../ReconstructionSystem/config"

for profiler in "${profilers[@]}"
do
    for config in "${configs[@]}"
    do
        "$profiler_path/$profiler" "$config_path/$config.json"
    done
done
