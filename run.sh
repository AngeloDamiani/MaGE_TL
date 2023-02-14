#!/bin/bash

count=50
prefix="PEND2MUJOCO_200"
script="source_2_target.py"
mode="random" #"default" | "random" | "debug" | "G1" | "G4" | "G5" | "ANG0"
source="Pend" #"Pend" | "MC" | MujocoPend
target="MujocoPend" #"Pend" | "MC" | MujocoPend

for ((i=1; i<= $count; i++)); do
    python3 "$script" "$i" "$prefix" "$mode" "$source" "$target"
done
