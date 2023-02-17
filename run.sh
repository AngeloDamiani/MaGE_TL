#!/bin/bash

count=20
base_logs="./logs"
prefix="ran_300"
script="source_2_target.py"
mode="random" #"default" | "random" | "debug" | "G1" | "G4" | "G5" | "ANG0"
source="Pend" #"Pend" | "MC"
target="MC" #"Pend" | "MC"

for ((i=1; i<= $count; i++)); do
    python3 "$script" "$i" "$prefix" "$mode" "$base_logs" "$source" "$target"
done
