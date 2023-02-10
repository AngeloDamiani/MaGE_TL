#!/bin/bash

count=20
prefix="RAN_100"
script="source_2_target.py"
mode="RAN1003" #"default" | "random" | "debug" | "G1" | "G4" | "G5" | "ANG0"
source="Pend" #"Pend" | "MC"
target="MC" #"Pend" | "MC"

for ((i=1; i<= $count; i++)); do
    python3 "$script" "$i" "$prefix" "$mode" "$source" "$target"
done
