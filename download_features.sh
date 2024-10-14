#!/usr/bin/env bash

# make sure the folder doesn't already exist
if [ -d "visual_features" ]; then
    echo "Data already downloaded!"
    exit 0
fi

wget --output-document features.zip https://www.cs.cmu.edu/~data4robotics/release/features.zip
unzip features.zip
rm features.zip
