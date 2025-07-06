#!/usr/bin/env bash
# Setup script for PCA_Business_Cycle project
set -e

# Ensure apt repositories are updated
sudo apt-get update

# Install python and pip if not already available
sudo apt-get install -y python3 python3-pip

# Upgrade pip
python3 -m pip install --upgrade pip

# Install Python dependencies
python3 -m pip install -r requirements.txt
