#!/bin/bash

# Install poetry
pip install poetry --break-system-packages

# Set up poetry
echo "export PYTHON_PATH='$PYTHON_PATH:$PWD'" >> ~/.zshrc
poetry install --with=dev

export temporary=$(poetry env info --path)
echo "export PATH='$temporary/bin:$PATH'" >> ~/.zshrc
