#!/bin/bash

# Set up poetry
echo "export PYTHON_PATH='$PYTHON_PATH:$PWD'" >> ~/.bashrc
poetry install --with=dev

export temporary=$(poetry env info --path)
echo "export PATH='$temporary/bin:$PATH'" >> ~/.bashrc
