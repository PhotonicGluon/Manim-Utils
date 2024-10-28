FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Install python and its related dependencies
RUN apt-get update -y
RUN apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv

# Install Manim-related dependencies
RUN apt-get install -y \
    libcairo2-dev \
    libpango1.0-dev \
    ffmpeg

# Install TeX-related dependencies
RUN apt-get install -y \
    texlive \
    texlive-latex-extra

# Install poetry
RUN pip install poetry --break-system-packages
