# Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
    - [Create The virtual environment](#create-the-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Add Dependencies](#add-dependencies)
  - [Run the Application](#run-the-application)
    - [CLI](#cli)
    - [Notebook kernel](#notebook-kernel)
- [Managing `.env` Environment](#managing-env-environment)
    - [Create a new environment](#create-a-new-environment)


# Installation

## Prerequisites

- [Python 3.12](https://www.python.org/downloads/windows/)
- [Poetry](https://python-poetry.org/docs/#installation)

### Start docker container

```bash
docker run --gpus=all -p 127.0.0.1:9060:8080 us-docker.pkg.dev/colab-images/public/runtime
```

### Create The virtual environment

```bash
py -3.12 -m venv venv
poetry env use venv/Scripts/python.exe
```

## Install Dependencies

```bash
poetry install && poe install-pytorch && poe install-torchvision
```

## Add Dependencies

```bash
poetry add <dependency>
```

## Run the Application

### CLI

```bash
poetry run python main.py
```

### Notebook kernel

In order to use the virtual environment in a notebook select the environment with the name that was created by poetry.

You can see all poetry environments with the following command.

```bash
poetry env list
```

# Managing `.env` Environment

## Create a new environment

1. create a new file `.env` at the root of the project
2. add the environment variables in the file from the file `.env.example.md`

