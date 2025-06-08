#!/usr/bin/env python3
import docker
import os
import sys

IMAGE_NAME = "qwen-cpu"
CONTAINER_NAME = "qwen-cpu-server"
DOCKERFILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PORT = 8080


def build_image():
    client = docker.from_env()
    print(f"Building Docker image '{IMAGE_NAME}'...")
    image, logs = client.images.build(
        path=DOCKERFILE_DIR,
        tag=IMAGE_NAME,
        rm=True
    )
    for chunk in logs:
        if 'stream' in chunk:
            sys.stdout.write(chunk['stream'])
    print("Build complete.")


def run_container():
    client = docker.from_env()
    try:
        ctr = client.containers.get(CONTAINER_NAME)
        print(f"Stopping & removing existing container '{CONTAINER_NAME}'...")
        ctr.stop()
        ctr.remove()
    except docker.errors.NotFound:
        pass

    print(f"Starting container '{CONTAINER_NAME}' on port {MODEL_PORT}...")
    container = client.containers.run(
        IMAGE_NAME,
        detach=True,
        name=CONTAINER_NAME,
        ports={f"{MODEL_PORT}/tcp": MODEL_PORT},
    )
    for line in container.logs(stream=True):
        print(line.decode(), end='')

if __name__ == '__main__':
    build_image()
    run_container()