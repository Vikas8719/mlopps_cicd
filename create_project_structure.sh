#!/bin/bash

PROJECT_NAME="my-minikube-cicd"

echo "Creating project structure: $PROJECT_NAME"

mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Base folders
mkdir -p src
mkdir -p .github/workflows
mkdir -p k8s
mkdir -p scripts

# SRC files
touch src/app.py
touch src/train.py
touch src/utils.py

# Root files
touch Dockerfile
touch requirements.txt
touch Makefile

# GitHub Actions
touch .github/workflows/ci-cd.yml

# Kubernetes manifests
touch k8s/app-deployment.yaml
touch k8s/app-service.yaml
touch k8s/mlflow-deployment.yaml

# Scripts
touch scripts/build_push_minikube.sh

echo "All folders and files created successfully!"
echo "Project Structure:"
tree .
