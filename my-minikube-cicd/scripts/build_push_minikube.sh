#!/usr/bin/env bash
set -e

IMAGE_NAME="my-app:latest"
# switch docker env to minikube's docker daemon
eval $(minikube docker-env)

docker build -t ${IMAGE_NAME} .
# apply manifests (use image name that k8s uses: for local minikube you can use just my-app:latest)
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/app-deployment.yaml
kubectl apply -f k8s/app-service.yaml

echo "Deployed to minikube. Use: minikube service my-app-service"
