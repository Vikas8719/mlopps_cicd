.PHONY: build local-deploy

build:
	docker build -t ghcr.io/<OWNER>/<REPO>:latest .

local-deploy:
	./scripts/build_push_minikube.sh
