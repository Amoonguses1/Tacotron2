PROJECT_NAME=mystudy
USER=hoge
IMAGE_NAME=${USER}_${PROJECT_NAME}
CONTAINER_NAME=${USER}_${PROJECT_NAME}
PORT=8888
SHM_SIZE=2g
FORCE_RM=true
build:
	docker build \
		--build-arg USER_ID=$(shell id -u) \
		--build-arg GROUP_ID=$(shell id -g) \
		-f docker/Dockerfile \
		-t ${IMAGE_NAME} \
		--force-rm=${FORCE_RM}\
		.

restart: stop run

run:
	docker run \
		-dit \
		--gpus all \
		-v $(PWD):/workspace \
		--name $(CONTAINER_NAME) \
		--rm \
		--shm-size $(SHM_SIZE) \
		--platform=linux/amd64 \
		$(IMAGE_NAME)

exec:
	docker exec \
		-it \
		$(CONTAINER_NAME) bash 

stop:
	docker stop $(IMAGE_NAME)

run_jupyter:
	jupyter nbextension enable --py widgetsnbextension
	jupyter notebook --ip 0.0.0.0 --port ${PORT} --allow-root
