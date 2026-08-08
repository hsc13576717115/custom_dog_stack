SHELL := /usr/bin/env bash

.PHONY: bootstrap validate check-system train-smoke train tensorboard ros2-build

bootstrap:
	./scripts/bootstrap.sh

validate:
	./scripts/validate.sh

check-system:
	./scripts/check_system.sh

train-smoke:
	./scripts/train_smoke.sh

train:
	./scripts/train.sh

tensorboard:
	./scripts/tensorboard.sh

ros2-build:
	./scripts/build_ros2.sh
