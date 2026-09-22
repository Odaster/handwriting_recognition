.PHONY: install test run docker

install:
	pip install -r requirements.txt

install-test:
	pip install -r requirements-test.txt

test:
	pytest

run:
	python3 -m app

run-fake:
	OCR_BACKEND=fake python3 -m app

docker:
	docker build -t handwriting-recognition .
