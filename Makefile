.PHONY: install test run docker

install:
	pip install -r requirements.txt

install-test:
	pip install -r requirements-test.txt

test:
	pytest

run:
	python -m app

run-fake:
	OCR_BACKEND=fake python -m app

docker:
	docker build -t handwriting-recognition .
