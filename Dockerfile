FROM python:3.12-slim as requirements-stage

WORKDIR /tmp

RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry self add poetry-plugin-export # Explicitly install the export plugin
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.12-slim

WORKDIR /home

COPY --from=requirements-stage /tmp/requirements.txt /home/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /home/requirements.txt

COPY ./app /home/app
COPY ./data/docs /home/lawgicx/data/docs

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
