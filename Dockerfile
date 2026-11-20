# One image for every pipeline step. ECS/Fargate and SageMaker both run it with a
# different command, e.g. ["predict", "--config", "configs/kelmarsh.yaml"].
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# libgomp is required by LightGBM and XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock pyproject.toml ./
RUN pip install -r requirements.lock
COPY src ./src
COPY configs ./configs
RUN pip install -e . --no-deps

RUN useradd --create-home runner && chown -R runner /app
USER runner
ENTRYPOINT ["windcast"]
CMD ["run-all", "--config", "configs/synthetic.yaml"]
