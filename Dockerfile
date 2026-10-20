FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# xgboost needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

USER app

# The Cloud Run Job overrides the args with ["replay"].
ENTRYPOINT ["nbaxp"]
CMD ["api"]
