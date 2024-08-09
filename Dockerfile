FROM python:3.10-slim AS base
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends ffmpeg

FROM base AS pip-build
WORKDIR /wheels
COPY requirements.txt requirements.txt
RUN pip install -U pip  \
    && pip wheel -r requirements.txt --index-url=https://download.pytorch.org/whl/cpu --extra-index-url=https://pypi.org/simple 

FROM base
COPY --from=pip-build /wheels /wheels
WORKDIR /app
RUN pip install -U pip  \
    && pip install --no-cache-dir \
    --no-index \
    -r /wheels/requirements.txt \
    -f /wheels \
    && pip install --no-cache-dir \
    --no-index \
    torch \
    -f /wheels \
    && rm -rf /wheels

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "-u", "/app/main.py"]
