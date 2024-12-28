FROM python:3.12.8-slim-bookworm

WORKDIR /app

COPY . .
RUN set -eux; \
  pip install torch torchvision matplotlib flask numpy Pillow

EXPOSE 5000
ENTRYPOINT []
CMD ["python3", "app.py"]
