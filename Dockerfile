FROM python:3.11-slim

WORKDIR /app

COPY Model/v8/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY Model /app/Model

ENV VLLM_BASE_URL=http://beamdata-vllm:8000

EXPOSE 8000

CMD ["uvicorn", "Model.v8.api:app", "--host", "0.0.0.0", "--port", "8000"]
