FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY research_lab ./research_lab

RUN pip install --no-cache-dir .

RUN mkdir -p /data/campaigns

EXPOSE 8765

CMD ["python", "-m", "research_lab.ui", "--store", "/data/campaigns", "--port", "8765"]
