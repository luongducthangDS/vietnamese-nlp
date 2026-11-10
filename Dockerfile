FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vnlp/ vnlp/
# Models come from `python -m vnlp.train ...` + `python -m vnlp.report`; mounted so retraining needs no rebuild.
ENV ARTIFACTS_DIR=/app/artifacts
EXPOSE 8001
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8001/health')"
CMD ["uvicorn", "vnlp.api:app", "--host", "0.0.0.0", "--port", "8001"]
