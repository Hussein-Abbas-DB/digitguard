FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY digitguard ./digitguard
RUN python -m digitguard.train && useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "digitguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
