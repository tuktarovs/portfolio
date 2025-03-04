FROM python:3.10.16

WORKDIR /app

COPY api.py requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
