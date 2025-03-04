FROM python:3.10.16
WORKDIR /app
COPY model.py requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "model.py"]