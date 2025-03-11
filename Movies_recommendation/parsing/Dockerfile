FROM python:3.10.16
WORKDIR /app
COPY imdb_scraper.py requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "imdb_scraper.py"]