FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY model_service.py .

EXPOSE 5001

CMD ["python", "model_service.py"]
