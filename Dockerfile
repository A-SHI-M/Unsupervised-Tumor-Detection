FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py params.yaml ./

COPY trainedmodels/ ./trainedmodels/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
