FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN apt-get update && apt-get upgrade -y

WORKDIR /app

COPY ["requirements.txt", "./"]
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# COPY [ "Pipfile", "Pipfile.lock", "./" ]

# RUN pipenv install --system --deploy

COPY [ "starter.py", "./" ]

ENTRYPOINT [ "python", "starter.py"]
