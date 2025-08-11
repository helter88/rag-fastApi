FROM python:3.13-slim as builder

# To są standardowe, dobre praktyki. 
# Pierwsza zapobiega tworzeniu plików .pyc w kontenerze (utrzymuje czystość). 
# Druga sprawia, że logi Pythona pojawiają się natychmiast, bez buforowania, co jest kluczowe do monitorowania.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Instalacja uv
RUN pip install uv

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie plików projektu
COPY pyproject.toml .

# Instalacja zależności
# Używamy --system, aby instalować w globalnym site-packages kontenera
# W tym etapie instalujemy też zależności dev, aby móc uruchomić testy w CI/CD
RUN uv pip install --system -e .[dev]

# Finalny obraz produkcyjny
FROM python:3.13-slim as final

# Ustawienie zmiennych środowiskowych
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie zależności z etapu buildera
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Kopiowanie kodu aplikacji
COPY ./app ./app

# Uruchomienie aplikacji
# Używamy gunicorn do uruchomienia na produkcji z workerami uvicorn
#-w 4: Uruchom 4 procesy robocze. Pozwala to na obsługę 4 zapytań prawdziwie równolegle na maszynie z wieloma rdzeniami CPU.

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]