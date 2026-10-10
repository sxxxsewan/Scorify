# Scorify

Scorify — проект кредитного скоринга. Модель на базе LightGBM оценивает вероятность серьёзной просрочки (дефолта) в течение двух лет по датасету Give Me Some Credit и отдаёт решение через REST API на FastAPI. Конвейер обучения версионируется через DVC, эксперименты логируются в MLflow, а на каждый push запускаются линтер, тесты и сборка Docker-образа.

## Стек

- **Язык:** Python 3.11
- **ML и данные:** LightGBM, scikit-learn, pandas, NumPy, PySpark
- **API:** FastAPI, Pydantic, Uvicorn
- **MLOps:** DVC, MLflow (SQLite), params.yaml, GitHub Actions
- **Тесты и качество кода:** pytest, HTTPX / TestClient, ruff
- **Инфраструктура:** Docker, Docker Compose, WSL 2 (Ubuntu)

## Результаты

Метрики считаются скриптом обучения и сохраняются в `metrics.json`:

| Метрика | Значение |
|---|---|
| ROC-AUC, OOF (5-fold CV) | 0.865 |
| ROC-AUC, holdout | 0.865 |

Решение принимается по порогу вероятности дефолта 0.30: значение ниже порога даёт `APPROVED`, значение 0.30 и выше — `REJECTED`.

Тест выравнивания (offline/online alignment) сравнивает предсказания модели на строках из обработанного датасета с ответами API. Допустимое расхождение — 0.05.

## Структура проекта

```text
app/                 FastAPI-сервис и схемы запросов и ответов
data/raw/            Исходные CSV (под управлением DVC, в Git лежат только .dvc-файлы)
data/processed/      Признаки и сабмит, создаются конвейером (не в Git)
models/              Сериализованная модель LightGBM (не в Git)
notebooks/           EDA и ноутбуки пайплайна
src/features/        Построение признаков (build_features.py)
src/models/          Обучение (train_model.py) и пакетный инференс (predict_model.py)
tests/               Тесты API и тест выравнивания
params.yaml          Пути и гиперпараметры
dvc.yaml             Стадии конвейера: build_features, train
.github/workflows/   CI: ruff, pytest, сборка Docker-образа
```

## Локальный запуск

Нужен Python 3.11. Команды для WSL или Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PySpark требует установленной Java 17 (или 11). Задайте `JAVA_HOME` до запуска конвейера.

Запуск API из корня проекта:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- `GET http://localhost:8000/health` — проверка состояния;
- `http://localhost:8000/docs` — интерактивная документация.

## Предиктивный API

`POST /predict` принимает десять исходных признаков. Допускаются оригинальные имена колонок с дефисами и варианты с подчёркиваниями.

Запрос:

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.05,
  "age": 45,
  "NumberOfTime30_59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.1,
  "MonthlyIncome": 10000,
  "NumberOfOpenCreditLinesAndLoans": 5,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60_89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 1
}
```

Ответ:

```json
{
  "default_probability": 0.0123,
  "decision": "APPROVED"
}
```

Пропуски заполняются сервисными значениями: для `MonthlyIncome` — 5400.0, для `NumberOfDependents` — 0.0. Для каждого поля также выставляется флаг пропуска. Пайплайн обучения заполняет пропуски медианами по децилям возраста, поэтому предсказания для неполных запросов могут немного отличаться от офлайн-предобработки.

## Воспроизведение конвейера

Данные и результаты стадий хранятся в DVC. Команды выполняются из корня проекта:

```bash
dvc pull
dvc repro
dvc push
```

- `dvc pull` восстанавливает raw-данные из настроенного хранилища;
- `dvc repro` запускает стадии `build_features` и `train` и обновляет `metrics.json`, `dvc.lock` и модель;
- `dvc push` отправляет результаты в хранилище.

По умолчанию DVC использует локальную папку `./dvc_storage`. Чтобы воспроизвести конвейер на другой машине, скопируйте эту папку или настройте общий remote (S3-совместимое хранилище или другой сервис) через `dvc remote`.

## MLflow

Обучение логирует параметры, метрики (OOF и holdout ROC-AUC) и модель в MLflow. Хранилище — SQLite-файл `mlflow.db` в корне проекта. Модель регистрируется в Model Registry под именем `scorify-lgbm`.

Открыть интерфейс:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

После этого откройте `http://127.0.0.1:5000`.

## Тесты и проверки

```bash
ruff check .
pytest tests/ -v
```

Тесты, которым нужны данные или модель из DVC, пропускаются, если артефакты не восстановлены локально.

## Docker

Сборка образа:

```bash
docker build --tag scorify:ci .
```

Запуск через Compose:

```bash
docker compose up --build
```

API станет доступен на порту 8000. Если сборка падает при загрузке пакетов, проверьте прокси и сетевой доступ.

## CI

GitHub Actions запускает на каждый push и pull request: установку зависимостей, `ruff check .`, `pytest tests/ -v` и сборку Docker-образа. В CI нет данных DVC, поэтому тесты, которым они нужны, пропускаются.
```