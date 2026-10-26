# Scorify

![CI Build](https://github.com/scorify/scorify/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5+-green.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.5+-orange.svg)
![DVC](https://img.shields.io/badge/DVC-3.59+-9cf.svg)

**Scorify** — это продуктовый сервис кредитного скоринга с открытым исходным кодом. Модель на базе **LightGBM** оценивает вероятность серьёзной просрочки (дефолта) в течение двух лет по датасету *Give Me Some Credit* и отдаёт вычисленный скоринг и решение по кредиту через REST API на **FastAPI**.

Конвейер подготовки данных и обучения полностью автоматизирован и версионируется с помощью **DVC**. Эксперименты и реестр моделей логируются в **MLflow**. Качество кода и работоспособность сервиса контролируются линтером **Ruff**, набором автоматических тестов на **pytest** и непрерывной интеграцией на **GitHub Actions**.

Добро пожаловать в проект! Мы рады любой помощи — от исправления документации и добавления тестов до оптимизации признаков и модели.

---

## Основные возможности

- **Высокое качество классификации:** модель LightGBM обучена с помощью 5-fold Stratified Cross-Validation и гиперпараметров, обеспечивая высокий ROC-AUC.
- **Масштабируемый Feature Engineering:** генерация и трансформация признаков выполнена с использованием **PySpark**.
- **Инференс в реальном времени:** REST API на **FastAPI** принимает входные анкетные данные, заполняет пропуски и мгновенно возвращает скоринговое решение.
- **Воспроизводимость MLOps:** конвейер `build_features` -> `train` описан в `dvc.yaml`, а метрики версионируются в `metrics.json`.
- **Проверка выравнивания (Alignment):** отдельный интеграционный тест гарантирует совпадение предсказаний оффлайн-модели и онлайн-API.
- **Контейнеризация:** готов к развертыванию через Docker и Docker Compose.

---

## Результаты и метрики

Метрики рассчитываются в процессе обучения и сохраняются в `metrics.json`:

| Метрика | Значение |
|---|---|
| **ROC-AUC (OOF, 5-fold CV)** | **0.865** |
| **ROC-AUC (Holdout test set)** | **0.865** |

### Принятие решения по кредиту

Решение принимается на основе порога вероятности дефолта **0.30**:
- Вероятность `< 0.30` -> `APPROVED` (Кредит одобрен)
- Вероятность `≥ 0.30` -> `REJECTED` (Отказ в выдаче)

Отклонение между предсказаниями оффлайн-модели и онлайн API контролируется тестом выравнивания (`tests/test_alignment.py`) с допустимым расхождением не более **0.05**.

---

## Технический стек

- **Язык программирования:** Python 3.11
- **Машинное обучение и данные:** LightGBM, PySpark, pandas, NumPy, scikit-learn, Parquet
- **REST API:** FastAPI, Pydantic v2, Uvicorn
- **MLOps & DVC:** DVC, MLflow (SQLite backend & model registry), params.yaml
- **Качество и тестирование:** pytest, HTTPX / TestClient, Ruff
- **Инфраструктура:** Docker, Docker Compose, GitHub Actions CI

---

## Структура проекта

```text
.
├── app/                  # REST API сервис на FastAPI (роуты, схемы Pydantic, логика инференса)
├── data/
│   ├── raw/              # Исходные CSV датасеты (под управлением DVC)
│   └── processed/        # Обработанные признаковые датасеты (Parquet, создаются конвейером)
├── models/               # Сериализованная модель LightGBM (.pkl, под управлением DVC)
├── notebooks/            # Изыскательский анализ данных (EDA) и эксперименты
├── src/
│   ├── features/         # Построение и трансформации признаков на PySpark (build_features.py)
│   └── models/           # Обучение модели (train_model.py) и пакетный инференс (predict_model.py)
├── tests/                # Юнит- и интеграционные тесты (API, Alignment)
├── .github/workflows/    # CI конвейеры GitHub Actions
├── compose.yaml          # Определение сервиса Docker Compose
├── Dockerfile            # Инструкция сборки Docker-образа сервиса
├── dvc.yaml              # Конструкция стадий DVC (build_features, train)
├── metrics.json          # Актуальные метрики качества модели
└── params.yaml           # Централизованная конфигурация путей и гиперпараметров
```

---

## Локальное развертывание и запуск

### Требования к окружению

1. **Python:** 3.11+
2. **Java JDK:** Java 21 (или Java 11+) для работы PySpark engine.

### 1. Установка зависимостей

Клонируйте репозиторий и создайте виртуальное окружение:

```bash
git clone https://github.com/scorify/scorify.git
cd scorify

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Убедитесь, что переменная `JAVA_HOME` указывает на установленный JDK:

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64  # Пример пути для Ubuntu/Debian
```

### 2. Запуск REST API

Для запуска веб-сервера выполните команду из корня проекта:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

После запуска доступны следующие адреса:
- **Health check:** `GET http://localhost:8000/health`
- **Интерактивная документация (Swagger UI):** `http://localhost:8000/docs`
- **ReDoc documentation:** `http://localhost:8000/redoc`

---

## Описание API

### `POST /predict`

Принимает анкетные показатели заёмщика и возвращает вероятностную оценку риска и решение. Допускаются как названия полей с дефисами (оригинальный датасет Kaggle), так и варианты с подчёркиваниями.

#### Пример запроса:

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

#### Пример ответа:

```json
{
  "default_probability": 0.0123,
  "decision": "APPROVED"
}
```

*Примечание по пропускам:* Если в запросе не переданы `MonthlyIncome` или `NumberOfDependents`, сервис подставляет значения по умолчанию (5400.0 и 0.0 соответственно) и устанавливает соответствующий флаг наличия пропуска.

---

## Воспроизведение MLOps-конвейера (DVC)

Все стадии обработки данных и обучения модели управляются через DVC.

```bash
# Загрузка актуальных данных и модели из хранилища DVC
dvc pull

# Запуск воспроизведения всех стадий конвейера
dvc repro

# Сохранение обновленных артефактов в DVC remote
dvc push
```

- `dvc pull` — получает исходный датасет `cs-training.csv` и веса модели из DVC remote.
- `dvc repro` — последовательно запускает стадии `build_features` и `train`, обновляя Parquet-признаки, `metrics.json`, `dvc.lock` и обученную модель в `models/`.

По умолчанию локальное DVC хранилище настроено в `./dvc_storage`.

---

## Логирование и реестр моделей (MLflow)

При запуске `train_model.py` гиперпараметры, метрики ROC-AUC и артефакты модели автоматически логируются в **MLflow** (`mlflow.db` и `mlflow_registry.db`). Модель регистрируется под именем `scorify-lgbm`.

Для просмотра UI экспериментов выполните:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Затем откройте в браузере `http://127.0.0.1:5000`.

---

## Тестирование и проверка качества кода

В проекте используются `ruff` для статического анализа и `pytest` для автотестов:

```bash
# Проверка качества кода и линтинг
ruff check .

# Запуск полного набора автотестов
pytest tests/ -v
```

---

## Запуск в Docker

### Сборка и запуск одиночного контейнера

```bash
docker build -t scorify:latest .
docker run -p 8000:8000 scorify:latest
```

### Запуск через Docker Compose

```bash
docker compose up --build
```

Сервис будет доступен по адресу `http://localhost:8000`.

---

## Непрерывная интеграция (CI/CD)

В репозитории настроен GitHub Actions workflow (`.github/workflows/ci.yml`), который при каждом `push` и `pull_request`:
1. Устанавливает Java 21 и Python 3.11.
2. Устанавливает зависимости из `requirements.txt`.
3. Выполняет статический анализ кода с помощью `ruff check .`.
4. Восстанавливает данные DVC либо генерирует синтетический датасет и воспроизводит конвейер обучения при отсутствии доступа к DVC remote.
5. Запускает интеграционные и юнит-тесты `pytest tests/ -v`.
6. Собирает тестовый Docker-образ сервиса.
