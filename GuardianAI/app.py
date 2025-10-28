from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import catboost
import re

# === Инициализация приложения ===
app = FastAPI(title="Guardian AI NLP", description="Анти-РПП NLP на CatBoost", version="1.0")

# === Загрузка модели ===
model = catboost.CatBoostClassifier()
model.load_model("rpp_detector.cbm")

# === Загрузка триггеров ===
trigger_df = pd.read_csv("triggers.csv")
TRIGGERS = [str(t).strip().lower() for t in trigger_df["trigger"].tolist()]

def has_trigger(text: str) -> int:
    text_lower = text.lower()
    return int(any(re.search(t, text_lower) for t in TRIGGERS))

# === Схема запроса ===
class Post(BaseModel):
    text: str

# === Эндпоинт предсказания ===
@app.post("/predict")
def predict(post: Post):
    flag = has_trigger(post.text)
    length = len(post.text)
    X = [[flag, length]]
    label = model.predict(X)[0]
    return {
        "label": str(label),
        "trigger_flag": flag,
        "text_length": length
    }

# === Домашняя страница ===
@app.get("/")
def home():
    return {"message": "Guardian AI NLP API работает. Перейди на /docs для теста."}
