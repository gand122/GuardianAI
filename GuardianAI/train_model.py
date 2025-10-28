import pandas as pd
import re
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# === Загрузка списка триггеров ===
trigger_df = pd.read_csv("triggers.csv")
TRIGGERS = [str(t).strip().lower() for t in trigger_df["trigger"].tolist()]

def has_trigger(text: str) -> int:
    """Проверка наличия триггеров в тексте"""
    text_lower = text.lower()
    return int(any(re.search(t, text_lower) for t in TRIGGERS))

# === Загрузка обучающих данных ===
df = pd.read_csv("posts.csv")

# === Извлечение фич ===
df["trigger_flag"] = df["text"].apply(has_trigger)
df["text_length"] = df["text"].apply(len)

X = df[["trigger_flag", "text_length"]]
y = df["label"]

# === Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# === Обучение модели CatBoost ===
model = CatBoostClassifier(
    iterations=200,
    depth=5,
    learning_rate=0.1,
    loss_function="MultiClass",
    verbose=False
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# === Сохранение модели ===
model.save_model("rpp_detector.cbm")
print("✅ Модель обучена и сохранена как rpp_detector.cbm")
