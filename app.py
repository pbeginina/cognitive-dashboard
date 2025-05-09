import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Заголовок
st.title("Оценка когнитивных показателей")

st.markdown("Введите свои данные, чтобы получить прогноз когнитивных показателей и рекомендации.")

# Ввод данных
age = st.number_input("Возраст", min_value=10, max_value=100, step=1)
gender = st.selectbox("Пол", ["Male", "Female"])
height_cm = st.number_input("Рост (в см)", min_value=100, max_value=250, step=1)
weight_kg = st.number_input("Вес (в кг)", min_value=30, max_value=200, step=1)
sleep_hours = st.slider("Количество часов сна в сутки", 0.0, 12.0, 7.0, step=0.1)
caffeine = st.slider("Кофеин (кол-во кружек кофе в день)", 0, 10, 2)
activity = st.slider("Физическая активность (часы в неделю)", 0.0, 20.0, 3.0, step=0.5)

# Вычисление BMI
height_m = height_cm / 100
bmi = weight_kg / (height_m ** 2)
bmi_category = "Нормальный"
if bmi < 18.5:
    bmi_category = "Недовес"
elif bmi > 25:
    bmi_category = "Избыточный"

st.markdown(f"**Ваш ИМТ (BMI):** {bmi:.1f} — *{bmi_category}*")

# Создание DataFrame
X_input = pd.DataFrame({
    "Sleep_Hours": [sleep_hours],
    "Caffeine_Intake": [caffeine],
    "Physical_Activity_Level": [activity],
    "BMI": [bmi],
    "Age": [age]
})

# Подготовка модели (упрощённо: обучаем в коде)
# Пример обучающих данных — подменить своими в рабочем приложении
X_fake = pd.DataFrame({
    "Sleep_Hours": np.random.uniform(4, 9, 100),
    "Caffeine_Intake": np.random.randint(0, 6, 100),
    "Physical_Activity_Level": np.random.uniform(0, 10, 100),
    "BMI": np.random.uniform(18, 35, 100),
    "Age": np.random.randint(18, 65, 100)
})

y_fake = pd.DataFrame({
    "Daytime_Sleepiness": np.random.uniform(0, 24, 100),
    "Stroop_Task_Reaction_Time": np.random.uniform(2, 5, 100),
    "N_Back_Accuracy": np.random.uniform(50, 100, 100),
    "PVT_Reaction_Time": np.random.uniform(250, 500, 100)
})

# Масштабируем
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fake)
X_input_scaled = scaler.transform(X_input)

# Обучаем модели
models = {
    'Daytime_Sleepiness': Ridge().fit(X_scaled, y_fake['Daytime_Sleepiness']),
    'Stroop': Ridge().fit(X_scaled, y_fake['Stroop_Task_Reaction_Time']),
    'N_Back': Ridge().fit(X_scaled, y_fake['N_Back_Accuracy']),
    'PVT': Ridge().fit(X_scaled, y_fake['PVT_Reaction_Time'])
}

# Предсказания
preds = {name: model.predict(X_input_scaled)[0] for name, model in models.items()}

# Вывод
st.subheader("Предполагаемые когнитивные показатели:")
st.write(f"**Дневная сонливость:** {preds['Daytime_Sleepiness']:.1f}")
st.write(f"**Время реакции (Stroop):** {preds['Stroop']:.2f} сек")
st.write(f"**Точность памяти (N-Back Accuracy):** {preds['N_Back']:.1f}%")
st.write(f"**Время реакции (PVT):** {preds['PVT']:.0f} мс")

# Рекомендации
st.subheader("💬 Рекомендации:")

all_ok = True

# Сон
if sleep_hours < 6:
    st.warning("Вы спите меньше 6 часов. Увеличьте продолжительность сна до 6–9 часов для лучшей концентрации и реакции.")
    all_ok = False
elif sleep_hours > 9:
    st.warning("Вы спите больше 9 часов. Слишком длинный сон может вызывать вялость. Старайтесь спать 6–9 часов.")
    all_ok = False

# ИМТ
if bmi_category == "Недовес":
    st.warning("Ваш вес ниже нормы. Попробуйте откорректировать рацион или проконсультироваться со специалистом.")
    all_ok = False
elif bmi_category == "Избыточный":
    st.warning("Ваш вес выше нормы. Оптимальный BMI — 18.5–25.")
    all_ok = False

# Кофеин
if caffeine > 3:
    st.warning("Вы пьёте много кофе. Ограничьте потребление до 1–3 кружек в день — это может улучшить качество сна.")
    all_ok = False

# Физическая активность
if activity < 2:
    st.warning("Низкая физическая активность. Рекомендуется минимум 2–6 часов активности в неделю для улучшения сна и когнитивных функций.")
    all_ok = False


if all_ok:
    st.success("Все ключевые параметры в хорошем диапазоне! Ваши показатели, скорее всего, находятся на оптимальном уровне.")


