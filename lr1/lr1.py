import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import LocalOutlierFactor
from optuna import create_study, Trial
from optuna.samplers import TPESampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Загрузка данных
data = fetch_california_housing()
X = data.data  # Признаки
y = data.target  # Целевая переменная (медианная стоимость дома)

# Feature Engineering: добавление полиномиальных признаков и преобразование Box-Cox
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Преобразование Box-Cox для стабилизации дисперсии
pt = PowerTransformer(method='box-cox', standardize=True)
X_transformed = pt.fit_transform(X_poly)

# Обработка выбросов с помощью LOF (Local Outlier Factor)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outliers = lof.fit_predict(X_transformed)
mask = outliers == 1
X_cleaned = X_transformed[mask]
y_cleaned = y[mask]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Стандартизация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Оптимизация гиперпараметров ансамблей с помощью Optuna
def objective(trial: Trial):
    # Гиперпараметры для XGBoost
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 300),
        "learning_rate": trial.suggest_loguniform("xgb_learning_rate", 1e-3, 1e-1),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
    }
    
    # Гиперпараметры для LightGBM
    lgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 300),
        "learning_rate": trial.suggest_loguniform("lgbm_learning_rate", 1e-3, 1e-1),
        "max_depth": trial.suggest_int("lgbm_max_depth", 3, 10),
    }
    
    # Гиперпараметры для CatBoost
    catboost_params = {
        "iterations": trial.suggest_int("catboost_iterations", 100, 300),
        "learning_rate": trial.suggest_loguniform("catboost_learning_rate", 1e-3, 1e-1),
        "depth": trial.suggest_int("catboost_depth", 3, 10),
    }
    
    # Гиперпараметры для GradientBoostingRegressor
    gb_params = {
        "n_estimators": trial.suggest_int("gb_n_estimators", 100, 300),
        "learning_rate": trial.suggest_loguniform("gb_learning_rate", 1e-3, 1e-1),
        "max_depth": trial.suggest_int("gb_max_depth", 3, 10),
    }
    
    # Гиперпараметры для RandomForestRegressor
    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 300),
        "max_depth": trial.suggest_int("rf_max_depth", 3, 10),
    }
    
    # Создание моделей
    models = [
        XGBRegressor(**xgb_params),
        LGBMRegressor(**lgbm_params),
        CatBoostRegressor(**catboost_params, verbose=0, random_state=42),
        GradientBoostingRegressor(**gb_params, random_state=42),
        RandomForestRegressor(**rf_params, random_state=42)
    ]
    
    # Stacking ансамбль
    meta_model = Ridge(alpha=trial.suggest_loguniform("ridge_alpha", 1e-3, 1e1))
    stacking_regressor = StackingCVRegressor(
        regressors=models,
        meta_regressor=meta_model,
        cv=5,
        use_features_in_secondary=True
    )
    
    # Кросс-валидация
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(stacking_regressor, X_train, y_train, scoring="neg_mean_squared_error", cv=kf)
    return -np.mean(scores)

# Создание и выполнение исследования Optuna
study = create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)

# Лучшие гиперпараметры
best_params = study.best_params
print("Лучшие гиперпараметры:", best_params)

# Создание финального ансамбля с лучшими гиперпараметрами
models = [
    XGBRegressor(n_estimators=best_params["xgb_n_estimators"], learning_rate=best_params["xgb_learning_rate"], max_depth=best_params["xgb_max_depth"]),
    LGBMRegressor(n_estimators=best_params["lgbm_n_estimators"], learning_rate=best_params["lgbm_learning_rate"], max_depth=best_params["lgbm_max_depth"]),
    CatBoostRegressor(iterations=best_params["catboost_iterations"], learning_rate=best_params["catboost_learning_rate"], depth=best_params["catboost_depth"], verbose=0, random_state=42),
    GradientBoostingRegressor(n_estimators=best_params["gb_n_estimators"], learning_rate=best_params["gb_learning_rate"], max_depth=best_params["gb_max_depth"], random_state=42),
    RandomForestRegressor(n_estimators=best_params["rf_n_estimators"], max_depth=best_params["rf_max_depth"], random_state=42)
]

meta_model = Ridge(alpha=best_params["ridge_alpha"])
stacking_regressor = StackingCVRegressor(
    regressors=models,
    meta_regressor=meta_model,
    cv=5,
    use_features_in_secondary=True
)

# Обучение и предсказание ансамбля
stacking_regressor.fit(X_train, y_train)
y_pred_ensemble = stacking_regressor.predict(X_test)

# Оценка качества
mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

print(f"Среднеквадратичная ошибка (MSE) ансамбля: {mse_ensemble:.4f}")
print(f"Коэффициент детерминации (R²) ансамбля: {r2_ensemble:.4f}")

# Создание и обучение нейронной сети
def build_nn():
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

nn_model = build_nn()
history = nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=5)])

# Предсказание на тестовых данных
y_pred_nn = nn_model.predict(X_test).flatten()

# Оценка качества нейронной сети
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print(f"Среднеквадратичная ошибка (MSE) нейронной сети: {mse_nn:.4f}")
print(f"Коэффициент детерминации (R²) нейронной сети: {r2_nn:.4f}")

# Визуализация результатов
plt.figure(figsize=(12, 6))

# Ансамбль
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ensemble, alpha=0.7, label="Ансамбль", color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Линия идеального совпадения
plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.title("Ансамбль")
plt.legend()

# Нейронная сеть
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_nn, alpha=0.7, label="Нейронная сеть", color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Линия идеального совпадения
plt.xlabel("Истинные значения")
plt.ylabel("Предсказанные значения")
plt.title("Нейронная сеть")
plt.legend()

plt.tight_layout()
plt.show()