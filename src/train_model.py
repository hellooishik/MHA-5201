import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def train_models(X_train, y_train):
    models = {}

    # 🔥 Logistic Regression Pipeline + Tuning
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])

    lr_params = {
        "model__C": [0.1, 1, 10],
        "model__max_iter": [1000, 2000]
    }

    lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=3, scoring="f1")
    lr_grid.fit(X_train, y_train)

    models["logistic"] = lr_grid.best_estimator_

    # 🌲 Random Forest (Tuned)
    rf = RandomForestClassifier()

    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }

    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring="f1")
    rf_grid.fit(X_train, y_train)

    models["random_forest"] = rf_grid.best_estimator_

    # ⚡ XGBoost (Tuned)
    xgb = XGBClassifier(eval_metric='logloss')

    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.1]
    }

    xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring="f1")
    xgb_grid.fit(X_train, y_train)

    models["xgboost"] = xgb_grid.best_estimator_

    return models


def save_models(models):
    for name, model in models.items():
        joblib.dump(model, f"models/{name}.pkl")