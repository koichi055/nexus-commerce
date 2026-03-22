import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib

mlflow.set_tracking_uri("file:///Users/koichi/nexus-commerce/mlruns")

# ── Cargar datos limpios ─────────────────────────────────────────────────────
df = pd.read_csv("../data/processed/books_clean.csv")

# ── Features y target ────────────────────────────────────────────────────────
X = df[["price"]]
y = df["rating"]

# ── Split train/test ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Experimento MLflow ───────────────────────────────────────────────────────
mlflow.set_experiment("books-rating-predictor")

with mlflow.start_run():
    # Parámetros del modelo
    n_estimators = 100
    max_depth = 5

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Accuracy: {accuracy:.2%}")
    print("\nReporte completo:")
    print(classification_report(y_test, y_pred))

    # Guardar modelo localmente
    joblib.dump(model, "../models/rating_predictor.pkl")
    print("Modelo guardado en models/rating_predictor.pkl")

print("\nExperimento guardado en MLflow.")
print("Para ver el dashboard de MLflow corre: mlflow ui")