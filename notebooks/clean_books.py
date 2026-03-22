import pandas as pd
import json

# ── Cargar datos crudos ──────────────────────────────────────────────────────
with open("../data/raw/books.json", "r") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)

# ── Limpiar precio ───────────────────────────────────────────────────────────
df["price"] = df["price"].str.replace("£", "").astype(float)

# ── Limpiar rating ───────────────────────────────────────────────────────────
rating_map = {
    "star-rating One":   1,
    "star-rating Two":   2,
    "star-rating Three": 3,
    "star-rating Four":  4,
    "star-rating Five":  5,
}
df["rating"] = df["rating"].map(rating_map)

# ── Limpiar availability ─────────────────────────────────────────────────────
df["availability"] = df["availability"].apply(
    lambda x: "In stock" if "In stock" in " ".join(x) else "Out of stock"
)

# ── Ver resultado ────────────────────────────────────────────────────────────
print("Dataset limpio:")
print(df.head(5))
print("\nTipos de datos:")
print(df.dtypes)
print("\nEstadisticas:")
print(df.describe())

# ── Guardar dataset limpio ───────────────────────────────────────────────────
df.to_csv("../data/processed/books_clean.csv", index=False)
print("\nGuardado en data/processed/books_clean.csv")