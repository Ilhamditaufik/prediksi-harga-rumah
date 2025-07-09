import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def train_model(df):
    # Encode house_type ke angka
    house_type_mapping = {"Kecil": 0, "Sedang": 1, "Besar": 2}
    df["house_type_encoded"] = df["house_type"].map(house_type_mapping)

    # Pilih fitur
    X = df[["bed", "bath", "listing-floorarea", "listing-floorarea 2", "house_type_encoded"]]
    y = df["price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    mse = mean_squared_error(y, model.predict(X_scaled))

    return model, mse, scaler, X.columns.tolist()
