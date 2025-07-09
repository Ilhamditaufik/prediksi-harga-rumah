import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from data_loader import load_rumah_indonesia
from model import train_model
from fpdf import FPDF

st.set_page_config(page_title="Prediksi Harga Rumah Indonesia", layout="wide")

st.title("🏠 Prediksi Harga Rumah Indonesia")
st.markdown("""
**Disclaimer:** Dataset ini bersifat simulasi dan hanya untuk pembelajaran.
""")

# Load data
try:
    df = load_rumah_indonesia()
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

if df.empty:
    st.error("❌ DataFrame kosong! Pastikan data sudah dimuat dengan benar.")
    st.stop()

st.write("🔍 DataFrame Awal:", df.head())
st.write("🔍 Dtypes:", df.dtypes)

# Latih model
model, mse, scaler, feature_names = train_model(df)

# Sidebar input
st.sidebar.header("Input Fitur Rumah")

house_type = st.sidebar.selectbox(
    "Tipe Rumah",
    ["Kecil", "Sedang", "Besar"]
)

house_type_mapping = {"Kecil": 0, "Sedang": 1, "Besar": 2}
house_type_encoded = house_type_mapping[house_type]

location_options = df["listing-location"].unique()
selected_location = st.sidebar.selectbox("Lokasi:", location_options)

location_row = df[df["listing-location"] == selected_location].iloc[0]
latitude = location_row["Latitude"]
longitude = location_row["Longitude"]

# Input fitur numerik
features = {}
feature_columns = ["bed", "bath", "listing-floorarea", "listing-floorarea 2"]

for col in feature_columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())

    if min_val == max_val:
        min_val -= 1
        max_val += 1

    features[col] = st.sidebar.slider(
        col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        step=1.0
    )

# Tambahkan house_type_encoded ke fitur prediksi
features["house_type_encoded"] = house_type_encoded

# Prediksi
if st.button("🔮 Prediksi Harga"):
    X_pred = pd.DataFrame([features])[feature_names]
    input_scaled = scaler.transform(X_pred)
    prediction = model.predict(input_scaled)[0]

    st.session_state["prediction"] = prediction
    st.session_state["features"] = features
    st.session_state["latitude"] = latitude
    st.session_state["longitude"] = longitude
    st.session_state["selected_location"] = selected_location
    st.session_state["house_type"] = house_type

# Tampilkan hasil
if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    features = st.session_state["features"]
    latitude = st.session_state["latitude"]
    longitude = st.session_state["longitude"]
    selected_location = st.session_state["selected_location"]
    house_type = st.session_state["house_type"]

    st.success(f"💰 Prediksi Harga Rumah: **Rp {prediction:,.0f}** di {selected_location} (Tipe: {house_type})")

    st.bar_chart(pd.DataFrame([features]).drop(columns="house_type_encoded").T)

    st.subheader("📄 Data Historis Contoh Rumah")
    st.dataframe(df.head())

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Laporan Prediksi Harga Rumah", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Lokasi: {selected_location}", ln=True)
    pdf.cell(0, 10, f"Tipe Rumah: {house_type}", ln=True)
    for k, v in features.items():
        if k != "house_type_encoded":
            pdf.cell(0, 10, f"{k}: {v}", ln=True)
    pdf.cell(0, 10, f"Prediksi Harga: Rp {prediction:,.0f}", ln=True)

    pdf_file = "report.pdf"
    pdf.output(pdf_file)

    with open(pdf_file, "rb") as f:
        st.download_button("📄 Unduh Laporan PDF", f, file_name="prediksi_harga.pdf")

    st.subheader("🗺️ Lokasi Rumah pada Peta")
    m = folium.Map(location=[latitude, longitude], zoom_start=13)
    folium.Marker(
        [latitude, longitude],
        popup=f"{selected_location} ({house_type})",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(m)
    st_folium(m, width=700, height=450)
