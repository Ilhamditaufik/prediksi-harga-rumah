import pandas as pd

def load_rumah_indonesia():
    """
    Load dataset rumah Indonesia yang sudah ada kolom Latitude dan Longitude.
    """
    df = pd.read_csv("rumah_70_kota.csv")
    return df


    # Bersihkan nama kolom
    df.columns = df.columns.str.strip()

    selected_columns = ["price", "bed", "bath", "listing-floorarea", "listing-floorarea 2"]

    # Konfirmasi semua kolom tersedia
    missing_cols = [col for col in selected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ada di CSV: {missing_cols}")

    df = df[selected_columns]

    # Konversi numerik
    df = df.apply(pd.to_numeric, errors="coerce")

    # Jika semua baris null, isi dummy
    if df.isna().all().all():
        raise ValueError("Semua kolom kosong di CSV!")

    # Isi NaN dengan nilai default agar tidak semua drop
    df = df.fillna({
        "price": 100000000,
        "bed": 3,
        "bath": 2,
        "listing-floorarea": 100,
        "listing-floorarea 2": 50
    })

    # Tambah dummy koordinat
    df["Latitude"] = -6.2
    df["Longitude"] = 106.8

    return df
