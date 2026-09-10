import pandas as pd
import re
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

# ==== 0. Upload dataset (muncul tombol pilih file, pilih data_set_mdn_praproses.xlsx) ====
from google.colab import files
uploaded = files.upload()

# ==== 1. Baca dataset ====
FILE_PATH = "data_set_mdn_praproses.xlsx"  # otomatis sesuai nama file yang diupload
df = pd.read_excel(FILE_PATH)
df["TANGGAL PERMOHONAN"] = pd.to_datetime(df["TANGGAL PERMOHONAN"])

# ==== 2. Volume kunjungan per hari ====
per_day = df.groupby(df["TANGGAL PERMOHONAN"].dt.date).size()

jumlah_hari_valid = len(per_day)
jumlah_total_data = int(per_day.sum())
volume_min = int(per_day.min())
volume_max = int(per_day.max())
volume_mean = per_day.mean()
q1 = per_day.quantile(0.25, interpolation="linear")
median = per_day.quantile(0.50, interpolation="linear")
q3 = per_day.quantile(0.75, interpolation="linear")
BULAN_ID = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
            "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
def format_tanggal_id(ts):
    return f"{ts.day:02d} {BULAN_ID[ts.month - 1]} {ts.year}"

tanggal_awal = format_tanggal_id(df["TANGGAL PERMOHONAN"].min())
tanggal_akhir = format_tanggal_id(df["TANGGAL PERMOHONAN"].max())

# ==== 3. Kecamatan & kelurahan ====
# Diambil KEC. terakhir pada teks alamat, karena sebagian alamat memiliki
# format ganda/duplikat sehingga kemunculan pertama "KEC." bisa salah tangkap.
def ambil_kecamatan(alamat):
    hasil = re.findall(r"KEC\.([^,]+)", str(alamat))
    return hasil[-1].strip() if hasil else None

df["KECAMATAN"] = df["ALAMAT INSTALASI"].apply(ambil_kecamatan)
jumlah_kecamatan = df["KECAMATAN"].nunique()
jumlah_kelurahan = df["Nama Kelurahan"].nunique()

# ==== 4. Cetak hasil (Tabel 13) ====
print("=== Tabel 13. Statistik Deskriptif Volume Kunjungan Harian ===")
print(f"Jumlah hari valid       : {jumlah_hari_valid} hari")
print(f"Jumlah total data       : {jumlah_total_data} data")
print(f"Rentang waktu           : {tanggal_awal} - {tanggal_akhir}")
print(f"Volume minimum per hari : {volume_min} titik")
print(f"Volume maksimum per hari: {volume_max} titik")
print(f"Rata-rata volume/hari   : {volume_mean:.1f} titik")
print(f"Kuartil 1 (Q1)          : {q1:.0f} titik")
print(f"Median                  : {median:.0f} titik")
print(f"Kuartil 3 (Q3)          : {q3:.0f} titik")
print(f"Jumlah kecamatan        : {jumlah_kecamatan} kecamatan")
print(f"Jumlah kelurahan        : {jumlah_kelurahan} kelurahan")

# ==== 4b. Detail Tambahan Distribusi Volume Kunjungan Harian ====
# -- Jumlah hari valid per bin histogram (bin sama persis dengan Gambar 2) --
bin_edges = list(range(5, 92, 5))
bin_counts, _ = np.histogram(per_day.values, bins=bin_edges)

print("\n=== Jumlah Hari Valid per Bin Histogram (sesuai Gambar 2) ===")
for i in range(len(bin_counts)):
    print(f"{bin_edges[i]}-{bin_edges[i+1]} titik : {bin_counts[i]} hari")

# -- Skewness distribusi volume kunjungan harian --
skewness_volume = skew(per_day.values)
print(f"\nSkewness distribusi volume harian : {skewness_volume:.3f}")

# -- Lima hari dengan volume kunjungan tertinggi --
top5_hari = per_day.sort_values(ascending=False).head(5)
print("\n=== Lima Hari dengan Volume Kunjungan Tertinggi ===")
for tanggal, jumlah in top5_hari.items():
    tanggal_ts = pd.Timestamp(tanggal)
    print(f"{format_tanggal_id(tanggal_ts)} : {jumlah} titik")

# -- Jumlah dan persentase hari di atas ambang 40 dan 50 titik --
n_above_40 = int((per_day > 40).sum())
pct_above_40 = n_above_40 / jumlah_hari_valid * 100
n_above_50 = int((per_day > 50).sum())
pct_above_50 = n_above_50 / jumlah_hari_valid * 100
print(f"\nJumlah hari dengan volume > 40 titik/hari : {n_above_40} hari ({pct_above_40:.2f}%)")
print(f"Jumlah hari dengan volume > 50 titik/hari : {n_above_50} hari ({pct_above_50:.2f}%)")

# -- Distribusi jumlah data per kecamatan (lima terbanyak) --
kecamatan_counts = df["KECAMATAN"].value_counts()
top5_kecamatan = kecamatan_counts.head(5)
print("\n=== Lima Kecamatan dengan Jumlah Data Terbanyak ===")
for kec, jumlah in top5_kecamatan.items():
    pct = jumlah / jumlah_total_data * 100
    print(f"{kec} : {jumlah} data ({pct:.2f}%)")

# ==== 6. Analisis Duplikasi Koordinat pada Lima Hari Volume Tertinggi ====
df["lat_r"] = df["latitude"].round(6)
df["lon_r"] = df["longitude"].round(6)

rows_koordinat = []
for tanggal, jumlah in top5_hari.items():
    sub = df[df["TANGGAL PERMOHONAN"].dt.date == tanggal]
    n_titik = len(sub)
    coord_unik = sub[["lat_r", "lon_r"]].drop_duplicates().shape[0]
    counts_coord = sub.groupby(["lat_r", "lon_r"]).size().sort_values(ascending=False)
    modus_coord = counts_coord.index[0]
    n_modus = int(counts_coord.iloc[0])
    pct_modus = n_modus / n_titik * 100
    contoh_alamat = sub[
        (sub["lat_r"] == modus_coord[0]) & (sub["lon_r"] == modus_coord[1])
    ]["ALAMAT INSTALASI"].iloc[0]
    rows_koordinat.append([
        format_tanggal_id(pd.Timestamp(tanggal)),
        n_titik,
        coord_unik,
        n_modus,
        f"{pct_modus:.2f}%",
        contoh_alamat
    ])

tabel_koordinat = pd.DataFrame(
    rows_koordinat,
    columns=["Tanggal", "Jumlah Titik", "Koordinat Unik", "Titik pada Modus", "Persentase Modus", "Contoh Alamat"]
)
pd.set_option("display.max_colwidth", None)
print("\n=== Analisis Duplikasi Koordinat pada Lima Hari Volume Tertinggi ===")
print(tabel_koordinat.to_string(index=False))

# ==== 5. Gambar 2. Distribusi Volume Kunjungan Harian ====
plt.figure(figsize=(6.5, 4))
plt.hist(per_day.values, bins=range(5, 92, 5), color="#4472C4", edgecolor="white")
plt.title("Distribusi Volume Kunjungan Harian (224 Hari)")
plt.xlabel("Jumlah Titik Kunjungan per Hari")
plt.ylabel("Jumlah Hari")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("gambar2_distribusi_volume_harian.png", dpi=200)
plt.show()