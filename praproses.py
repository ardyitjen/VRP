# =============================================================================
# Praproses Data Permohonan SLO -- VERSI 1 CELL UNTUK GOOGLE COLAB
# =============================================================================
# Cara pakai:
#   1. Jalankan cell ini (RUN).
#   2. Nanti akan diminta upload file Excel data mentah -> pilih file kamu.
#   3. Kalau punya file kamus koordinat manual (opsional), akan diminta upload
#      juga -- kalau tidak punya, tinggal klik Cancel/lewati.
#   4. Masukkan email kamu saat diminta (wajib, untuk User-Agent Nominatim).
#   5. Tunggu sampai selesai -- file hasil otomatis ke-download.
#
# TIDAK PERLU upload file .py terpisah. Paste SELURUH isi file ini ke 1 cell
# Colab, lalu jalankan.
#
# Format file kamus koordinat manual (opsional), kolom wajib persis ini:
#   alamat_instalasi, latitude, longitude
# Kolom alamat_instalasi harus cocok PERSIS (karakter demi karakter, setelah
# spasi/newline dirapikan otomatis) dengan ALAMAT INSTALASI di data mentah.
# =============================================================================

get_ipython().system('pip install openpyxl -q')  # aman meski openpyxl sudah terpasang

import re
import sys
import time

import pandas as pd
import requests

import re
import sys
import time

import pandas as pd
import requests

# =========================================================================
# KONFIGURASI
# =========================================================================
USER_AGENT = None  # diisi otomatis dari argumen --email saat script dijalankan
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
REQUEST_DELAY_SECONDS = 1.1          # Nominatim: maks. 1 request/detik, dikasih margin
REQUIRED_RAW_COLUMNS = [
    "TANGGAL PERMOHONAN",
    "NAMA INSTALASI",
    "ALAMAT INSTALASI",
    "Nama Kelurahan",
    "Daya",
    "NO REGISTRASI",  # ID unik asli -- dipakai untuk dedup, TIDAK dimasukkan ke output final
]
# Kolom opsional, kalau ada di file input akan dicek homogenitasnya sebagai sanity-check
# scope (bukan filter otomatis -- filter scope dilakukan manual sebelum file diserahkan).
SCOPE_CHECK_COLUMNS = ["AREA LIT", "STATUS PERMOHONAN"]


# =========================================================================
# 1. BACA DATA MENTAH
# =========================================================================
def load_raw(input_path: str, sheet: str | None) -> pd.DataFrame:
    if input_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path, sheet_name=sheet if sheet else 0)
    else:
        df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di file input: {missing}\n"
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    # Sanity-check scope (bukan filter otomatis) -- kasih tahu kalau data ternyata
    # bukan hasil filter satu-scope, supaya tidak terlewat tanpa disadari.
    for col in SCOPE_CHECK_COLUMNS:
        if col in df.columns:
            unik = df[col].dropna().unique()
            if len(unik) > 1:
                print(f"  [PERHATIAN] Kolom '{col}' TIDAK homogen: {list(unik)} "
                      f"-- pastikan ini memang disengaja.")

    return df


# =========================================================================
# 2. AMBIL KOLOM YANG DIPAKAI SAJA
# =========================================================================
def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[REQUIRED_RAW_COLUMNS].copy()
    # Paksa kolom teks jadi string (beberapa sel bisa terbaca sebagai angka oleh
    # pandas kalau isinya kebetulan berupa digit murni), tapi tetap pertahankan
    # NaN asli supaya drop_incomplete() masih bisa mendeteksi baris kosong.
    for col in ["NAMA INSTALASI", "ALAMAT INSTALASI", "Nama Kelurahan", "Daya", "NO REGISTRASI"]:
        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else x)

    # Bersihkan karakter newline/tab tersembunyi (ditemukan cukup banyak di ALAMAT
    # INSTALASI -- sel Excel sumber mengandung Alt+Enter). Kalau dibiarkan, field
    # ini tetap valid secara CSV (dikutip otomatis), tapi Excel sering salah
    # mengartikan newline di dalam sel sebagai baris baru saat file dibuka langsung
    # (double-click), sehingga jumlah baris tampak membengkak dan kolom bergeser.
    for col in ["NAMA INSTALASI", "ALAMAT INSTALASI", "Nama Kelurahan"]:
        df[col] = df[col].apply(lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x)

    return df


# =========================================================================
# 3. BUANG BARIS DENGAN FIELD KUNCI KOSONG
# =========================================================================
def drop_incomplete(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna(subset=["NAMA INSTALASI", "ALAMAT INSTALASI", "TANGGAL PERMOHONAN", "NO REGISTRASI"])
    df = df[df["NAMA INSTALASI"].astype(str).str.strip() != ""]
    df = df[df["ALAMAT INSTALASI"].astype(str).str.strip() != ""]
    df = df[df["NO REGISTRASI"].astype(str).str.strip() != ""]
    removed = before - len(df)
    return df, removed


# =========================================================================
# 4. DEDUPLIKASI (berdasarkan NO REGISTRASI -- ID unik asli)
# =========================================================================
def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Dedup memakai NO REGISTRASI (ID unik asli dari sistem SIUJANG Gatrik), BUKAN
    kombinasi nama+alamat+tanggal. Pendekatan nama+alamat+tanggal sempat dicoba dan
    terbukti keliru: baris dengan nama, alamat, dan tanggal permohonan yang identik
    ternyata pada praktiknya bisa berupa beberapa aplikasi SLO yang sah dan berbeda
    (instalasi berbeda diajukan bersamaan), dibuktikan lewat NO REGISTRASI dan
    NO SERTIFIKAT yang berbeda pada tiap baris. NO REGISTRASI dipertahankan sebagai
    kunci dedup karena bersifat unik per aplikasi SLO dan tidak bergantung pada
    kualitas penulisan nama/alamat yang bervariasi.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["NO REGISTRASI"], keep="first")
    removed = before - len(df)
    return df, removed


# =========================================================================
# 5. FILTER VOLUME HARIAN (buang hari dengan volume < 7 titik)
# =========================================================================
def filter_low_volume_days(df: pd.DataFrame, min_per_day: int = 7) -> tuple[pd.DataFrame, int, int]:
    """
    Buang seluruh baris pada hari (TANGGAL PERMOHONAN) yang jumlah permohonannya
    di bawah ambang kapasitas minimum (default 7 titik/hari), sesuai arahan dosen
    pembimbing. Hari dengan volume di bawah kapasitas satu petugas tidak relevan
    untuk pemodelan VRP karena tidak pernah membutuhkan lebih dari satu petugas
    dan tidak merepresentasikan kompleksitas rute yang menjadi fokus penelitian.

    Return: (df_terfilter, jumlah_hari_dibuang, jumlah_baris_dibuang)
    """
    counts_per_day = df.groupby("TANGGAL PERMOHONAN")["TANGGAL PERMOHONAN"].transform("size")
    mask = counts_per_day >= min_per_day
    n_hari_dibuang = df.loc[~mask, "TANGGAL PERMOHONAN"].nunique()
    n_baris_dibuang = int((~mask).sum())
    return df[mask].copy(), n_hari_dibuang, n_baris_dibuang


# =========================================================================
# 5. PARSING KECAMATAN / KOTA-KABUPATEN / PROVINSI DARI ALAMAT INSTALASI
# =========================================================================
def parse_admin_area(alamat: str) -> tuple[str | None, str | None, str | None]:
    """
    Beberapa ALAMAT INSTALASI mengandung teks ganda/rusak (mis. plus-code diikuti
    alamat informal yang diulang dalam format terstruktur DESA/KEL...,KEC...,KOTA/KAB...).
    Karena itu diambil KECOCOKAN TERAKHIR (bukan pertama) sebagai bagian yang valid.
    """
    alamat = str(alamat)
    kec_all = re.findall(r"KEC\.\s*([^,]+)", alamat, flags=re.IGNORECASE)
    kota_all = re.findall(r"KOTA\s+([^,]+)", alamat, flags=re.IGNORECASE)
    kab_all = re.findall(r"KAB\.\s*([^,]+)", alamat, flags=re.IGNORECASE)

    kecamatan = kec_all[-1].strip().title() if kec_all else None
    kota = kota_all[-1].strip().title() if kota_all else None
    kabupaten = kab_all[-1].strip().title() if kab_all else None

    parts = [p.strip() for p in alamat.split(",") if p.strip()]
    provinsi = parts[-1].title() if parts else None

    kota_or_kab = kota if kota else kabupaten
    return kecamatan, kota_or_kab, provinsi


_PLUS_CODE_RE = re.compile(r"\b[A-Z0-9]{4,8}\+[A-Z0-9]{2,4}\b", flags=re.IGNORECASE)


def extract_street(alamat: str) -> str | None:
    """
    Ambil bagian teks alamat SEBELUM penanda struktural 'DESA/KEL.' (bagian informal/jalan),
    lalu dibersihkan: buang Google Plus Code, rapikan singkatan umum (JL./GG./LK.),
    rapikan spasi dan tanda baca sisa.

    Best-effort saja -- alamat sumber memang tidak konsisten formatnya, jadi hasil
    fungsi ini tidak dijamin selalu bersih sempurna. Itu bagian dari limitasi yang
    perlu ditulis di Bab III.
    """
    alamat = str(alamat)
    before_desakel = re.split(r"DESA/KEL\.", alamat, maxsplit=1, flags=re.IGNORECASE)[0]

    s = _PLUS_CODE_RE.sub("", before_desakel)
    s = re.sub(r"\bJL\.{1,2}\s*", "Jalan ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bGG\.{1,2}\s*", "Gang ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bLK\.{1,2}\s*", "Lingkungan ", s, flags=re.IGNORECASE)
    s = re.sub(r"[-,]+", ",", s)
    s = re.sub(r"\s+", " ", s).strip(" ,-")

    # buang fragmen kosong/sisa kata "Jalan" sendirian akibat plus-code yang dihapus
    segments = [seg.strip() for seg in s.split(",")]
    segments = [seg for seg in segments if seg and seg.lower() != "jalan"]
    s = ", ".join(segments)

    return s if s else None


# =========================================================================
# 6. GEOCODING (NOMINATIM, TIGA LEVEL, DENGAN CACHE)
# =========================================================================
_geocode_cache: dict[str, dict] = {}


def geocode_query(query: str) -> dict | None:
    """Kirim satu request ke Nominatim (free-text). Return dict hasil pertama (dengan address breakdown), atau None kalau kosong.

    Cache hanya menyimpan hasil SUKSES (dict). Hasil kosong dari server dan
    kegagalan jaringan sengaja TIDAK di-cache, supaya kelompok alamat yang
    kebetulan gagal karena timeout sesaat masih diberi kesempatan berhasil
    di percobaan berikutnya (lewat baris lain yang share teks query sama).
    """
    if query in _geocode_cache:
        return _geocode_cache[query]

    params = {"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "id", "addressdetails": 1}
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException as e:
        print(f"  [WARNING] Gagal request untuk query '{query}': {e}", file=sys.stderr)
        time.sleep(REQUEST_DELAY_SECONDS)
        return None  # jangan di-cache -- biar bisa dicoba lagi lewat baris lain

    time.sleep(REQUEST_DELAY_SECONDS)  # patuhi rate limit Nominatim

    if not results:
        return None  # server balas kosong -- juga tidak di-cache

    result = results[0]
    _geocode_cache[query] = result
    return result


def geocode_structured(street: str | None, kelurahan: str | None, kecamatan: str | None,
                        kota_or_kab: str | None, provinsi: str | None) -> dict | None:
    """Kirim structured query ke Nominatim (field terpisah, bukan free-text).

    Nominatim mendukung pencarian dengan komponen terpisah -- kadang berhasil
    menemukan alamat jalan yang gagal ditemukan lewat pencarian free-text
    karena algoritma pencocokannya bekerja berbeda. Dipakai sebagai percobaan
    tambahan di tingkat pencarian jalan.
    """
    params = {"format": "jsonv2", "limit": 1, "countrycodes": "id", "addressdetails": 1}
    if street:
        params["street"] = street
    if kelurahan or kecamatan:
        params["city"] = kelurahan or kecamatan
    if kota_or_kab:
        params["county"] = kota_or_kab
    if provinsi:
        params["state"] = provinsi
    cache_key = "STRUCT|" + "|".join(str(params.get(k, "")) for k in ["street", "city", "county", "state"])
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        results = resp.json()
    except requests.RequestException as e:
        print(f"  [WARNING] Gagal structured request: {e}", file=sys.stderr)
        time.sleep(REQUEST_DELAY_SECONDS)
        return None

    time.sleep(REQUEST_DELAY_SECONDS)
    if not results:
        return None
    _geocode_cache[cache_key] = results[0]
    return results[0]


def _is_street_level_match(hasil: dict) -> bool:
    """
    Nominatim q= bersifat fuzzy: kalau nama jalan tidak ada di OSM, dia tetap bisa
    mengembalikan hasil berdasarkan token lain (kelurahan/kecamatan) di query gabungan,
    TANPA memberi tahu bahwa jalannya tidak ketemu. Karena itu, hasil hanya dipercaya
    sebagai match tingkat jalan kalau address breakdown-nya benar-benar memuat
    komponen 'road' / 'pedestrian' / 'house_number'.
    """
    addr = hasil.get("address", {}) or {}
    return bool(addr.get("road") or addr.get("pedestrian") or addr.get("house_number"))


# Kamus koordinat manual (opsional): dict alamat_asli -> (lat, lon).
# Kalau ada isinya, akan dicek DULUAN sebelum geocoding otomatis jalan.
# Diisi dari file CSV terpisah lewat load_manual_overrides().
_manual_overrides: dict[str, tuple[float, float]] = {}


def load_manual_overrides(csv_path: str | None) -> int:
    """Muat kamus koordinat manual dari file CSV. Format kolom wajib:
       alamat_instalasi, latitude, longitude
    Kolom 'alamat_instalasi' dicocokkan persis (exact match, case-sensitive)
    dengan ALAMAT INSTALASI di data mentah.
    Return: jumlah entri berhasil dimuat.
    """
    global _manual_overrides
    _manual_overrides = {}
    if not csv_path:
        return 0
    try:
        m = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  [INFO] File kamus manual '{csv_path}' tidak ada, dilewati.")
        return 0
    req = {"alamat_instalasi", "latitude", "longitude"}
    if not req.issubset(set(m.columns)):
        raise ValueError(f"Kamus manual harus punya kolom: {req}. Ditemukan: {list(m.columns)}")
    m = m.dropna(subset=["alamat_instalasi", "latitude", "longitude"])
    for _, row in m.iterrows():
        alamat = re.sub(r"\s+", " ", str(row["alamat_instalasi"])).strip()
        try:
            _manual_overrides[alamat] = (float(row["latitude"]), float(row["longitude"]))
        except (ValueError, TypeError):
            continue
    return len(_manual_overrides)


def geocode_row(street: str | None, kelurahan: str, kecamatan: str | None,
                kota_or_kab: str | None, provinsi: str | None,
                alamat_asli: str | None = None) -> dict:
    """
    Empat tingkat percobaan, berurutan (berhenti begitu salah satu ketemu):
      0. manual_override            -> koordinat dari kamus manual (alamat cocok persis)
      1. found_alamat_jalan         -> teks jalan hasil bersihan + kelurahan/kecamatan/kota/provinsi
                                        dicoba dua cara: free-text DAN structured query
      2. found_primary              -> kelurahan + kecamatan/kota/provinsi (centroid kelurahan)
      3. found_fallback_kecamatan   -> kecamatan + kota/provinsi (centroid kecamatan)
    """
    # Tingkat 0: kamus manual (paling akurat, paling diprioritaskan)
    if alamat_asli and alamat_asli in _manual_overrides:
        lat, lon = _manual_overrides[alamat_asli]
        return {
            "latitude": lat,
            "longitude": lon,
            "status_geocoding": "manual_override",
            "query_final": alamat_asli,
            "sumber_koordinat": "Kamus koordinat manual (input peneliti)",
        }

    kelurahan = str(kelurahan).strip().title() if pd.notna(kelurahan) else None

    query_jalan = ", ".join([p for p in [street, kelurahan, kecamatan, kota_or_kab, provinsi, "Indonesia"] if p])
    query_utama = ", ".join([p for p in [kelurahan, kecamatan, kota_or_kab, provinsi, "Indonesia"] if p])
    query_fallback = ", ".join([p for p in [kecamatan, kota_or_kab, provinsi, "Indonesia"] if p])

    if street:
        # Percobaan 1a: free-text combined query
        hasil = geocode_query(query_jalan)
        if hasil and _is_street_level_match(hasil):
            return {
                "latitude": float(hasil["lat"]),
                "longitude": float(hasil["lon"]),
                "status_geocoding": "found_alamat_jalan",
                "query_final": query_jalan,
                "sumber_koordinat": "Nominatim OpenStreetMap - hasil pencarian alamat jalan (free-text)",
            }
        # Percobaan 1b: structured query (field terpisah) -- kadang berhasil di kasus 1a gagal
        hasil2 = geocode_structured(street, kelurahan, kecamatan, kota_or_kab, provinsi)
        if hasil2 and _is_street_level_match(hasil2):
            return {
                "latitude": float(hasil2["lat"]),
                "longitude": float(hasil2["lon"]),
                "status_geocoding": "found_alamat_jalan_structured",
                "query_final": f"street={street}|city={kelurahan or kecamatan}|county={kota_or_kab}|state={provinsi}",
                "sumber_koordinat": "Nominatim OpenStreetMap - hasil pencarian alamat jalan (structured query)",
            }

    hasil = geocode_query(query_utama)
    if hasil:
        return {
            "latitude": float(hasil["lat"]),
            "longitude": float(hasil["lon"]),
            "status_geocoding": "found_primary",
            "query_final": query_utama,
            "sumber_koordinat": "Nominatim OpenStreetMap - centroid administratif kelurahan",
        }

    hasil = geocode_query(query_fallback)
    if hasil:
        return {
            "latitude": float(hasil["lat"]),
            "longitude": float(hasil["lon"]),
            "status_geocoding": "found_fallback_kecamatan",
            "query_final": query_fallback,
            "sumber_koordinat": "Nominatim OpenStreetMap - centroid administratif kecamatan",
        }

    return {
        "latitude": None,
        "longitude": None,
        "status_geocoding": "not_found",
        "query_final": query_fallback,
        "sumber_koordinat": None,
    }


def generate_customer_code(n: int, prefix: str = "Pelanggan", width: int = 4) -> list[str]:
    """Buat kode pelanggan berurutan: Pelanggan0001, Pelanggan0002, ..., pengganti nama asli."""
    return [f"{prefix}{i:0{width}d}" for i in range(1, n + 1)]


# =========================================================================
# PIPELINE (bisa dipanggil langsung sebagai fungsi -- disarankan untuk Colab/notebook)
# =========================================================================
def run(input_path: str, email: str, output: str = "data_bersih", sheet: str | None = None,
        manual_overrides_path: str | None = None, min_per_day: int = 7):
    """
    Jalankan seluruh pipeline praproses + geocoding.

    manual_overrides_path (opsional): path CSV kamus koordinat manual dengan kolom
        alamat_instalasi, latitude, longitude
    Alamat yang cocok persis dengan ALAMAT INSTALASI akan pakai koordinat ini,
    TANPA lewat geocoding otomatis sama sekali (paling diprioritaskan).

    min_per_day: ambang kapasitas minimum titik/hari (default 7). Hari dengan
        volume di bawah ambang ini dibuang total dari dataset (arahan dosen
        pembimbing), karena tidak pernah membutuhkan lebih dari satu petugas.

    Contoh pemakaian di Colab / Jupyter (tanpa command line sama sekali):
        import praproses_slo as pp
        pp.run(input_path="data_mentah_4464.csv", email="emailkamu@gmail.com", output="data_bersih")
    """
    global USER_AGENT
    USER_AGENT = f"TA-SLO-Routing-Ardy/1.0 ({email})"

    n_manual = load_manual_overrides(manual_overrides_path)
    if n_manual:
        print(f"[0/8] Kamus koordinat manual dimuat: {n_manual} alamat")

    print(f"[1/8] Membaca data mentah dari: {input_path}")
    df = load_raw(input_path, sheet)
    n_awal = len(df)
    print(f"      Jumlah baris awal: {n_awal}")

    print("[2/8] Mengambil kolom yang dipakai...")
    df = select_columns(df)

    print("[3/8] Membuang baris dengan field kunci kosong...")
    df, n_incomplete = drop_incomplete(df)
    print(f"      Baris dibuang (field kosong): {n_incomplete}")

    print("[4/8] Menghapus duplikat (berdasarkan NO REGISTRASI, ID unik asli)...")
    df, n_dup = deduplicate(df)
    print(f"      Baris duplikat dihapus: {n_dup}")
    print(f"      Sisa baris unik: {len(df)}")

    print(f"[5/8] Membuang hari dengan volume < {min_per_day} titik (kapasitas minimum per hari)...")
    df, n_hari_dibuang, n_baris_dibuang_hari = filter_low_volume_days(df, min_per_day=min_per_day)
    print(f"      Hari dibuang (volume < {min_per_day}): {n_hari_dibuang}")
    print(f"      Baris dibuang: {n_baris_dibuang_hari}")
    print(f"      Sisa baris: {len(df)}")

    print("[6/8] Parsing kecamatan / kota-kabupaten / provinsi / alamat jalan...")
    parsed = df["ALAMAT INSTALASI"].map(parse_admin_area)
    df["_kecamatan"] = parsed.map(lambda x: x[0])
    df["_kota_kab"] = parsed.map(lambda x: x[1])
    df["_provinsi"] = parsed.map(lambda x: x[2])
    df["_street"] = df["ALAMAT INSTALASI"].map(extract_street)

    print("[7/8] Menjalankan geocoding via Nominatim (mungkin lama, ada delay rate-limit)...")
    records = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"      ({i}/{total}) {str(row['NAMA INSTALASI'])[:30]}...", end="\r")
        hasil = geocode_row(row["_street"], row["Nama Kelurahan"], row["_kecamatan"],
                             row["_kota_kab"], row["_provinsi"], row["ALAMAT INSTALASI"])
        records.append(hasil)
    print()

    geo_df = pd.DataFrame(records)
    df = df.reset_index(drop=True)
    df_final = pd.concat([df, geo_df], axis=1)

    print("[8/8] Membuat NOMOR urut + kode pelanggan anonim (pengganti NAMA INSTALASI)...")
    df_final.insert(0, "NOMOR", range(1, len(df_final) + 1))
    df_final["KODE_PELANGGAN"] = generate_customer_code(len(df_final))

    # ---- Output 0 (PRIVAT -- JANGAN dibagikan/upload ke mana pun): pemetaan kode ke data asli ----
    mapping_cols = ["NOMOR", "KODE_PELANGGAN", "NO REGISTRASI", "TANGGAL PERMOHONAN",
                     "NAMA INSTALASI", "ALAMAT INSTALASI"]
    mapping_path = f"{output}_mapping_privat.csv"
    df_final[mapping_cols].to_csv(mapping_path, index=False)

    # ---- Output 1: data bersih siap pakai (NAMA INSTALASI diganti KODE_PELANGGAN) ----
    output_cols = ["NOMOR", "TANGGAL PERMOHONAN", "KODE_PELANGGAN", "ALAMAT INSTALASI",
                   "Nama Kelurahan", "Daya", "latitude", "longitude"]
    df_clean = df_final[output_cols]
    clean_path = f"{output}.csv"
    df_clean.to_csv(clean_path, index=False)

    # ---- Output 2: log audit (untuk pembuktian metodologi di Bab III/IV) ----
    log_cols = ["NOMOR", "KODE_PELANGGAN", "TANGGAL PERMOHONAN", "ALAMAT INSTALASI", "Nama Kelurahan", "Daya",
                "latitude", "longitude", "status_geocoding", "query_final", "sumber_koordinat"]
    log_path = f"{output}_log.csv"
    df_final[log_cols].to_csv(log_path, index=False)

    # ---- Ringkasan ----
    n_not_found = (df_final["status_geocoding"] == "not_found").sum()
    n_manual_used = (df_final["status_geocoding"] == "manual_override").sum()
    n_jalan = (df_final["status_geocoding"] == "found_alamat_jalan").sum()
    n_jalan_struct = (df_final["status_geocoding"] == "found_alamat_jalan_structured").sum()
    n_primary = (df_final["status_geocoding"] == "found_primary").sum()
    n_fallback = (df_final["status_geocoding"] == "found_fallback_kecamatan").sum()

    print("\n=== RINGKASAN PRAPROSES ===")
    print(f"Jumlah baris awal              : {n_awal}")
    print(f"Dibuang (field kosong)         : {n_incomplete}")
    print(f"Dibuang (duplikat)             : {n_dup}")
    print(f"Dibuang (hari volume < {min_per_day})    : {n_baris_dibuang_hari} baris ({n_hari_dibuang} hari)")
    print(f"Jumlah baris akhir             : {len(df_final)}")
    print(f"Kamus manual (dipakai)         : {n_manual_used}")
    print(f"Geocoding berhasil (jln, free) : {n_jalan}")
    print(f"Geocoding berhasil (jln, struct): {n_jalan_struct}")
    print(f"Geocoding berhasil (kelurahan) : {n_primary}")
    print(f"Geocoding berhasil (kecamatan) : {n_fallback}")
    print(f"Geocoding GAGAL total          : {n_not_found}")
    print(f"\nOutput data bersih (KODE, bukan nama) : {clean_path}")
    print(f"Output log audit (KODE, bukan nama)   : {log_path}")
    print(f"Output mapping PRIVAT (ada nama asli) : {mapping_path}")
    print("  ==> JANGAN upload/bagikan file mapping_privat ke mana pun, termasuk ke Claude.")

    if n_not_found > 0:
        print(f"\n[PERHATIAN] Ada {n_not_found} baris gagal di-geocode sama sekali "
              f"(lat/lon kosong). Baris ini harus ditinjau manual sebelum masuk ke tahap GA.")

    return df_clean, df_final



# =============================================================================
# UPLOAD FILE + JALANKAN OTOMATIS
# =============================================================================
from google.colab import files

print("=" * 70)
print("Upload file Excel data mentah kamu (misal: Data_Set_Medan_2025.xlsx)")
print("=" * 70)
uploaded = files.upload()
INPUT_FILENAME = list(uploaded.keys())[0]
print(f"\nFile diterima: {INPUT_FILENAME}")

print("\n" + "=" * 70)
print("Upload file kamus koordinat manual (OPSIONAL).")
print("Kalau tidak punya, klik 'Cancel upload' / tutup dialog ini saja.")
print("=" * 70)
MANUAL_OVERRIDES_PATH = None
try:
    uploaded_manual = files.upload()
    if uploaded_manual:
        MANUAL_OVERRIDES_PATH = list(uploaded_manual.keys())[0]
        print(f"File kamus manual diterima: {MANUAL_OVERRIDES_PATH}")
except Exception:
    pass
if not MANUAL_OVERRIDES_PATH:
    print("Tidak ada file kamus manual -- lanjut tanpa kamus (semua lewat geocoding otomatis).")

EMAIL = input("\nMasukkan email kamu (wajib, untuk User-Agent Nominatim): ").strip()
while not EMAIL or "@" not in EMAIL:
    EMAIL = input("Email tidak valid, coba lagi: ").strip()

OUTPUT_PREFIX = "data_bersih"

print("\nMemulai praproses...\n")
df_clean, df_final = run(input_path=INPUT_FILENAME, email=EMAIL, output=OUTPUT_PREFIX,
                          manual_overrides_path=MANUAL_OVERRIDES_PATH)

print("\nMengunduh file hasil...")
files.download(f"{OUTPUT_PREFIX}.csv")
files.download(f"{OUTPUT_PREFIX}_log.csv")
files.download(f"{OUTPUT_PREFIX}_mapping_privat.csv")
print("\nSelesai. Cek folder Downloads browser kamu untuk 3 file hasil.")
print("INGAT: jangan bagikan file _mapping_privat.csv ke mana pun.")