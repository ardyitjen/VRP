# =============================================================================
# TA ARDY — GA-CVRP PENGUJIAN VARIASI PARAMETER (untuk Subbab 4.3)
# =============================================================================
# Menguji efek ukuran populasi (N) dan maksimum generasi (MaxGen) terhadap
# hasil GA-CVRP, dijalankan di seluruh 224 hari valid (bukan instans terbatas),
# supaya bisa langsung dibandingkan dengan format Tabel 14 (subbab 4.2).
#
# Pendekatan one-factor-at-a-time: hanya N atau MaxGen yang diubah, parameter
# lain tetap di nilai baseline (N=100, Pc=0.8, Pm=0.1, MaxGen=100). Baseline
# TIDAK dijalankan ulang di sini karena sudah ada hasilnya dari subbab 4.2
# (hasil_224_hari.csv), dan sudah dicantumkan manual sebagai baris pembanding.
#
# 4 konfigurasi tambahan yang dijalankan:
#   N=50   (MaxGen=100)
#   N=200  (MaxGen=100)
#   MaxGen=50   (N=100)
#   MaxGen=200  (N=100)
#
# Fungsi inti (haversine, operator GA, run_single_ga, run_multiple) IDENTIK
# dengan skrip subbab 4.2 yang sudah dikoreksi logika crossover-nya.
# =============================================================================

import math
import random
import statistics
import time
import pandas as pd
import matplotlib.pyplot as plt

R_RUNS = 50
SEEDS = range(1, R_RUNS + 1)

# =============================================================================
# 1. KONSTANTA TERKUNCI (baseline, sesuai Bab III)
# =============================================================================
DEPOT = (3.546833, 98.645710)
N_POP_BASELINE = 100
PC = 0.8
PM = 0.1
MAX_GEN_BASELINE = 100
CAPACITY = 7

# Baseline (N=100, MaxGen=100) — hasil sudah ada dari subbab 4.2, dicantumkan
# di sini hanya untuk pembanding pada tabel akhir, TIDAK dijalankan ulang.
BASELINE_STATS = {
    "d_mean": 97.77, "d_min": 15.44, "d_max": 405.03, "d_std": 52.90,
    "fit_mean": 0.01322, "fit_std": 0.00763,
}

# Konfigurasi tambahan yang benar-benar dijalankan
KONFIGURASI = [
    {"label": "N=50",      "n_pop": 50,  "max_gen": 100, "csv": "hasil_N50.csv"},
    {"label": "N=200",     "n_pop": 200, "max_gen": 100, "csv": "hasil_N200.csv"},
    {"label": "MaxGen=50", "n_pop": 100, "max_gen": 50,  "csv": "hasil_MaxGen50.csv"},
    {"label": "MaxGen=200","n_pop": 100, "max_gen": 200, "csv": "hasil_MaxGen200.csv"},
]

# =============================================================================
# 2. HAVERSINE & MATRIKS JARAK (identik dengan skrip 4.2)
# =============================================================================
def haversine(coord1, coord2):
    R = 6371.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def build_distance_matrix(depot, nodes):
    points = [depot] + list(nodes)
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine(points[i], points[j])
    return matrix

# =============================================================================
# 3. ENCODING & KROMOSOM (identik dengan skrip 4.2)
# =============================================================================
def split_into_routes(chromosome, capacity=CAPACITY):
    return [chromosome[i:i + capacity] for i in range(0, len(chromosome), capacity)]

def route_distance(route, dist_matrix):
    if not route:
        return 0.0
    total = dist_matrix[0][route[0]]
    for a, b in zip(route[:-1], route[1:]):
        total += dist_matrix[a][b]
    total += dist_matrix[route[-1]][0]
    return total

def chromosome_distance(chromosome, dist_matrix, capacity=CAPACITY):
    routes = split_into_routes(chromosome, capacity)
    return sum(route_distance(r, dist_matrix) for r in routes)

# =============================================================================
# 4. INISIALISASI POPULASI (identik dengan skrip 4.2)
# =============================================================================
def intake_chromosome(n):
    return list(range(1, n + 1))

def random_chromosome(n, rng):
    chrom = list(range(1, n + 1))
    rng.shuffle(chrom)
    return chrom

def initialize_population(n, pop_size, rng):
    population = [intake_chromosome(n)]
    for _ in range(pop_size - 1):
        population.append(random_chromosome(n, rng))
    return population

# =============================================================================
# 5. FITNESS (identik dengan skrip 4.2)
# =============================================================================
def fitness(distance):
    return float("inf") if distance <= 0 else 1.0 / distance

def evaluate_population(population, dist_matrix, capacity=CAPACITY):
    distances = [chromosome_distance(c, dist_matrix, capacity) for c in population]
    fitness_values = [fitness(d) for d in distances]
    return distances, fitness_values

# =============================================================================
# 6. OPERATOR GENETIKA (identik dengan skrip 4.2)
# =============================================================================
def roulette_wheel_selection(population, fitness_values, rng):
    total_fitness = sum(fitness_values)
    pick = rng.uniform(0, total_fitness)
    running = 0.0
    for chrom, f in zip(population, fitness_values):
        running += f
        if running >= pick:
            return chrom
    return population[-1]

def order_crossover(parent1, parent2, rng):
    n = len(parent1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    segment = parent1[a:b + 1]
    child[a:b + 1] = segment
    segment_set = set(segment)
    fill_positions = [(b + 1 + i) % n for i in range(n)]
    fill_positions = [p for p in fill_positions if child[p] is None]
    source_order = [(b + 1 + i) % n for i in range(n)]
    genes_from_parent2 = [parent2[p] for p in source_order if parent2[p] not in segment_set]
    for pos, gene in zip(fill_positions, genes_from_parent2):
        child[pos] = gene
    return child

def swap_mutation(chromosome, pm, rng):
    chrom = list(chromosome)
    if rng.random() < pm:
        i, j = rng.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom

# =============================================================================
# 7. SATU KALI RUN GA (identik dengan skrip 4.2 — logika crossover terkoreksi)
# =============================================================================
def run_single_ga(nodes, dist_matrix, seed, pop_size, pc=PC, pm=PM,
                   max_gen=MAX_GEN_BASELINE, capacity=CAPACITY):
    n = len(nodes)
    rng = random.Random(seed)
    population = initialize_population(n, pop_size, rng)
    distances, fitness_vals = evaluate_population(population, dist_matrix, capacity)

    best_idx = distances.index(min(distances))
    best_chromosome = list(population[best_idx])
    best_distance = distances[best_idx]

    for _ in range(max_gen):
        new_population = []
        while len(new_population) < pop_size:
            parent1 = roulette_wheel_selection(population, fitness_vals, rng)
            parent2 = roulette_wheel_selection(population, fitness_vals, rng)
            if rng.random() < pc:
                child = order_crossover(parent1, parent2, rng)
                child = swap_mutation(child, pm, rng)
                new_population.append(child)
            else:
                offspring1 = swap_mutation(parent1, pm, rng)
                offspring2 = swap_mutation(parent2, pm, rng)
                new_population.append(offspring1)
                new_population.append(offspring2)

        new_population = new_population[:pop_size]
        population = new_population
        distances, fitness_vals = evaluate_population(population, dist_matrix, capacity)

        gen_best_idx = distances.index(min(distances))
        if distances[gen_best_idx] < best_distance:
            best_distance = distances[gen_best_idx]
            best_chromosome = list(population[gen_best_idx])

    return {"best_chromosome": best_chromosome, "best_distance": best_distance}

def baseline_distance(nodes, dist_matrix, capacity=CAPACITY):
    return chromosome_distance(intake_chromosome(len(nodes)), dist_matrix, capacity)

# =============================================================================
# 8. MULTIPLE-RUN (identik dengan skrip 4.2)
# =============================================================================
def run_multiple(nodes, dist_matrix, seeds, pop_size, max_gen):
    per_run = []
    for seed in seeds:
        r = run_single_ga(nodes, dist_matrix, seed=seed, pop_size=pop_size, max_gen=max_gen)
        per_run.append(r["best_distance"])
    return min(per_run)

# =============================================================================
# 9. UPLOAD & BACA DATA EXCEL (identik dengan skrip 4.2)
# =============================================================================
try:
    from google.colab import files
    print("Silakan upload file data_set_mdn_praproses.xlsx")
    uploaded = files.upload()
    excel_path = list(uploaded.keys())[0]
except ImportError:
    excel_path = "data_set_mdn_praproses.xlsx"

df = pd.read_excel(excel_path)
df["TANGGAL PERMOHONAN"] = pd.to_datetime(df["TANGGAL PERMOHONAN"])

date_counts = df.groupby(df["TANGGAL PERMOHONAN"].dt.date).size().sort_index()
tanggal_list = list(date_counts.index)
print(f"\nDitemukan {len(tanggal_list)} hari valid.")
print(f"Akan menjalankan {len(KONFIGURASI)} konfigurasi tambahan di seluruh hari ini.\n")

# =============================================================================
# 10. LOOP: UNTUK SETIAP KONFIGURASI, JALANKAN SELURUH 224 HARI
# =============================================================================
ringkasan_konfigurasi = []

for konfig in KONFIGURASI:
    label = konfig["label"]
    n_pop = konfig["n_pop"]
    max_gen = konfig["max_gen"]
    print(f"=== Menjalankan konfigurasi {label} (N={n_pop}, MaxGen={max_gen}) ===")

    hasil_harian = []
    t_mulai = time.time()

    for i, tgl in enumerate(tanggal_list, start=1):
        sample = df[df["TANGGAL PERMOHONAN"].dt.date == tgl].reset_index(drop=True)
        nodes = list(zip(sample["latitude"], sample["longitude"]))

        dist_matrix = build_distance_matrix(DEPOT, nodes)
        d_optimal = run_multiple(nodes, dist_matrix, SEEDS, pop_size=n_pop, max_gen=max_gen)
        fitness_optimal = fitness(d_optimal)

        hasil_harian.append({
            "tanggal": tgl,
            "jumlah_titik": len(nodes),
            "d_optimal": d_optimal,
            "fitness_optimal": fitness_optimal,
        })

        if i % 40 == 0 or i == len(tanggal_list):
            elapsed = time.time() - t_mulai
            print(f"  [{i}/{len(tanggal_list)}] {tgl} selesai — waktu berjalan {elapsed/60:.1f} menit")
            pd.DataFrame(hasil_harian).to_csv(konfig["csv"], index=False)

    df_hasil = pd.DataFrame(hasil_harian)
    df_hasil.to_csv(konfig["csv"], index=False)

    ringkasan_konfigurasi.append({
        "label": label,
        "d_mean": df_hasil["d_optimal"].mean(),
        "d_min": df_hasil["d_optimal"].min(),
        "d_max": df_hasil["d_optimal"].max(),
        "d_std": df_hasil["d_optimal"].std(),
        "fit_mean": df_hasil["fitness_optimal"].mean(),
        "fit_std": df_hasil["fitness_optimal"].std(),
    })
    print(f"  Konfigurasi {label} selesai dalam {(time.time()-t_mulai)/60:.1f} menit. "
          f"Disimpan ke {konfig['csv']}\n")

# =============================================================================
# 11. PERBANDINGAN UKURAN POPULASI (MaxGen=100 tetap)
# =============================================================================
# Catatan: kolom fitness HANYA menampilkan rata-rata dan std (agregat yang sah
# dihitung independen). Min/Maks fitness sengaja TIDAK ditampilkan berdampingan
# dengan Min/Maks D, karena fitness = 1/D sehingga D terkecil justru berarti
# fitness TERBESAR (bukan terkecil) -- menampilkan keduanya sebagai satu baris
# independen akan menyesatkan (D dan fitness pada baris yang sama bisa berasal
# dari hari yang berbeda).
def cetak_tabel(judul, baris_label_nilai):
    print(f"\n=== {judul} ===")
    print(f"{'Konfigurasi':<16}{'Rata-rata D':<14}{'Min D':<10}{'Maks D':<12}{'Std D':<10}"
          f"{'Rata-rata Fit':<15}{'Std Fit'}")
    for label, s in baris_label_nilai:
        print(f"{label:<16}{s['d_mean']:<14.2f}{s['d_min']:<10.2f}{s['d_max']:<12.2f}{s['d_std']:<10.2f}"
              f"{s['fit_mean']:<15.5f}{s['fit_std']:.5f}")

hasil_by_label = {r["label"]: r for r in ringkasan_konfigurasi}
baseline_row = {"label": "N=100 (baseline)", **BASELINE_STATS}

cetak_tabel(
    "Perbandingan Ukuran Populasi N (MaxGen=100)",
    [("N=50", hasil_by_label["N=50"]),
     ("N=100 (baseline)", BASELINE_STATS),
     ("N=200", hasil_by_label["N=200"])],
)

cetak_tabel(
    "Perbandingan Maksimum Generasi (N=100)",
    [("MaxGen=50", hasil_by_label["MaxGen=50"]),
     ("MaxGen=100 (baseline)", BASELINE_STATS),
     ("MaxGen=200", hasil_by_label["MaxGen=200"])],
)

print("\nSemua hasil per konfigurasi tersimpan di masing-masing file CSV "
      "(hasil_N50.csv, hasil_N200.csv, hasil_MaxGen50.csv, hasil_MaxGen200.csv).")