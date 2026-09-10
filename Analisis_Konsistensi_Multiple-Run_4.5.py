# =============================================================================
# TA ARDY — ANALISIS KONSISTENSI MULTIPLE-RUN (untuk Subbab 4.5)
# =============================================================================
# Skrip ini SELALU menjalankan pipeline GA-CVRP langsung dari
# data_set_mdn_praproses.xlsx (bukan dari file antara), supaya sumber datanya
# jelas dan konsisten dengan subbab 4.2. Fungsi inti identik dengan
# subbab_4_2_full224.py (logika crossover terkoreksi).
# =============================================================================

import math
import random
import statistics
import time
import pandas as pd
import matplotlib.pyplot as plt

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

R_RUNS = 50
SEEDS = range(1, R_RUNS + 1)

# =============================================================================
# 1. KONSTANTA TERKUNCI & FUNGSI INTI (identik dengan subbab_4_2_full224.py)
# =============================================================================
DEPOT = (3.546833, 98.645710)
N_POP = 100
PC = 0.8
PM = 0.1
MAX_GEN = 100
CAPACITY = 7

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

def fitness(distance):
    return float("inf") if distance <= 0 else 1.0 / distance

def evaluate_population(population, dist_matrix, capacity=CAPACITY):
    distances = [chromosome_distance(c, dist_matrix, capacity) for c in population]
    fitness_values = [fitness(d) for d in distances]
    return distances, fitness_values

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

def run_single_ga(nodes, dist_matrix, seed, pop_size=N_POP, pc=PC, pm=PM,
                   max_gen=MAX_GEN, capacity=CAPACITY):
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

def run_multiple(nodes, dist_matrix, seeds=SEEDS):
    per_run = []
    for seed in seeds:
        r = run_single_ga(nodes, dist_matrix, seed=seed)
        per_run.append(r["best_distance"])
    return {
        "best": min(per_run),
        "mean": statistics.mean(per_run),
        "worst": max(per_run),
        "std": statistics.stdev(per_run) if len(per_run) > 1 else 0.0,
    }

# =============================================================================
# 2. BACA DATASET PRAPROSES & JALANKAN SELURUH 224 HARI
# =============================================================================
if IN_COLAB:
    print("Silakan upload file data_set_mdn_praproses.xlsx")
    uploaded = files.upload()
    excel_path = list(uploaded.keys())[0]
else:
    excel_path = "data_set_mdn_praproses.xlsx"

raw = pd.read_excel(excel_path)
raw["TANGGAL PERMOHONAN"] = pd.to_datetime(raw["TANGGAL PERMOHONAN"])
date_counts = raw.groupby(raw["TANGGAL PERMOHONAN"].dt.date).size().sort_index()
tanggal_list = list(date_counts.index)
print(f"Ditemukan {len(tanggal_list)} hari valid. Menjalankan GA untuk seluruh hari...\n")

hasil_harian = []
t_mulai = time.time()
for i, tgl in enumerate(tanggal_list, start=1):
    sample = raw[raw["TANGGAL PERMOHONAN"].dt.date == tgl].reset_index(drop=True)
    nodes = list(zip(sample["latitude"], sample["longitude"]))
    n = len(nodes)
    dist_matrix = build_distance_matrix(DEPOT, nodes)
    d_manual = baseline_distance(nodes, dist_matrix)
    mr = run_multiple(nodes, dist_matrix, seeds=SEEDS)
    hasil_harian.append({
        "tanggal": tgl, "jumlah_titik": n, "d_manual": d_manual,
        "d_optimal": mr["best"], "d_mean_run": mr["mean"], "d_worst_run": mr["worst"],
        "std_antar_run": mr["std"],
    })
    if i % 40 == 0 or i == len(tanggal_list):
        print(f"  [{i}/{len(tanggal_list)}] selesai — {(time.time()-t_mulai)/60:.1f} menit")

df = pd.DataFrame(hasil_harian)
df.to_csv("hasil_konsistensi_224_hari.csv", index=False)
print(f"\nSeluruh {len(tanggal_list)} hari selesai dalam {(time.time()-t_mulai)/60:.1f} menit.")
print("Hasil disimpan ke hasil_konsistensi_224_hari.csv")

# =============================================================================
# 3. HITUNG KOEFISIEN VARIASI (CV = std_antar_run / d_optimal x 100%)
# =============================================================================
df["cv_persen"] = df["std_antar_run"] / df["d_optimal"] * 100

# =============================================================================
# 4. STATISTIK KONSISTENSI MULTIPLE-RUN (224 HARI)
# =============================================================================
# Baris ekstrem diurutkan berdasarkan CV (metrik utama), dengan std mentah pada
# hari yang sama ditampilkan berdampingan, supaya kedua angka tetap konsisten
# berasal dari satu hari (bukan independen per kolom).
print("\n=== Statistik Konsistensi Multiple-Run (224 Hari) ===")
print(f"{'Kategori':<24}{'Tanggal':<14}{'Std Antar-Run (km)':<22}{'Koefisien Variasi (%)'}")
print(f"{'Rata-rata (224 hari)':<24}{'—':<14}{df['std_antar_run'].mean():<22.2f}{df['cv_persen'].mean():.2f}")

baris_cv_min = df.loc[df['cv_persen'].idxmin()]
print(f"{'CV terendah':<24}{str(baris_cv_min['tanggal']):<14}{baris_cv_min['std_antar_run']:<22.2f}{baris_cv_min['cv_persen']:.2f}")

baris_cv_max = df.loc[df['cv_persen'].idxmax()]
print(f"{'CV tertinggi':<24}{str(baris_cv_max['tanggal']):<14}{baris_cv_max['std_antar_run']:<22.2f}{baris_cv_max['cv_persen']:.2f}")

print(f"{'Std Deviasi (224 hari)':<24}{'—':<14}{df['std_antar_run'].std():<22.2f}{df['cv_persen'].std():.2f}")

jumlah_std_nol = (df["std_antar_run"] == 0).sum()
print(f"\nJumlah hari dengan std_antar_run = 0 (seluruh 50 run identik): "
      f"{jumlah_std_nol} dari {len(df)} hari ({jumlah_std_nol/len(df)*100:.1f}%)")

korelasi_std_volume = df["std_antar_run"].corr(df["jumlah_titik"])
korelasi_cv_volume = df["cv_persen"].corr(df["jumlah_titik"])
print(f"Korelasi std mentah vs volume harian : r = {korelasi_std_volume:.2f}")
print(f"Korelasi CV vs volume harian         : r = {korelasi_cv_volume:.2f}")

# =============================================================================
# 5. HISTOGRAM DISTRIBUSI CV (%)
# =============================================================================
fig7, ax7 = plt.subplots(figsize=(6.5, 4))
ax7.hist(df["cv_persen"], bins=20, color="#F4A261", edgecolor="white")
ax7.set_xlabel("Koefisien Variasi (%)")
ax7.set_ylabel("Jumlah Hari")
ax7.set_title("Distribusi Koefisien Variasi Antar-Run (224 Hari)")
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("gambar7_distribusi_cv.png", dpi=200)
plt.show()

# =============================================================================
# 6. CV (%) vs VOLUME KUNJUNGAN HARIAN
# =============================================================================
fig8, ax8 = plt.subplots(figsize=(6.5, 4))
ax8.scatter(df["jumlah_titik"], df["cv_persen"], alpha=0.6, color="#8338EC", s=25)
ax8.set_xlabel("Jumlah Titik Kunjungan per Hari")
ax8.set_ylabel("Koefisien Variasi (%)")
ax8.set_title("Koefisien Variasi vs Volume Kunjungan Harian")
ax8.spines["top"].set_visible(False)
ax8.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("gambar8_cv_vs_volume.png", dpi=200)
plt.show()

print("\nGambar tersimpan: gambar7_distribusi_cv.png, gambar8_cv_vs_volume.png")

# =============================================================================
# 7. PERBANDINGAN 1x RUN vs MULTI-RUN (R=50)
# =============================================================================
# "1x run" diwakili oleh RATA-RATA dari 50 run (perkiraan hasil tipikal kalau
# GA hanya dijalankan sekali), dibandingkan terhadap BEST dari 50 run yang
# dipakai sebagai solusi akhir. Selisih keduanya menunjukkan keuntungan nyata
# dari strategi multiple-run itu sendiri.
df["peningkatan_multirun_persen"] = (df["d_mean_run"] - df["d_optimal"]) / df["d_mean_run"] * 100

print("\n=== Perbandingan Hasil 1x Run vs Multi-Run (R=50, 224 Hari) ===")
print(f"{'Kategori':<28}{'Tanggal':<14}{'Rata-rata 1x Run (km)':<24}{'Terbaik 50 Run (km)':<22}{'Peningkatan (%)'}")
print(f"{'Rata-rata (224 hari)':<28}{'—':<14}{df['d_mean_run'].mean():<24.2f}{df['d_optimal'].mean():<22.2f}{df['peningkatan_multirun_persen'].mean():.2f}")

baris_pi_min = df.loc[df['peningkatan_multirun_persen'].idxmin()]
print(f"{'Peningkatan terendah':<28}{str(baris_pi_min['tanggal']):<14}{baris_pi_min['d_mean_run']:<24.2f}{baris_pi_min['d_optimal']:<22.2f}{baris_pi_min['peningkatan_multirun_persen']:.2f}")

baris_pi_max = df.loc[df['peningkatan_multirun_persen'].idxmax()]
print(f"{'Peningkatan tertinggi':<28}{str(baris_pi_max['tanggal']):<14}{baris_pi_max['d_mean_run']:<24.2f}{baris_pi_max['d_optimal']:<22.2f}{baris_pi_max['peningkatan_multirun_persen']:.2f}")

print(f"{'Std Deviasi (224 hari)':<28}{'—':<14}{df['d_mean_run'].std():<24.2f}{df['d_optimal'].std():<22.2f}{df['peningkatan_multirun_persen'].std():.2f}")

fig9, ax9 = plt.subplots(figsize=(6.5, 4))
ax9.hist(df["peningkatan_multirun_persen"], bins=20, color="#6A994E", edgecolor="white")
ax9.set_xlabel("Peningkatan dari Multiple-Run (%)")
ax9.set_ylabel("Jumlah Hari")
ax9.set_title("Selisih Rata-rata 1x Run terhadap Terbaik 50 Run (224 Hari)")
ax9.spines["top"].set_visible(False)
ax9.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("gambar9_peningkatan_multirun.png", dpi=200)
plt.show()
print("Gambar tersimpan: gambar9_peningkatan_multirun.png")