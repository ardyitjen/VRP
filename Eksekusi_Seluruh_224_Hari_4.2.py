# =============================================================================
# TA ARDY — GA-CVRP EKSEKUSI OTOMATIS SELURUH 224 HARI (untuk Subbab 4.2 dan 4.4)
# =============================================================================
# Adaptasi dari TA_Ardy_GA_CVRP_SatuCell_v3.py: fungsi inti (haversine, operator
# GA, run_single_ga, run_multiple) TIDAK diubah. Bagian input() interaktif untuk
# memilih satu tanggal diganti loop otomatis atas seluruh hari valid, supaya
# menghasilkan statistik agregat (Tabel 14) dan grafik konvergensi hari median
# (Gambar 4) tanpa perlu memilih tanggal satu per satu secara manual.
# =============================================================================

import math
import random
import statistics
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

R_RUNS = 50
SEEDS = range(1, R_RUNS + 1)

# =============================================================================
# 1. KONSTANTA TERKUNCI (sesuai Bab III, tidak berubah)
# =============================================================================
DEPOT = (3.546833, 98.645710)
N_POP = 100
PC = 0.8
PM = 0.1
MAX_GEN = 100
CAPACITY = 7

# =============================================================================
# 2. HAVERSINE & MATRIKS JARAK (identik dengan skrip asli)
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
# 3. ENCODING & KROMOSOM (identik dengan skrip asli)
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
# 4. INISIALISASI POPULASI (identik dengan skrip asli)
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
# 5. FITNESS (identik dengan skrip asli)
# =============================================================================
def fitness(distance):
    return float("inf") if distance <= 0 else 1.0 / distance

def evaluate_population(population, dist_matrix, capacity=CAPACITY):
    distances = [chromosome_distance(c, dist_matrix, capacity) for c in population]
    fitness_values = [fitness(d) for d in distances]
    return distances, fitness_values

# =============================================================================
# 6. OPERATOR GENETIKA (identik dengan skrip asli)
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
# 7. SATU KALI RUN GA (dipanggil berulang oleh multiple-run di bawah)
# =============================================================================
def run_single_ga(nodes, dist_matrix, seed, pop_size=N_POP, pc=PC, pm=PM,
                   max_gen=MAX_GEN, capacity=CAPACITY):
    n = len(nodes)
    rng = random.Random(seed)
    population = initialize_population(n, pop_size, rng)
    distances, fitness_vals = evaluate_population(population, dist_matrix, capacity)

    best_idx = distances.index(min(distances))
    best_chromosome = list(population[best_idx])
    best_distance = distances[best_idx]
    distance_history = [best_distance]

    for _ in range(max_gen):
        new_population = []
        while len(new_population) < pop_size:
            parent1 = roulette_wheel_selection(population, fitness_vals, rng)
            parent2 = roulette_wheel_selection(population, fitness_vals, rng)
            if rng.random() < pc:
                # crossover berhasil: satu kromosom anak
                child = order_crossover(parent1, parent2, rng)
                child = swap_mutation(child, pm, rng)
                new_population.append(child)
            else:
                # crossover gagal: PASANGAN induk diteruskan apa adanya
                # (masing-masing tetap kena mutasi sebagai operator terpisah)
                offspring1 = swap_mutation(parent1, pm, rng)
                offspring2 = swap_mutation(parent2, pm, rng)
                new_population.append(offspring1)
                new_population.append(offspring2)

        # jalur "gagal crossover" menghasilkan 2 individu per iterasi, jadi
        # populasi baru bisa melebihi pop_size tepat di iterasi terakhir;
        # potong ke ukuran tetap.
        new_population = new_population[:pop_size]

        population = new_population
        distances, fitness_vals = evaluate_population(population, dist_matrix, capacity)

        gen_best_idx = distances.index(min(distances))
        if distances[gen_best_idx] < best_distance:
            best_distance = distances[gen_best_idx]
            best_chromosome = list(population[gen_best_idx])

        distance_history.append(best_distance)

    return {"best_chromosome": best_chromosome, "best_distance": best_distance,
            "distance_history": distance_history}

def baseline_distance(nodes, dist_matrix, capacity=CAPACITY):
    return chromosome_distance(intake_chromosome(len(nodes)), dist_matrix, capacity)

def savings_percentage(d_manual, d_optimal):
    return 0.0 if d_manual == 0 else (d_manual - d_optimal) / d_manual * 100

# =============================================================================
# 8. MULTIPLE-RUN (R=50, seed 1..50) — mencari SOLUSI AKHIR (sesuai 2.9)
# =============================================================================
def run_multiple(nodes, dist_matrix, d_manual, seeds=SEEDS, **ga_params):
    per_run, all_results = [], []
    for seed in seeds:
        r = run_single_ga(nodes, dist_matrix, seed=seed, **ga_params)
        per_run.append(r["best_distance"])
        all_results.append(r)

    seeds_list = list(seeds)
    best_idx = per_run.index(min(per_run))
    n_beat = sum(1 for d in per_run if d < d_manual)

    return {
        "per_run": per_run,
        "seeds": seeds_list,
        "best": min(per_run),
        "mean": statistics.mean(per_run),
        "worst": max(per_run),
        "std": statistics.stdev(per_run) if len(per_run) > 1 else 0.0,
        "n_beat_baseline": n_beat,
        "n_runs": len(per_run),
        "best_seed": seeds_list[best_idx],
        "best_chromosome": all_results[best_idx]["best_chromosome"],
        "best_distance_history": all_results[best_idx]["distance_history"],
    }

# =============================================================================
# 9. UPLOAD & BACA DATA EXCEL (identik dengan skrip asli)
# =============================================================================
try:
    from google.colab import files
    print("Silakan upload file data_set_mdn_praproses.xlsx")
    uploaded = files.upload()
    excel_path = list(uploaded.keys())[0]
except ImportError:
    excel_path = "data_set_mdn_praproses.xlsx"  # fallback pengetesan lokal

df = pd.read_excel(excel_path)
df["TANGGAL PERMOHONAN"] = pd.to_datetime(df["TANGGAL PERMOHONAN"])

date_counts = df.groupby(df["TANGGAL PERMOHONAN"].dt.date).size().sort_index()
tanggal_list = list(date_counts.index)
print(f"\nDitemukan {len(tanggal_list)} hari valid. Memulai eksekusi otomatis seluruh hari...")

# =============================================================================
# 10. LOOP OTOMATIS SELURUH HARI VALID (pengganti input() interaktif)
# =============================================================================
median_volume = date_counts.median()
hari_median = min(tanggal_list, key=lambda d: (abs(date_counts[d] - median_volume), d))
print(f"Hari median dipilih untuk walkthrough detail: {hari_median} ({date_counts[hari_median]} titik)")

hasil_harian = []
t_mulai = time.time()

for i, tgl in enumerate(tanggal_list, start=1):
    sample = df[df["TANGGAL PERMOHONAN"].dt.date == tgl].reset_index(drop=True)
    nodes = list(zip(sample["latitude"], sample["longitude"]))
    n = len(nodes)

    dist_matrix = build_distance_matrix(DEPOT, nodes)
    d_manual = baseline_distance(nodes, dist_matrix)

    mr = run_multiple(nodes, dist_matrix, d_manual, seeds=SEEDS)
    d_optimal = mr["best"]
    fitness_optimal = fitness(d_optimal)
    penghematan_persen = savings_percentage(d_manual, d_optimal)

    hasil_harian.append({
        "tanggal": tgl,
        "jumlah_titik": n,
        "d_manual": d_manual,
        "d_optimal": d_optimal,
        "d_mean_run": mr["mean"],
        "d_worst_run": mr["worst"],
        "fitness_optimal": fitness_optimal,
        "penghematan_persen": penghematan_persen,
        "std_antar_run": mr["std"],
    })

    if tgl == hari_median:
        distance_history_median = mr["best_distance_history"]

    if i % 20 == 0 or i == len(tanggal_list):
        elapsed = time.time() - t_mulai
        print(f"[{i}/{len(tanggal_list)}] {tgl} selesai ({n} titik) — "
              f"waktu berjalan {elapsed/60:.1f} menit")
        pd.DataFrame(hasil_harian).to_csv("hasil_224_hari.csv", index=False)

df_hasil = pd.DataFrame(hasil_harian)
df_hasil.to_csv("hasil_224_hari.csv", index=False)
print(f"\nSeluruh {len(tanggal_list)} hari selesai dalam {(time.time()-t_mulai)/60:.1f} menit.")
print("Hasil per hari disimpan ke hasil_224_hari.csv")

# =============================================================================
# 11. STATISTIK AGREGAT (RATA-RATA/EKSTREM) 224 HARI
# =============================================================================
# Catatan: karena fitness = 1/D, hari dengan D terkecil otomatis punya fitness
# TERBESAR (bukan terkecil). Baris ekstrem karena itu disajikan per-hari yang
# sama untuk D dan fitness, bukan min/max independen per kolom.
print("\n=== Statistik Agregat Hasil Eksekusi Algoritma Genetika (224 Hari) ===")
print(f"{'Kategori':<28}{'Tanggal':<14}{'Jarak Tempuh Terbaik (D)':<28}{'Nilai Fitness Terbaik'}")
print(f"{'Rata-rata (224 hari)':<28}{'—':<14}{df_hasil['d_optimal'].mean():<28.2f}{df_hasil['fitness_optimal'].mean():.5f}")

baris_d_min = df_hasil.loc[df_hasil['d_optimal'].idxmin()]
print(f"{'D terpendek':<28}{str(baris_d_min['tanggal']):<14}{baris_d_min['d_optimal']:<28.2f}{baris_d_min['fitness_optimal']:.5f}")

baris_d_max = df_hasil.loc[df_hasil['d_optimal'].idxmax()]
print(f"{'D terpanjang (n=' + str(int(baris_d_max['jumlah_titik'])) + ')':<28}{str(baris_d_max['tanggal']):<14}{baris_d_max['d_optimal']:<28.2f}{baris_d_max['fitness_optimal']:.5f}")

print(f"{'Std Deviasi (224 hari)':<28}{'—':<14}{df_hasil['d_optimal'].std():<28.2f}{df_hasil['fitness_optimal'].std():.5f}")

# =============================================================================
# 11b. PERBANDINGAN D MANUAL vs D OPTIMAL vs PENGHEMATAN (224 HARI)
# =============================================================================
# Catatan: baris bukan "Rata-rata" masing-masing merujuk pada SATU hari yang
# sama untuk ketiga kolom (bukan min/max independen per kolom), supaya D
# Manual, D Optimal, dan Penghematan yang ditampilkan tetap konsisten sebagai
# satu kasus nyata, bukan kombinasi dari hari yang berbeda-beda.
print("\n=== Perbandingan Rute Manual dan Rute Optimasi (224 Hari) ===")
print(f"{'Kategori':<26}{'Tanggal':<14}{'D Manual (km)':<16}{'D Optimal (km)':<16}{'Penghematan (%)'}")

rata2 = (df_hasil['d_manual'].mean(), df_hasil['d_optimal'].mean(), df_hasil['penghematan_persen'].mean())
print(f"{'Rata-rata (224 hari)':<26}{'—':<14}{rata2[0]:<16.2f}{rata2[1]:<16.2f}{rata2[2]:.2f}")

baris_tertinggi = df_hasil.loc[df_hasil['penghematan_persen'].idxmax()]
print(f"{'Penghematan tertinggi':<26}{str(baris_tertinggi['tanggal']):<14}"
      f"{baris_tertinggi['d_manual']:<16.2f}{baris_tertinggi['d_optimal']:<16.2f}{baris_tertinggi['penghematan_persen']:.2f}")

baris_terendah = df_hasil.loc[df_hasil['penghematan_persen'].idxmin()]
print(f"{'Penghematan terendah':<26}{str(baris_terendah['tanggal']):<14}"
      f"{baris_terendah['d_manual']:<16.2f}{baris_terendah['d_optimal']:<16.2f}{baris_terendah['penghematan_persen']:.2f}")

baris_n87 = df_hasil[df_hasil['jumlah_titik'] == df_hasil['jumlah_titik'].max()].iloc[0]
print(f"{'Volume ekstrem (n=' + str(int(baris_n87['jumlah_titik'])) + ')':<26}{str(baris_n87['tanggal']):<14}"
      f"{baris_n87['d_manual']:<16.2f}{baris_n87['d_optimal']:<16.2f}{baris_n87['penghematan_persen']:.2f}")

# =============================================================================
# 11c. GAMBAR — HISTOGRAM DISTRIBUSI PERSENTASE PENGHEMATAN (224 HARI)
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(6.5, 4))
ax4.hist(df_hasil["penghematan_persen"], bins=range(0, 56, 5), color="#2A9D8F", edgecolor="white")
ax4.set_xlabel("Penghematan Jarak (%)")
ax4.set_ylabel("Jumlah Hari")
ax4.set_title("Distribusi Persentase Penghematan Jarak (224 Hari)")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("gambar4_distribusi_penghematan.png", dpi=200)
plt.show()

# =============================================================================
# 11d. GAMBAR — PENGHEMATAN (%) vs VOLUME KUNJUNGAN HARIAN
# =============================================================================
korelasi = df_hasil["penghematan_persen"].corr(df_hasil["jumlah_titik"])
fig5, ax5 = plt.subplots(figsize=(6.5, 4))
ax5.scatter(df_hasil["jumlah_titik"], df_hasil["penghematan_persen"], alpha=0.6, color="#457B9D", s=25)
ax5.set_xlabel("Jumlah Titik Kunjungan per Hari")
ax5.set_ylabel("Penghematan Jarak (%)")
ax5.set_title("Penghematan Jarak vs Volume Kunjungan Harian")
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("gambar5_penghematan_vs_volume.png", dpi=200)
plt.show()

print(f"\nKorelasi penghematan vs volume kunjungan harian: r = {korelasi:.2f}")
print("Gambar tersimpan: gambar4_distribusi_penghematan.png, gambar5_penghematan_vs_volume.png")

# =============================================================================
# 12. GAMBAR — KONVERGENSI JARAK DAN FITNESS PER GENERASI (HARI MEDIAN)
# =============================================================================
# titik-titik generasi tempat solusi terbaik membaik (best_so_far berubah)
perbaikan = []
prev = None
for gen, d in enumerate(distance_history_median):
    if prev is None or d < prev:
        perbaikan.append((gen, d, fitness(d)))
        prev = d

gens = [g for g, d, f in perbaikan]
ds = [d for g, d, f in perbaikan]
fs = [f for g, d, f in perbaikan]

fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.step(range(len(distance_history_median)), distance_history_median,
        where="post", color="#2E86AB", linewidth=2.2, zorder=2)

ax.scatter(gens[:-1], ds[:-1], color="#2E86AB", s=50, zorder=3,
           edgecolor="white", linewidth=1)
ax.scatter([gens[-1]], [ds[-1]], color="#E63946", s=80, zorder=4,
           edgecolor="white", linewidth=1.2)

# offset label (dx, dy) dalam points; kalibrasi khusus untuk 7 titik perbaikan
# (kasus hari median 19 Feb 2025). Kalau jumlah titik perbaikan berbeda,
# jatuh ke default (semua label kanan-atas titik) dan mungkin perlu
# dikalibrasi ulang manual kalau ada yang tumpang tindih.
default_offsets = [(35, 8)] * len(perbaikan)
label_offsets = default_offsets
if len(perbaikan) == 7:
    label_offsets = [(35, 8), (35, 6), (35, 12), (35, 16), (35, -14), (35, 6), (16, 14)]

for (g, d, f), (dx, dy) in zip(perbaikan, label_offsets):
    is_last = (g == gens[-1])
    ax.annotate(
        f"g{g} · {d:.2f} km", (g, d), textcoords="offset points",
        xytext=(dx, dy), ha="left" if dx > 0 else "right", va="center",
        fontsize=8.2, color="#9D0208" if is_last else "#1D3557",
        fontweight="bold" if is_last else "normal",
    )

ax.axvline(gens[-1], color="#E63946", linestyle="--", linewidth=1, alpha=0.4, zorder=1)
ax.axhline(ds[-1], color="#E63946", linestyle=":", linewidth=1, alpha=0.4, zorder=1)

ax.set_xlabel("Generasi")
ax.set_ylabel("Jarak Terbaik (km)", color="#2E86AB")
ax.tick_params(axis="y", colors="#2E86AB")
ax.set_title(f"Konvergensi Jarak dan Fitness per Generasi — Hari Median "
             f"({hari_median}, {date_counts[hari_median]} titik)")
ax.grid(alpha=0.25)
ax.spines["top"].set_visible(False)

ax2 = ax.secondary_yaxis("right", functions=(lambda d: 1.0 / d, lambda f: 1.0 / f))
ax2.set_ylabel("Nilai Fitness Terbaik", color="#8338EC")
ax2.tick_params(axis="y", colors="#8338EC")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.5f}"))

trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.annotate(f"{fs[-1]:.5f}", xy=(1.0, ds[-1]), xycoords=trans,
            xytext=(38, 0), textcoords="offset points", fontsize=8.5,
            color="#9D0208", fontweight="bold", va="center", ha="left",
            annotation_clip=False)

plt.tight_layout()
plt.savefig("gambar4_konvergensi_median.png", dpi=200)
plt.show()

print(f"\nHari median  : {hari_median} ({date_counts[hari_median]} titik)")
median_row = df_hasil[df_hasil['tanggal'] == hari_median].iloc[0]
print(f"D optimal    : {median_row['d_optimal']:.2f} km")
print(f"Fitness      : {median_row['fitness_optimal']:.5f}")
print(f"Generasi konvergen: {gens[-1]} (dari {MAX_GEN} generasi)")
print("Titik-titik perbaikan (generasi, D km, fitness):")
for g, d, f in perbaikan:
    print(f"  gen {g:>3} : D = {d:7.3f} km | fitness = {f:.5f}")