# ==========================================================
# GA-CVRP OPTIMASI RUTE PEMERIKSAAN SLO
# + PENGUJIAN MANDIRI VARIASI Pc DAN Pm (bekal persiapan sidang,
#   BUKAN bagian dari metodologi resmi Bab IV, yang resmi cuma
#   menguji ukuran populasi dan maksimum generasi di subbab 4.3)
# ==========================================================

import math
import random
import statistics
import pandas as pd
import matplotlib.pyplot as plt

R_RUNS = 50
SEEDS = range(1, R_RUNS + 1)

# ==========================================================
# 1. VARIABEL
# ==========================================================
DEPOT = (3.546833, 98.645710)
N_POP = 100
PC = 0.8
PM = 0.1
MAX_GEN = 100
CAPACITY = 7

# --- variasi Pc dan Pm untuk pengujian mandiri, plus minus 0,1 dari baseline ---
PC_DELTA = 0.1
PM_DELTA = 0.1
PC_VALUES = [round(PC - PC_DELTA, 2), PC, round(PC + PC_DELTA, 2)]          # [0.7, 0.8, 0.9]
PM_VALUES = [round(max(0.0, PM - PM_DELTA), 2), PM, round(PM + PM_DELTA, 2)]  # [0.0, 0.1, 0.2]

# ==========================================================
# 2. HAVERSINE & MATRIKS JARAK
#==========================================================
def haversine(coord1, coord2):
    R = 6371.0  # Radius Bumi (Km)
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    lambda_1 = math.radians(lon1)
    lambda_2 = math.radians(lon2)

    delta_phi = phi_1 - phi_2
    delta_lambda = lambda_1 - lambda_2

    a = (math.sin(delta_phi / 2) ** 2
         + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def build_distance_matrix(depot, nodes):
    points = [depot] + list(nodes)
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine(points[i], points[j])
    return matrix

# ==========================================================
# 3. ENCODING & KROMOSOM
#==========================================================
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

# ==========================================================
# 4. INISIALISASI POPULASI
#==========================================================
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

# ==========================================================
# 5. FITNESS (Persamaan 5: fi = 1/Di)
#==========================================================
def fitness(D_i):
    """Persamaan (5): fi = 1 / Di"""
    return float("inf") if D_i <= 0 else 1.0 / D_i

def evaluate_population(population, dist_matrix, capacity=CAPACITY):
    distances = [chromosome_distance(c, dist_matrix, capacity) for c in population]
    fitness_values = [fitness(D_i) for D_i in distances]
    return distances, fitness_values

# ==========================================================
# 6. OPERATOR GENETIKA
#==========================================================
def roulette_wheel_selection(population, fitness_values, rng):
    """Persamaan (6): Prob_i = F_i / sum_{j=1}^{PopSize} F_j"""
    sum_Fj = sum(fitness_values)
    Prob = [F_i / sum_Fj for F_i in fitness_values]
    pick = rng.random()
    cumulative = 0.0
    for chrom, Prob_i in zip(population, Prob):
        cumulative += Prob_i
        if cumulative >= pick:
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

# ==========================================================
# 7. SATU KALI RUN GA (dipanggil berulang oleh multiple-run)
#==========================================================
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

        distance_history.append(best_distance)

    return {"best_chromosome": best_chromosome, "best_distance": best_distance,
            "distance_history": distance_history}

def baseline_distance(nodes, dist_matrix, capacity=CAPACITY):
    return chromosome_distance(intake_chromosome(len(nodes)), dist_matrix, capacity)

def savings_percentage(D_manual, D_optimasi):
    """Persamaan (7): Penghematan % = (Dmanual - Doptimasi) / Dmanual x 100%"""
    return 0.0 if D_manual == 0 else (D_manual - D_optimasi) / D_manual * 100

# ==========================================================
# 8. MULTIPLE-RUN (R=50, seed 1..50) — mencari SOLUSI AKHIR (sesuai 2.9)
#==========================================================
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

def summarize_run(per_run):
    """Ringkasan satu konfigurasi dari daftar jarak 50 run, format sama
    seperti kolom pada Tabel 14/15 (Rata-rata D, D Maksimum, Std D,
    Rata-rata Fitness). Untuk pengujian mandiri satu hari ini, basisnya
    adalah 50 run pada hari yang sama, bukan rata-rata 224 hari seperti
    di Bab IV resmi."""
    fitness_vals = [fitness(d) for d in per_run]
    return {
        "Rata-rata D (km)": statistics.mean(per_run),
        "D Maksimum (km)": max(per_run),
        "Std D (km)": statistics.stdev(per_run) if len(per_run) > 1 else 0.0,
        "Rata-rata Fitness": statistics.mean(fitness_vals),
    }

# ==========================================================
# 8b. UTILITAS TABEL (dipindah lebih awal supaya bisa dipakai
#     oleh pengujian variasi Pc/Pm di bagian 10b, sebelum dipakai
#     lagi untuk tabel urutan kunjungan di bagian 17)
#==========================================================
def render_table_image(df_table, title, filename):
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(df_table) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title(title, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=140, bbox_inches="tight")
    plt.show()

# ==========================================================
# 9. UPLOAD & BACA DATA EXCEL
#==========================================================
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

print(f"\nDitemukan {len(tanggal_list)} tanggal pada data.")
print("Daftar tanggal tersedia (nomor. tanggal - jumlah titik):\n")
for i, tgl in enumerate(tanggal_list, start=1):
    print(f"{i:>3}. {tgl}  ({date_counts[tgl]} titik)")

while True:
    pilihan = input("\nMasukkan NOMOR tanggal yang ingin diproses: ").strip()
    if pilihan.isdigit() and 1 <= int(pilihan) <= len(tanggal_list):
        TANGGAL_HITUNG = str(tanggal_list[int(pilihan) - 1])
        break
    print("Nomor tidak valid, coba lagi.")

print(f"\nTanggal dipilih: {TANGGAL_HITUNG}")

sample = df[df["TANGGAL PERMOHONAN"].dt.date.astype(str) == TANGGAL_HITUNG].reset_index(drop=True)
nodes = list(zip(sample["latitude"], sample["longitude"]))
kode_pelanggan = sample["KODE_PELANGGAN"].tolist()
kelurahan = sample["Nama Kelurahan"].tolist()
n = len(nodes)

print(f"Jumlah titik       : {n}")
print(f"Jumlah petugas     : {math.ceil(n / CAPACITY)}")

# ==========================================================
# 10. MATRIKS JARAK, BASELINE, & MULTIPLE-RUN (CARI SOLUSI AKHIR)
#     Konfigurasi baseline ini (Pc=0.8, Pm=0.1) dipakai juga sebagai
#     baris baseline pada kedua tabel perbandingan di bagian 10b.
# ==========================================================
dist_matrix = build_distance_matrix(DEPOT, nodes)
d_manual = baseline_distance(nodes, dist_matrix)
intake_chrom = intake_chromosome(n)

print(f"\nMenjalankan multiple-run (R={R_RUNS}, seed 1-{R_RUNS}) untuk {TANGGAL_HITUNG}...")
mr = run_multiple(nodes, dist_matrix, d_manual, seeds=SEEDS)

best_chromosome = mr["best_chromosome"]
d_optimal = mr["best"]
best_seed = mr["best_seed"]
distance_history = mr["best_distance_history"]

penghematan_km = d_manual - d_optimal
penghematan_persen = savings_percentage(d_manual, d_optimal)

print(f"\n--- HASIL MULTIPLE-RUN (R={R_RUNS}) ---")
print(f"Best  : {mr['best']:.1f} km  <- solusi akhir, dari seed={best_seed}")
print(f"Mean  : {mr['mean']:.1f} km")
print(f"Worst : {mr['worst']:.1f} km")
print(f"Std   : {mr['std']:.2f}")
print(f"Menang vs baseline manual ({d_manual:.1f} km): {mr['n_beat_baseline']}/{mr['n_runs']} run")

print(f"\n--- SOLUSI AKHIR (rute terpendek dari {R_RUNS} pengulangan) ---")
print(f"Total jarak manual  : {d_manual:.1f} km")
print(f"Total jarak optimal : {d_optimal:.1f} km  (seed={best_seed})")
print(f"Penghematan         : {penghematan_km:.1f} km ({penghematan_persen:.2f}%)")
print(f"Kromosom terbaik    : {best_chromosome}")

# ==========================================================
# 10b. PENGUJIAN MANDIRI VARIASI Pc DAN Pm (bekal persiapan sidang)
#      Pola sama seperti Tabel 14/15, satu parameter diubah sementara
#      parameter lain tetap di baseline, R=50, satu tanggal yang sama
#      dengan pengujian utama di atas. Baseline (Pc=0.8, Pm=0.1) dipakai
#      ulang dari hasil "mr" di bagian 10, tidak dijalankan dua kali.
# ==========================================================
print(f"\n{'='*60}")
print("PENGUJIAN MANDIRI VARIASI Pc DAN Pm (bukan bagian resmi Bab IV)")
print(f"{'='*60}")

# --- variasi Pc, Pm tetap di baseline 0.1 ---
pc_results = {}
for pc_val in PC_VALUES:
    if pc_val == PC:
        pc_results[pc_val] = mr  # baseline, pakai ulang hasil bagian 10
        print(f"Pc={pc_val} (baseline, Pm={PM}) -> pakai ulang hasil di bagian 10")
        continue
    print(f"Menjalankan Pc={pc_val}, Pm={PM} (baseline) ...")
    pc_results[pc_val] = run_multiple(nodes, dist_matrix, d_manual, seeds=SEEDS, pc=pc_val, pm=PM)

# --- variasi Pm, Pc tetap di baseline 0.8 ---
pm_results = {}
for pm_val in PM_VALUES:
    if pm_val == PM:
        pm_results[pm_val] = mr  # baseline, pakai ulang hasil bagian 10
        print(f"Pm={pm_val} (baseline, Pc={PC}) -> pakai ulang hasil di bagian 10")
        continue
    print(f"Menjalankan Pc={PC} (baseline), Pm={pm_val} ...")
    pm_results[pm_val] = run_multiple(nodes, dist_matrix, d_manual, seeds=SEEDS, pc=PC, pm=pm_val)

# --- tabel ringkasan, format sama seperti Tabel 14/15 ---
tabel_pc = pd.DataFrame([
    {"Konfigurasi": f"Pc={pc_val}" + (" (baseline)" if pc_val == PC else ""), **summarize_run(pc_results[pc_val]["per_run"])}
    for pc_val in PC_VALUES
])
tabel_pm = pd.DataFrame([
    {"Konfigurasi": f"Pm={pm_val}" + (" (baseline)" if pm_val == PM else ""), **summarize_run(pm_results[pm_val]["per_run"])}
    for pm_val in PM_VALUES
])

for col in ["Rata-rata D (km)", "D Maksimum (km)", "Std D (km)"]:
    tabel_pc[col] = tabel_pc[col].round(2)
    tabel_pm[col] = tabel_pm[col].round(2)
tabel_pc["Rata-rata Fitness"] = tabel_pc["Rata-rata Fitness"].round(5)
tabel_pm["Rata-rata Fitness"] = tabel_pm["Rata-rata Fitness"].round(5)

print(f"\nTabel Perbandingan Variasi Pc (Pm={PM} baseline) - {TANGGAL_HITUNG}")
print(tabel_pc.to_string(index=False))
print(f"\nTabel Perbandingan Variasi Pm (Pc={PC} baseline) - {TANGGAL_HITUNG}")
print(tabel_pm.to_string(index=False))

render_table_image(tabel_pc, f"Perbandingan Variasi Pc (Pm={PM} baseline) - {TANGGAL_HITUNG}",
                    "9_tabel_variasi_pc.png")
render_table_image(tabel_pm, f"Perbandingan Variasi Pm (Pc={PC} baseline) - {TANGGAL_HITUNG}",
                    "10_tabel_variasi_pm.png")

# --- grafik Pc1 vs Pc2 vs Pc3, jarak tiap run (seed 1-50) ---
fig_pc, ax_pc = plt.subplots(figsize=(9, 4.5))
pc_colors = ["#457B9D", "#2A9D8F", "#F4A261"]
for color, pc_val in zip(pc_colors, PC_VALUES):
    label = f"Pc={pc_val}" + (" (baseline)" if pc_val == PC else "")
    ax_pc.plot(range(1, R_RUNS + 1), pc_results[pc_val]["per_run"], marker="o", markersize=3,
               linewidth=1.2, color=color, label=label)
ax_pc.set_xlabel("Run ke- (nomor seed)")
ax_pc.set_ylabel("Jarak terbaik (km)")
ax_pc.set_title(f"Perbandingan Variasi Pc per Run - {TANGGAL_HITUNG} (Pm={PM} baseline)")
ax_pc.legend()
ax_pc.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("11_perbandingan_pc.png", dpi=150)
plt.show()

# --- grafik Pm1 vs Pm2 vs Pm3, jarak tiap run (seed 1-50) ---
fig_pm, ax_pm = plt.subplots(figsize=(9, 4.5))
pm_colors = ["#E63946", "#2A9D8F", "#8338EC"]
for color, pm_val in zip(pm_colors, PM_VALUES):
    label = f"Pm={pm_val}" + (" (baseline)" if pm_val == PM else "")
    ax_pm.plot(range(1, R_RUNS + 1), pm_results[pm_val]["per_run"], marker="o", markersize=3,
               linewidth=1.2, color=color, label=label)
ax_pm.set_xlabel("Run ke- (nomor seed)")
ax_pm.set_ylabel("Jarak terbaik (km)")
ax_pm.set_title(f"Perbandingan Variasi Pm per Run - {TANGGAL_HITUNG} (Pc={PC} baseline)")
ax_pm.legend()
ax_pm.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("12_perbandingan_pm.png", dpi=150)
plt.show()

print("\nCatatan: pengujian Pc/Pm di atas memakai satu tanggal dan R=50 run,")
print("untuk bekal persiapan sidang pribadi, bukan bagian metodologi resmi Bab IV.")

# ==========================================================
# 11. VISUALISASI PENDUKUNG — HASIL 50 RUN INDEPENDEN (BASELINE)
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(9, 4.5))
bar_colors = ["#F4A261" if s == best_seed else "#2A9D8F" for s in mr["seeds"]]
ax1.bar(range(1, mr["n_runs"] + 1), mr["per_run"], color=bar_colors)
ax1.axhline(d_manual, color="#E63946", linestyle="--", label=f"Baseline manual ({d_manual:.1f} km)")
ax1.axhline(mr["mean"], color="#264653", linestyle=":", label=f"Rata-rata GA ({mr['mean']:.1f} km)")
ax1.set_xlabel("Run ke- (nomor seed)")
ax1.set_ylabel("Jarak terbaik (km)")
ax1.set_title(f"Hasil {mr['n_runs']} Run Independen - {TANGGAL_HITUNG}\n"
              f"(oranye = run terbaik, seed={best_seed}, dipakai sebagai solusi akhir)")
ax1.legend()
ax1.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("1_multiple_run.png", dpi=150)
plt.show()

# ==========================================================
# 12. GAMBAR 9 — PETA RUTE BERLABEL (MANUAL vs SOLUSI AKHIR)
# ==========================================================
def route_points_labeled(chromosome, kode_list, capacity=CAPACITY):
    routes = split_into_routes(chromosome, capacity)
    out = []
    for r in routes:
        lats = [DEPOT[0]] + [nodes[i - 1][0] for i in r] + [DEPOT[0]]
        lons = [DEPOT[1]] + [nodes[i - 1][1] for i in r] + [DEPOT[1]]
        labels = ["Depot"] + [f"{urutan}. {kode_list[i - 1]}" for urutan, i in enumerate(r, start=1)] + ["Depot"]
        out.append((lats, lons, labels))
    return out

colors = ["#E63946", "#2A9D8F", "#457B9D", "#F4A261", "#8338EC", "#FFB703", "#6A994E"]
manual_r = route_points_labeled(intake_chrom, kode_pelanggan)
optimal_r = route_points_labeled(best_chromosome, kode_pelanggan)

fig2, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, routes, title in zip(axes, [manual_r, optimal_r],
                              ["Rute Manual (Intake)", f"Rute Solusi Akhir (seed={best_seed})"]):
    for idx, (lats, lons, labels) in enumerate(routes):
        ax.plot(lons, lats, "-o", color=colors[idx % len(colors)], markersize=5, label=f"Petugas {idx + 1}")
        for lat, lon, label in zip(lats, lons, labels):
            if label != "Depot":
                ax.annotate(label, (lon, lat), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.scatter([DEPOT[1]], [DEPOT[0]], color="black", marker="s", s=90, zorder=5, label="Depot")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("2_peta_rute.png", dpi=150)
plt.show()

# ==========================================================
# 13a. UTILITAS LABEL TITIK LOMPATAN (STEP CHANGE) UNTUK GRAFIK KONVERGENSI
# ==========================================================
def get_step_change_points(values):
    points = [0]
    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            points.append(i)
    if points[-1] != len(values) - 1:
        points.append(len(values) - 1)
    return points

def filter_close_points(points, min_gap):
    if len(points) <= 2:
        return points
    filtered = [points[0]]
    for idx in points[1:-1]:
        if idx - filtered[-1] >= min_gap:
            filtered.append(idx)
    if points[-1] - filtered[-1] < min_gap:
        filtered[-1] = points[-1]
    else:
        filtered.append(points[-1])
    return filtered

# ==========================================================
# 13. GAMBAR 6 — KONVERGENSI FITNESS TERBAIK (dari run pemenang, seed=best_seed)
# ==========================================================
fitness_history = [fitness(d) for d in distance_history]
fig3, ax3 = plt.subplots(figsize=(8, 4.5))
ax3.plot(range(len(fitness_history)), fitness_history, color="#8338EC", linewidth=2)

min_gap_gen = max(3, len(fitness_history) // 25)
fitness_step_points = filter_close_points(get_step_change_points(fitness_history), min_gap_gen)
for k, idx in enumerate(fitness_step_points):
    offset_y = 8 if k % 2 == 0 else -14
    ax3.annotate(f"{fitness_history[idx]:.4f}",
                 (idx, fitness_history[idx]),
                 xytext=(0, offset_y), textcoords="offset points",
                 fontsize=7.5, ha="center", color="#5b1fb0")

ax3.set_xlabel("Generasi")
ax3.set_ylabel("Nilai fitness terbaik")
ax3.set_title(f"Nilai Fitness per Generasi - {TANGGAL_HITUNG} (seed pemenang={best_seed})")
ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("3_fitness_per_generasi.png", dpi=150)
plt.show()

# ==========================================================
# 14. VISUALISASI PENDUKUNG — JARAK TERBAIK PER GENERASI (dari run pemenang)
# ==========================================================
gen_terbaik = distance_history.index(min(distance_history))
fig4, ax4 = plt.subplots(figsize=(8, 4.5))
ax4.plot(range(len(distance_history)), distance_history, color="#2E86AB", linewidth=2)

min_gap_gen_dist = max(3, len(distance_history) // 25)
distance_step_points = filter_close_points(get_step_change_points(distance_history), min_gap_gen_dist)
for k, idx in enumerate(distance_step_points):
    if idx == gen_terbaik:
        continue
    offset_y = 10 if k % 2 == 0 else -16
    ax4.annotate(f"{distance_history[idx]:.1f}",
                 (idx, distance_history[idx]),
                 xytext=(0, offset_y), textcoords="offset points",
                 fontsize=7.5, ha="center", color="#1c5f80")

ax4.scatter([gen_terbaik], [distance_history[gen_terbaik]], color="red", zorder=5, s=70)
ax4.annotate(f"Gen ke-{gen_terbaik}: {distance_history[gen_terbaik]:.1f} km",
             (gen_terbaik, distance_history[gen_terbaik]),
             xytext=(15, 15), textcoords="offset points",
             fontsize=9, fontweight="bold", color="red",
             arrowprops=dict(arrowstyle="->", color="red"))
ax4.set_xlabel("Generasi")
ax4.set_ylabel("Jarak terbaik (km)")
ax4.set_title(f"Konvergensi Jarak per Generasi - {TANGGAL_HITUNG} (seed pemenang={best_seed})")
ax4.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("4_jarak_per_generasi.png", dpi=150)
plt.show()

print(f"\nDalam run pemenang (seed={best_seed}), jarak terbaik ({d_optimal:.1f} km) "
      f"pertama kali dicapai pada generasi ke-{gen_terbaik} dari {MAX_GEN} generasi.")

# ==========================================================
# 15. VISUALISASI PENDUKUNG — TOTAL PENGHEMATAN SEBELUM vs SESUDAH VRP
# ==========================================================
fig5, ax5 = plt.subplots(figsize=(6, 4.5))
bars = ax5.bar(["Sebelum\n(rute berurut)", "Sesudah\n(VRP-GA)"],
                [d_manual, d_optimal], color=["#E63946", "#2A9D8F"])
for bar, val in zip(bars, [d_manual, d_optimal]):
    ax5.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f} km", ha="center", fontweight="bold")
ax5.set_ylabel("Total jarak tempuh (km)")
ax5.set_title(f"Total Penghematan - {TANGGAL_HITUNG}\nHemat {penghematan_km:.1f} km ({penghematan_persen:.2f}%)")
ax5.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("5_total_penghematan.png", dpi=150)
plt.show()

# ==========================================================
# 16. VISUALISASI PENDUKUNG — PERBANDINGAN BEBAN PETUGAS SEBELUM vs SESUDAH VRP
# ==========================================================
manual_split = split_into_routes(intake_chrom, CAPACITY)
optimal_split = split_into_routes(best_chromosome, CAPACITY)
manual_loads = [route_distance(r, dist_matrix) for r in manual_split]
optimal_loads = [route_distance(r, dist_matrix) for r in optimal_split]
petugas_labels = [f"Petugas {i + 1}" for i in range(len(manual_loads))]

x = range(len(petugas_labels))
width = 0.35
fig6, ax6 = plt.subplots(figsize=(8, 4.5))
ax6.bar([i - width / 2 for i in x], manual_loads, width, label="Sebelum (rute berurut)", color="#E63946")
ax6.bar([i + width / 2 for i in x], optimal_loads, width, label="Sesudah (VRP-GA)", color="#2A9D8F")
ax6.set_xticks(list(x))
ax6.set_xticklabels(petugas_labels)
ax6.set_ylabel("Jarak tempuh (km)")
ax6.set_title(f"Perbandingan Beban Petugas - {TANGGAL_HITUNG}")
ax6.legend()
ax6.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("6_beban_petugas.png", dpi=150)
plt.show()

# ==========================================================
# 17. TABEL PENDUKUNG — URUTAN KUNJUNGAN (MANUAL & SOLUSI AKHIR)
# ==========================================================
def build_order_table(chromosome, kode_list, kelurahan_list, capacity=CAPACITY):
    routes = split_into_routes(chromosome, capacity)
    rows = []
    no = 1
    for p_idx, r in enumerate(routes, start=1):
        for urutan, idx in enumerate(r, start=1):
            rows.append([no, f"Petugas {p_idx}", urutan, kode_list[idx - 1], kelurahan_list[idx - 1]])
            no += 1
    return pd.DataFrame(rows, columns=["No", "Petugas", "Urutan ke-", "Kode Pelanggan", "Kelurahan"])

df_manual_order = build_order_table(intake_chrom, kode_pelanggan, kelurahan)
df_optimal_order = build_order_table(best_chromosome, kode_pelanggan, kelurahan)

render_table_image(df_manual_order, f"Urutan Kunjungan Manual - {TANGGAL_HITUNG}", "7_tabel_urutan_manual.png")
render_table_image(df_optimal_order,
                    f"Urutan Kunjungan Solusi Akhir (seed={best_seed}) - {TANGGAL_HITUNG}",
                    "8_tabel_urutan_optimal.png")

print("\nSemua keluaran tersimpan:")
print("1_multiple_run.png, 2_peta_rute.png, 3_fitness_per_generasi.png, 4_jarak_per_generasi.png,")
print("5_total_penghematan.png, 6_beban_petugas.png, 7_tabel_urutan_manual.png, 8_tabel_urutan_optimal.png,")
print("9_tabel_variasi_pc.png, 10_tabel_variasi_pm.png, 11_perbandingan_pc.png, 12_perbandingan_pm.png")