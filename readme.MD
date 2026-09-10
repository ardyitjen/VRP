![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-tugas%20akhir-lightgrey)

# Optimasi Rute Pemeriksaan SLO menggunakan Algoritma Genetika berbasis CVRP

Repository ini berisi kode dan dataset yang digunakan dalam tugas akhir yang membahas penerapan *Genetic Algorithm* (GA) untuk menyelesaikan *Capacitated Vehicle Routing Problem* (CVRP) pada rute pemeriksaan Sertifikat Laik Operasi (SLO) instalasi listrik di sebuah perusahaan yang beroperasi di Kota Medan.

## Latar Belakang

SLO adalah dokumen legal yang menyatakan bahwa instalasi listrik telah memenuhi standar keselamatan dan layak dioperasikan. Proses penerbitannya melibatkan kunjungan pemeriksaan lapangan oleh petugas ke titik lokasi pelanggan yang tersebar di berbagai kelurahan dan kecamatan. Penentuan urutan kunjungan yang dilakukan secara manual berpotensi menghasilkan rute yang kurang efisien dari segi jarak tempuh.

Tugas akhir ini memodelkan permasalahan tersebut sebagai CVRP, dengan kapasitas kunjungan per petugas dibatasi tujuh titik per hari, kemudian menyelesaikannya menggunakan algoritma genetika.

## Metode

Pendekatan yang digunakan meliputi

- Representasi kromosom berbentuk *giant tour permutation* dengan skema *route-first cluster-second*, dipecah tiap tujuh gen sesuai kapasitas petugas
- *Order Crossover* (OX) sebagai operator persilangan
- *Swap Mutation* sebagai operator mutasi
- *Roulette Wheel Selection* sebagai mekanisme seleksi
- Jarak antar titik dihitung menggunakan rumus *Haversine*
- Nilai fitness dihitung sebagai kebalikan dari total jarak tempuh (f = 1/D)
- Setiap hari dijalankan sebanyak 50 kali (multiple run, seed 1 sampai 50), solusi akhir diambil dari run terbaik

Parameter GA yang digunakan dalam eksperimen

| Parameter | Nilai |
|---|---|
| Ukuran populasi (N) | 100 |
| Probabilitas crossover (Pc) | 0.8 |
| Probabilitas mutasi (Pm) | 0.1 |
| Jumlah generasi maksimum | 100 |
| Jumlah run per hari | 50 |
| Kapasitas per petugas | 7 titik kunjungan |

## Dataset

File `data_set_mdn_praproses.xlsx` berisi data kunjungan pemeriksaan SLO yang telah melalui tahap praproses, dengan rincian sebagai berikut

- Total 4.462 baris data
- 224 hari dengan jumlah titik kunjungan valid (tujuh titik atau lebih per hari)
- Mencakup 26 kecamatan dan 167 kelurahan di Kota Medan
- Koordinat depot berada pada titik (3.546833, 98.645710)

Koordinat pada dataset direkonstruksi menggunakan layanan geocoding Nominatim (OpenStreetMap), dengan mekanisme fallback bertingkat menuju sentroid kelurahan atau kecamatan apabila alamat tidak dapat ditemukan secara langsung, serta kamus koreksi manual sebagai prioritas tertinggi. Beberapa titik memiliki koordinat yang identik karena merepresentasikan pengajuan kolektif dari satu bangunan atau kompleks yang sama, bukan karena kesalahan data.

## Struktur Repository

```
.
├── VRP.py                                          # Kode utama GA-CVRP, dijalankan berdiri sendiri di Google Colab
├── Karakteristik_Data_Pengujian_4.1.py             # Analisis karakteristik dataset (Bab IV subbab 4.1)
├── Eksekusi_Seluruh_224_Hari_4.2.py                # Eksekusi penuh terhadap 224 hari valid (Bab IV subbab 4.2)
├── Pengujian_Variasi_Parameter_4.3.py              # Pengujian variasi parameter GA (Bab IV subbab 4.3)
├── Analisis_Konsistensi_Multiple-Run_4.5.py        # Analisis konsistensi antar run atau CV (Bab IV subbab 4.5)
├── data_set_mdn_praproses.xlsx                     # Dataset hasil praproses, digunakan oleh seluruh skrip
└── Data Set Medan 2025.xlsx                        # Dataset mentah sebelum praproses, disertakan sebagai referensi
```

Catatan, tidak ada skrip terpisah untuk subbab 4.4 di Bab IV.

## Cara Menjalankan

Setiap skrip dirancang untuk dijalankan sebagai satu sel kode di Google Colab.

1. Unggah `data_set_mdn_praproses.xlsx` ke sesi Colab atau hubungkan ke Google Drive
2. Buka skrip yang ingin dijalankan (`VRP.py` untuk proses GA-CVRP utama, atau salah satu skrip 4.1 sampai 4.5 untuk analisis Bab IV), salin isinya ke satu sel Colab
3. Sesuaikan path dataset pada bagian pembacaan data jika diperlukan
4. Jalankan sel tersebut

`VRP.py` berdiri sendiri dan tidak bergantung pada skrip 4.1 sampai 4.5. Skrip 4.1 sampai 4.5 masing-masing berdiri sendiri untuk keperluan analisis tertentu di Bab IV.

Dependensi utama yang dibutuhkan meliputi `pandas`, `numpy`, `geopy` (untuk Nominatim), dan `matplotlib` untuk visualisasi hasil.

## Hasil Utama

Berdasarkan eksekusi terhadap seluruh 224 hari valid dalam dataset

- Rata-rata penghematan jarak sebesar 20.39 persen, dari rata-rata 122.87 km menjadi 97.77 km
- Korelasi antara persentase penghematan dengan volume titik kunjungan harian sebesar r = -0.42
- Rata-rata *coefficient of variation* (CV) dari 50 run per hari, dirata-ratakan ke seluruh 224 hari, sebesar 2.48 persen

Dua kasus khusus ditemukan selama eksperimen. Pada 9 Januari 2025 dengan 87 titik kunjungan, algoritma mengalami konvergensi dini sehingga penghematan yang diperoleh hanya 2.37 persen. Pada 3 Mei 2025, penghematan tercatat 0 persen karena 50 dari 51 titik kunjungan memiliki koordinat geocoding yang identik. Pengujian variasi jumlah generasi maksimum (50, 100, dan 200) pada kasus konvergensi dini menunjukkan bahwa penambahan generasi tidak memperbaiki hasil, sehingga masalah ini bukan disebabkan oleh jumlah generasi yang kurang.

## Batasan

Koordinat pada dataset merupakan hasil geocoding retrospektif menggunakan Nominatim, bukan koordinat yang dicatat langsung pada saat pengajuan. Pendekatan ini berpotensi menghasilkan ketidakakuratan posisi, terutama untuk kasus dengan koordinat identik akibat pengajuan kolektif. Sebagai rekomendasi metodologis, pencatatan koordinat sebaiknya dilakukan langsung pada saat pengajuan permohonan, bukan direkonstruksi secara retrospektif.

## Catatan

Repository ini disusun untuk keperluan tugas akhir. Nama perusahaan pada dataset dan pembahasan telah dianonimkan.