# AMR Compound Activity Predictor

Aplikasi skrining senyawa kandidat antibakteri terhadap *Staphylococcus aureus* (termasuk MRSA), menggunakan model machine learning yang dilatih sendiri dari data publik ChEMBL. Lihat [PRD.md](PRD.md) untuk latar belakang dan desain lengkap.

**100% offline saat dipakai** — tidak ada API AI eksternal yang dipanggil saat runtime. Model dilatih sekali dari data ChEMBL, disimpan lokal, dan aplikasi Streamlit-nya cuma memuat file model itu.

## Setup (sekali saja)

```
pip install -r requirements.txt
```

Model sudah dilatih dan tersimpan di `models/` — tidak perlu dilatih ulang kecuali mau mengganti target/parameter (lihat bagian Training di bawah).

## Menjalankan aplikasi

```
streamlit run app.py
```

Buka di browser (biasanya http://localhost:8501). Ada 5 menu:

1. **Dashboard** — statistik dataset training & performa model (akurasi, ROC-AUC, confusion matrix).
2. **Skrining Senyawa** — input SMILES satu per satu, atau upload CSV (kolom `smiles`) untuk cek banyak senyawa sekaligus.
3. **Riwayat Prediksi** — semua senyawa yang pernah dicek, bisa difilter & diexport ke CSV/Excel.
4. **SAR Insight** — fitur/descriptor apa yang paling menentukan prediksi model, dan visualisasi apakah senyawa yang dicek "mirip" data training (applicability domain).
5. **Model Info** — detail metodologi model dan disclaimer keterbatasannya.

## Training ulang model (opsional)

Kalau mau melatih ulang (misal ganti threshold, atau server ChEMBL sudah lebih stabil dan mau ambil lebih banyak data):

```
python train_model.py --target saureus_mic
```

Catatan:
- Proses ini menarik data dari API ChEMBL (`www.ebi.ac.uk/chembl/api/data`), butuh koneksi internet — tapi ini **hanya untuk training**, bukan untuk pemakaian app sehari-hari.
- Server ChEMBL kadang tidak stabil (timeout/error 500 intermiten) — skrip ini sudah punya retry otomatis dan cache lokal (`data/raw_activities_cache_<target>.jsonl`) supaya kalau proses terputus, menjalankan ulang akan **melanjutkan dari titik terakhir**, bukan mulai dari nol. Kalau mau benar-benar mulai ulang, hapus file cache tersebut dulu.
- Defaultnya dibatasi ambil maksimal 30.000 record mentah dari ChEMBL (`MAX_RECORDS_TO_FETCH` di `train_model.py`) — sudah menghasilkan ±12.000 senyawa unik, lebih dari cukup untuk model yang solid. Bisa dinaikkan kalau mau dataset lebih besar (dengan konsekuensi waktu fetch lebih lama).

## Menambah target/organisme baru

Tambahkan satu entri baru di `config.py` (lihat contoh yang sudah ada), lalu jalankan `python train_model.py --target <id-baru>`. Tidak perlu mengubah kode lain — lihat PRD §5 untuk detail desain extensible ini.

## Keterbatasan

Model ini alat bantu skrining awal (in silico), bukan pengganti uji laboratorium — lihat tab **Model Info** di aplikasi atau PRD §8 untuk detail lengkap keterbatasannya.
