# PRD — AMR Compound Activity Predictor

Status: **Disepakati** — siap lanjut ke plan implementasi teknis.

## 1. Latar Belakang

Resistensi antimikroba (AMR) adalah salah satu tantangan kesehatan global yang mendesak, dan penemuan senyawa antibakteri baru merupakan proses yang panjang dan mahal. Sebelum kandidat senyawa masuk ke pengujian in vitro/in vivo, dibutuhkan tahap **skrining awal** untuk memprioritaskan senyawa mana yang paling layak diuji lebih lanjut — di sinilah pendekatan komputasional (in silico) berperan penting dalam mempercepat dan mengefisienkan alur riset.

**AMR Compound Activity Predictor** dikembangkan untuk menjawab kebutuhan tersebut: sebuah aplikasi berbasis machine learning yang memprediksi kemungkinan aktivitas antibakteri suatu senyawa terhadap target AMR, dilatih dari data bioaktivitas publik (ChEMBL) dan berjalan sepenuhnya lokal/offline tanpa bergantung pada layanan API eksternal — sehingga andal dipakai kapan saja tanpa risiko gangguan koneksi atau biaya berlangganan.

Proyek ini sekaligus selaras dengan konsentrasi Kimia Komputasi yang sedang ditempuh, dan dirancang agar dapat terus dikembangkan sebagai alat bantu riset jangka panjang, termasuk untuk mendukung penelitian tesis.

## 2. Masalah yang Diselesaikan

Peneliti/kimiawan yang punya daftar senyawa kandidat (misal dari sintesis atau skrining literatur) perlu cara cepat untuk **memprioritaskan** senyawa mana yang layak diuji lebih lanjut terhadap bakteri resisten, sebelum masuk ke uji in vitro yang mahal dan lama. Tools publik sejenis (SwissADME, pkCSM, ADMETlab) fokus ke sifat ADMET umum, bukan spesifik memprediksi aktivitas antibakteri terhadap target AMR — dan tidak bisa dilatih ulang/disesuaikan oleh penggunanya.

## 3. Tujuan (Goals)

1. Memprediksi kemungkinan suatu senyawa aktif secara antibakteri terhadap target AMR tertentu, dari input SMILES.
2. Memberi confidence/skor, bukan cuma label biner, supaya user bisa memprioritaskan.
3. Menampilkan insight sederhana (fitur struktural apa yang paling berpengaruh) supaya bukan black-box total.
4. Berjalan 100% offline setelah setup awal — tidak ada dependency ke API AI berbayar/rate-limited.

## 4. Target Pengguna

Peneliti, mahasiswa, dan praktisi kimia medisinal/kimia komputasi yang melakukan skrining awal senyawa kandidat antibakteri — baik di lingkungan akademik maupun industri farmasi — yang membutuhkan alat bantu prioritisasi cepat sebelum melangkah ke pengujian laboratorium.

## 5. Cakupan Data & Model

- **Sumber data**: ChEMBL (database bioaktivitas publik, gratis, bisa diunduh lewat `chembl_webresource_client` atau bulk download).
- **Target organisme**: dirancang **extensible** — daftar target didefinisikan di file konfigurasi terpisah (bukan hardcoded), dan tiap target punya model terlatih sendiri (satu organisme = satu model, karena data dan pola aktivitas berbeda antar organisme). Untuk MVP, dibangun **1 target lebih dulu**: ***Staphylococcus aureus*** (termasuk galur MRSA) — representasi data di ChEMBL paling banyak dibanding target AMR spesifik lain, dan MRSA adalah salah satu prioritas AMR global (WHO priority pathogen). Target lain (misal *E. coli*, *K. pneumoniae*) dapat ditambahkan kemudian dengan pola yang sama tanpa mengubah struktur app.
- **Label aktivitas**: senyawa dikategorikan "aktif" jika nilai MIC/IC50 pada assay ChEMBL berada di bawah ambang batas tertentu (konvensi umum di literatur QSAR: MIC ≤ 10 µM), "tidak aktif" jika di atasnya. Ambang ini didokumentasikan jelas di model card supaya bukan angka arbitrer yang disembunyikan.
- **Fitur (descriptor)**: RDKit molecular descriptors (MW, LogP, TPSA, jumlah H-bond donor/acceptor, dll) + Morgan fingerprint (representasi struktur).
- **Algoritma**: scikit-learn — Random Forest atau Gradient Boosting sebagai baseline (interpretable, cukup baik untuk dataset berukuran sedang, tidak butuh GPU).
- **Evaluasi**: train/test split (atau cross-validation karena data terbatas), metrik: accuracy, ROC-AUC, precision/recall — semua ditampilkan transparan di app, termasuk keterbatasannya (lihat §8).

## 6. Fitur & Struktur Menu

Didesain dengan filosofi serupa contoh dosen (e-CalQual: banyak menu, ada input data baru, ada dashboard) tapi domain compchem:

1. **Dashboard** — ringkasan dataset training (jumlah senyawa, distribusi aktif/tidak aktif), metrik performa model (accuracy, ROC-AUC), tanggal model terakhir dilatih.
2. **Skrining Senyawa** — input satu SMILES (quick check) atau upload file (CSV/daftar SMILES) untuk prediksi batch. Output: label prediksi + confidence score + descriptor kunci senyawa tersebut.
3. **Riwayat Prediksi** — semua senyawa yang pernah dicek tersimpan lokal (SQLite), bisa difilter dan diexport (CSV/Excel) — pola sama seperti app AMR Literature Triage sebelumnya.
4. **SAR Insight** — visualisasi feature importance dari model (descriptor apa yang paling berpengaruh terhadap prediksi), dan sebaran training data di ruang deskriptor (PCA/2D plot sederhana) untuk membantu user menilai apakah senyawa yang dicek "mirip" dengan data training atau di luar applicability domain.
5. **Model Info / Tentang** — sumber data, metodologi, disclaimer keterbatasan model (lihat §8), versi model.

## 7. Alur Teknis Singkat

1. **Skrip training (dijalankan sekali/berkala, offline)**: tarik data ChEMBL → bersihkan & label → hitung descriptor RDKit → latih model scikit-learn → simpan model (`model.joblib`) + metrik evaluasi (`model_metadata.json`).
2. **App Streamlit**: load model tersimpan → user input SMILES → hitung descriptor sama seperti training → `model.predict_proba()` → tampilkan hasil. Tidak ada pemanggilan API eksternal apa pun saat runtime app.

## 8. Keterbatasan & Disclaimer (wajib ditampilkan di app)

- Model ini alat bantu skrining awal (**in silico**), bukan pengganti uji in vitro/in vivo — hasil harus diverifikasi eksperimental.
- Data ChEMBL punya noise (variasi metode assay antar-lab), dan definisi ambang batas "aktif" adalah konvensi, bukan kebenaran absolut.
- Model hanya reliable untuk senyawa yang secara struktural mirip dengan data training (applicability domain) — prediksi untuk scaffold yang sangat berbeda berisiko tidak akurat, karenanya fitur SAR Insight (§6.4) penting untuk transparansi ini.
- Dataset kemungkinan besar tidak besar (ribuan, bukan jutaan senyawa) — performa model akan dilaporkan apa adanya, termasuk kalau ternyata sedang/kurang bagus, bukan diklaim sempurna.

## 9. Non-Goals (di luar cakupan MVP)

- Docking molekuler atau simulasi 3D.
- Melatih model untuk lebih dari 1 target organisme di MVP (arsitektur mendukung penambahan, tapi implementasi awal fokus 1 target dulu — lihat §5).
- Deployment sebagai layanan publik multi-user — ini alat riset yang dijalankan lokal oleh masing-masing pengguna.

## 10. Kriteria Selesai (Definition of Done)

- Skrip training berhasil menghasilkan model dengan metrik evaluasi yang terdokumentasi (berapa pun angkanya, asal jujur dan dijelaskan).
- App Streamlit jalan offline penuh, semua 5 menu berfungsi dengan data nyata (bukan dummy).
- User bisa input SMILES baru dan dapat hasil prediksi + confidence dalam hitungan detik.
- Riwayat prediksi tersimpan dan bisa diexport.
- Disclaimer keterbatasan model tampil jelas di app (bukan disembunyikan di README saja).

## 11. Risiko Implementasi

- **Ukuran & kualitas data ChEMBL untuk target terpilih** belum diverifikasi — perlu langkah eksplorasi data dulu sebelum komit ke pipeline training, karena kalau datanya terlalu sedikit/tidak seimbang, perlu pertimbangkan target alternatif atau strategi penyeimbangan data.
- **Environment**: perlu install `rdkit`, `scikit-learn`, `chembl_webresource_client` (belum ada di environment ini, beda dari app sebelumnya).
