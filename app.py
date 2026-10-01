"""AMR Compound Activity Predictor — Streamlit app (100% offline inference)."""
import io
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from rdkit.Chem.Draw import rdMolDraw2D

from config import DEFAULT_TARGET_ID, get_target
from db import (
    clear_predictions,
    delete_compound,
    delete_predictions,
    fetch_all,
    fetch_compounds,
    get_connection,
    insert_prediction,
    save_compound,
)
from featurize import DESCRIPTOR_NAMES, featurize, smiles_to_mol

MODELS_DIR = Path(__file__).parent / "models"

st.set_page_config(
    page_title="AMR Compound Activity Predictor",
    page_icon="🧫",
    layout="wide",
)

ACTIVE_COLOR = "#16a34a"
INACTIVE_COLOR = "#64748b"

# Label tampilan untuk tiap descriptor: (nama Indonesia, satuan, jumlah desimal)
DESCRIPTOR_LABELS = {
    "MolWt": ("Berat Molekul", "g/mol", 2),
    "LogP": ("LogP (lipofilisitas)", "", 2),
    "TPSA": ("TPSA (luas permukaan polar)", "Å²", 2),
    "NumHDonors": ("Donor Ikatan Hidrogen", "", 0),
    "NumHAcceptors": ("Akseptor Ikatan Hidrogen", "", 0),
    "NumRotatableBonds": ("Ikatan Dapat Berputar", "", 0),
    "NumAromaticRings": ("Cincin Aromatik", "", 0),
    "RingCount": ("Jumlah Cincin", "", 0),
    "FractionCSP3": ("Fraksi Karbon sp³", "", 3),
}

CONTOH_SENYAWA = {
    "Vancomycin (antibiotik MRSA)": "CC(C)C[C@H](NC)C(=O)N[C@H]1C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H]2C(=O)N[C@@H]3C(=O)N[C@H](C(=O)N[C@H](C(O)=O)c4cc(O)cc(O)c4-c4cc3ccc4O)[C@H](O)c3ccc(c(Cl)c3)Oc3cc2cc(c3O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O[C@H]2C[C@](C)(N)[C@H](O)[C@H](C)O2)Oc2ccc(cc2Cl)[C@H]1O",
    "Amoksisilin (beta-laktam)": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "Aspirin (bukan antibiotik)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Kafein (bukan antibiotik)": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
}

PILIHAN_KOSONG = "— ketik SMILES sendiri —"

CSV_CONTOH = (
    "smiles\n"
    "CC(=O)OC1=CC=CC=C1C(=O)O\n"
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C\n"
    "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O\n"
)


# ---------------------------------------------------------------- loaders (cached)
@st.cache_resource
def load_model(target_id):
    return joblib.load(MODELS_DIR / f"model_{target_id}.joblib")


@st.cache_resource
def load_pca(target_id):
    path = MODELS_DIR / f"pca_{target_id}.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_metadata(target_id):
    return json.loads((MODELS_DIR / f"metadata_{target_id}.json").read_text())


@st.cache_data
def load_pca_points(target_id):
    path = MODELS_DIR / f"pca_training_points_{target_id}.csv"
    return pd.read_csv(path) if path.exists() else None


def draw_molecule(mol, size=(430, 330)):
    """Render a molecule to PNG bytes on a white canvas (readable in any theme)."""
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().clearBackground = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def result_card(label, confidence):
    color = ACTIVE_COLOR if label == "Aktif" else INACTIVE_COLOR
    st.markdown(
        f"""
        <div style="border-radius:14px;padding:20px 22px;background:{color}1a;
                    border:1px solid {color}55;margin-bottom:14px;">
          <div style="font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;opacity:.7;">
            Hasil Prediksi
          </div>
          <div style="font-size:2.1rem;font-weight:700;color:{color};line-height:1.25;">
            {label}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Tingkat keyakinan model: **{confidence:.1%}**")
    st.progress(float(confidence))


def descriptor_grid(descriptors, columns=3):
    items = list(DESCRIPTOR_LABELS.items())
    for row_start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, (key, (label, unit, decimals)) in zip(cols, items[row_start:row_start + columns]):
            value = descriptors[key]
            shown = f"{value:,.{decimals}f}" + (f" {unit}" if unit else "")
            col.metric(label, shown, border=True)


def lipinski_summary(descriptors):
    """Rule of Five — aturan keterobatan oral yang lazim dipakai di kimia medisinal."""
    aturan = [
        ("Berat molekul ≤ 500", descriptors["MolWt"] <= 500, f"{descriptors['MolWt']:.1f}"),
        ("LogP ≤ 5", descriptors["LogP"] <= 5, f"{descriptors['LogP']:.2f}"),
        ("Donor ikatan H ≤ 5", descriptors["NumHDonors"] <= 5, f"{descriptors['NumHDonors']:.0f}"),
        ("Akseptor ikatan H ≤ 10", descriptors["NumHAcceptors"] <= 10,
         f"{descriptors['NumHAcceptors']:.0f}"),
    ]
    pelanggaran = sum(1 for _, lolos, _ in aturan if not lolos)
    df = pd.DataFrame({
        "Kriteria": [a[0] for a in aturan],
        "Nilai": [a[2] for a in aturan],
        "Status": ["Memenuhi" if a[1] else "Tidak memenuhi" for a in aturan],
    })
    return df, pelanggaran


# ---------------------------------------------------------------- setup
target_cfg = get_target(DEFAULT_TARGET_ID)
model_path = MODELS_DIR / f"model_{target_cfg['id']}.joblib"

if not model_path.exists():
    st.error(
        f"Model belum dilatih. Jalankan `python train_model.py --target {target_cfg['id']}` "
        "dari terminal terlebih dahulu."
    )
    st.stop()

model = load_model(target_cfg["id"])
pca = load_pca(target_cfg["id"])
metadata = load_metadata(target_cfg["id"])
conn = get_connection()

with st.sidebar:
    st.markdown("## 🧫 AMR Predictor")
    st.caption("Skrining aktivitas antibakteri berbasis machine learning")
    st.divider()
    halaman = st.radio(
        "Menu",
        ["Dashboard", "Skrining Senyawa", "Riwayat Prediksi", "SAR Insight", "Model Info"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**Target organisme**")
    st.markdown(f"*{target_cfg['label']}*")
    st.caption(f"Ambang aktif: MIC ≤ {target_cfg['active_threshold_um']} µM")
    st.caption(f"Akurasi model: **{metadata['metrics']['accuracy']:.1%}**")
    st.caption("Mode: 🔒 offline (tanpa API eksternal)")


def predict(smiles):
    vector, descriptors = featurize(smiles)
    if vector is None:
        return None
    proba = model.predict_proba(vector.reshape(1, -1))[0]
    label = "Aktif" if proba[1] >= 0.5 else "Tidak Aktif"

    pc1 = pc2 = None
    if pca is not None:
        coords = pca.transform(vector[: len(DESCRIPTOR_NAMES)].reshape(1, -1))
        pc1, pc2 = float(coords[0, 0]), float(coords[0, 1])

    return {
        "label": label,
        "confidence": float(max(proba)),
        "descriptors": descriptors,
        "pc1": pc1,
        "pc2": pc2,
    }


def save_prediction(smiles, result, nama=None):
    insert_prediction(conn, {
        "compound_name": nama or None,
        "smiles": smiles,
        "canonical_smiles": smiles,
        "target_id": target_cfg["id"],
        "predicted_label": result["label"],
        "confidence": result["confidence"],
        "mol_wt": result["descriptors"]["MolWt"],
        "log_p": result["descriptors"]["LogP"],
        "tpsa": result["descriptors"]["TPSA"],
        "num_h_donors": result["descriptors"]["NumHDonors"],
        "num_h_acceptors": result["descriptors"]["NumHAcceptors"],
        "pc1": result["pc1"],
        "pc2": result["pc2"],
    })


# ---------------------------------------------------------------- Dashboard
if halaman == "Dashboard":
    st.title("Dashboard")
    st.caption(f"Ringkasan dataset dan performa model untuk {target_cfg['label']}")

    m = metadata["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Senyawa Training", f"{metadata['n_compounds']:,}", border=True)
    c2.metric("Senyawa Aktif", f"{metadata['n_active']:,}", border=True)
    c3.metric("Akurasi", f"{m['accuracy']:.1%}", border=True)
    c4.metric("ROC-AUC", f"{m['roc_auc']:.3f}", border=True)

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Distribusi Kelas Data Training")
        dist = pd.DataFrame({
            "Kelas": ["Aktif", "Tidak Aktif"],
            "Jumlah Senyawa": [metadata["n_active"], metadata["n_inactive"]],
        })
        st.bar_chart(dist, x="Kelas", y="Jumlah Senyawa", height=280)

    with right:
        st.subheader("Metrik Evaluasi (data uji)")
        metrik_df = pd.DataFrame({
            "Metrik": ["Akurasi", "ROC-AUC", "Precision", "Recall", "F1-Score"],
            "Nilai": [m["accuracy"], m["roc_auc"], m["precision"], m["recall"], m["f1"]],
        })
        st.dataframe(
            metrik_df, hide_index=True, width="stretch",
            column_config={
                "Nilai": st.column_config.ProgressColumn(
                    "Nilai", format="%.3f", min_value=0.0, max_value=1.0
                )
            },
        )
        st.caption(f"Dilatih pada {m['n_train']:,} senyawa, diuji pada {m['n_test']:,} senyawa.")

    st.subheader("Confusion Matrix (data uji)")
    cm = m["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Aktual: Tidak Aktif", "Aktual: Aktif"],
        columns=["Prediksi: Tidak Aktif", "Prediksi: Aktif"],
    )
    st.dataframe(cm_df, width="stretch")
    st.caption(
        "Diagonal (kiri-atas & kanan-bawah) = prediksi benar. "
        "Sisanya adalah kesalahan prediksi model."
    )
    st.caption(f"Model terakhir dilatih: {metadata['trained_at'][:19].replace('T', ' ')} UTC")

# ---------------------------------------------------------------- Skrining Senyawa
elif halaman == "Skrining Senyawa":
    st.title("Skrining Senyawa")
    st.caption("Masukkan struktur senyawa dalam format SMILES untuk memprediksi aktivitas antibakterinya.")

    with st.expander("Apa itu SMILES dan dari mana mendapatkannya?"):
        st.markdown("""
**SMILES** (*Simplified Molecular Input Line Entry System*) adalah cara menuliskan struktur
molekul sebagai satu baris teks. Contohnya, aspirin ditulis `CC(=O)OC1=CC=CC=C1C(=O)O`.

Kode SMILES suatu senyawa bisa diperoleh dengan beberapa cara:

- **PubChem** (pubchem.ncbi.nlm.nih.gov) — cari nama senyawanya, lalu salin bagian *Canonical SMILES*
- **ChemDraw, MarvinSketch, atau Avogadro** — gambar strukturnya, lalu salin sebagai SMILES
- **Dropdown contoh di bawah** — berisi beberapa senyawa siap pakai untuk mencoba

Senyawa yang Anda beri nama akan tersimpan dan muncul di dropdown tersebut, sehingga bisa
dipanggil kembali tanpa menyalin ulang SMILES-nya.
""")

    tersimpan = {f"★ {c['name']}": c["smiles"] for c in fetch_compounds(conn)}
    semua_pilihan = {**tersimpan, **CONTOH_SENYAWA}

    pilihan = st.selectbox(
        "Pilih senyawa tersimpan atau contoh (opsional)",
        [PILIHAN_KOSONG] + list(semua_pilihan.keys()),
    )
    nilai_awal = semua_pilihan.get(pilihan, "")
    nama_awal = pilihan[2:] if pilihan.startswith("★ ") else ""

    with st.form("single_form"):
        c_nama, c_smiles = st.columns([1, 2])
        nama_input = c_nama.text_input(
            "Nama senyawa (opsional)", value=nama_awal, placeholder="misal: Senyawa uji A"
        )
        smiles_input = c_smiles.text_input(
            "SMILES", value=nilai_awal, placeholder="CC(=O)OC1=CC=CC=C1C(=O)O"
        )
        simpan = st.checkbox(
            "Simpan ke daftar senyawa saya (agar muncul di dropdown)", value=bool(nama_awal)
        )
        submitted = st.form_submit_button("Prediksi Aktivitas", type="primary")

    if submitted:
        smiles = smiles_input.strip()
        nama = nama_input.strip()
        if not smiles:
            st.warning("Isi kolom SMILES terlebih dahulu.")
        else:
            mol = smiles_to_mol(smiles)
            if mol is None:
                st.error(
                    "SMILES tidak valid dan tidak bisa diproses. "
                    "Periksa kembali formatnya (pastikan tanda kurung dan cincin tertutup dengan benar)."
                )
            else:
                result = predict(smiles)
                save_prediction(smiles, result, nama)

                if simpan:
                    if nama:
                        save_compound(conn, nama, smiles)
                        st.success(f"Senyawa **{nama}** tersimpan ke daftar Anda.")
                    else:
                        st.warning("Beri nama senyawanya dulu agar bisa disimpan ke daftar.")

                kiri, kanan = st.columns([1, 1])
                with kiri:
                    st.image(draw_molecule(mol), caption=nama or "Struktur molekul")
                with kanan:
                    result_card(result["label"], result["confidence"])
                    if result["label"] == "Aktif":
                        st.info(
                            "Model memperkirakan senyawa ini **berpotensi aktif** terhadap "
                            f"{target_cfg['label']}. Tetap perlu konfirmasi uji laboratorium."
                        )
                    else:
                        st.info(
                            "Model memperkirakan senyawa ini **kurang berpotensi aktif** terhadap "
                            f"{target_cfg['label']} pada ambang MIC ≤ {target_cfg['active_threshold_um']} µM."
                        )

                st.divider()
                st.subheader("Sifat Fisikokimia")
                descriptor_grid(result["descriptors"])

                st.subheader("Aturan Lipinski (Rule of Five)")
                lip_df, pelanggaran = lipinski_summary(result["descriptors"])
                st.dataframe(lip_df, hide_index=True, width="stretch")
                if pelanggaran == 0:
                    st.caption("Memenuhi seluruh kriteria Lipinski.")
                else:
                    st.caption(
                        f"Terdapat {pelanggaran} kriteria yang tidak terpenuhi. Perlu dicatat bahwa "
                        "banyak antibiotik memang melanggar aturan ini — vankomisin, misalnya, "
                        "melanggar hampir seluruhnya namun tetap efektif secara klinis. Aturan "
                        "Lipinski memperkirakan keterserapan obat oral, bukan potensi aktivitasnya."
                    )

    st.divider()
    with st.expander("Kelola daftar senyawa saya"):
        daftar = fetch_compounds(conn)
        if not daftar:
            st.caption("Belum ada senyawa tersimpan. Beri nama saat melakukan prediksi di atas.")
        else:
            st.dataframe(
                pd.DataFrame(daftar)[["name", "smiles", "created_at"]],
                hide_index=True, width="stretch",
                column_config={
                    "name": st.column_config.TextColumn("Nama"),
                    "smiles": st.column_config.TextColumn("SMILES", width="large"),
                    "created_at": st.column_config.TextColumn("Disimpan pada"),
                },
            )
            hapus = st.selectbox("Hapus senyawa", [""] + [c["name"] for c in daftar])
            if st.button("Hapus", disabled=not hapus):
                delete_compound(conn, hapus)
                st.rerun()

    st.divider()
    st.subheader("Skrining Banyak Senyawa Sekaligus")
    st.caption("Unggah berkas CSV yang memiliki kolom bernama `smiles`.")
    st.download_button(
        "⬇️ Unduh contoh berkas CSV", CSV_CONTOH, "contoh_senyawa.csv", "text/csv"
    )
    uploaded = st.file_uploader("Pilih berkas CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        if "smiles" not in batch_df.columns:
            st.error("Berkas CSV harus memiliki kolom bernama `smiles`.")
        else:
            progress = st.progress(0.0, text="Memproses senyawa...")
            hasil = []
            for i, smiles in enumerate(batch_df["smiles"]):
                smiles = str(smiles).strip()
                result = predict(smiles)
                if result is None:
                    hasil.append({"SMILES": smiles, "Prediksi": "SMILES tidak valid", "Confidence": None})
                else:
                    save_prediction(smiles, result)
                    hasil.append({
                        "SMILES": smiles,
                        "Prediksi": result["label"],
                        "Confidence": result["confidence"] * 100,
                    })
                progress.progress((i + 1) / len(batch_df), text=f"Memproses {i + 1}/{len(batch_df)} senyawa...")

            progress.empty()
            hasil_df = pd.DataFrame(hasil)
            n_aktif = int((hasil_df["Prediksi"] == "Aktif").sum())
            n_invalid = int((hasil_df["Prediksi"] == "SMILES tidak valid").sum())
            st.success(
                f"Selesai memproses {len(hasil_df)} senyawa — {n_aktif} diprediksi aktif"
                + (f", {n_invalid} SMILES tidak valid." if n_invalid else ".")
            )
            st.dataframe(
                hasil_df, hide_index=True, width="stretch",
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", format="%.1f%%", min_value=0, max_value=100
                    )
                },
            )

# ---------------------------------------------------------------- Riwayat Prediksi
elif halaman == "Riwayat Prediksi":
    st.title("Riwayat Prediksi")
    st.caption("Semua senyawa yang pernah diskrining, tersimpan lokal di komputer ini.")

    rows = fetch_all(conn, target_id=target_cfg["id"])
    if not rows:
        st.info("Belum ada prediksi tersimpan. Mulai dari menu **Skrining Senyawa**.")
    else:
        df = pd.DataFrame(rows)

        c1, c2, c3 = st.columns([1, 1, 2])
        filter_label = c1.selectbox("Filter hasil", ["Semua", "Aktif", "Tidak Aktif"])
        urutan = c2.selectbox("Urutkan", ["Terbaru", "Confidence tertinggi"])
        cari = c3.text_input("Cari nama atau SMILES", placeholder="ketik sebagian nama/SMILES...")

        filtered = df.copy()
        if filter_label != "Semua":
            filtered = filtered[filtered["predicted_label"] == filter_label]
        if cari:
            cocok_smiles = filtered["smiles"].str.contains(cari, case=False, na=False)
            cocok_nama = filtered["compound_name"].fillna("").str.contains(cari, case=False, na=False)
            filtered = filtered[cocok_smiles | cocok_nama]
        filtered = filtered.sort_values(
            "confidence" if urutan == "Confidence tertinggi" else "predicted_at",
            ascending=False,
        )

        n_aktif = int((filtered["predicted_label"] == "Aktif").sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Ditampilkan", len(filtered), border=True)
        k2.metric("Diprediksi Aktif", n_aktif, border=True)
        k3.metric("Diprediksi Tidak Aktif", len(filtered) - n_aktif, border=True)

        tampil = filtered[[
            "id", "compound_name", "smiles", "predicted_label", "confidence",
            "mol_wt", "log_p", "tpsa", "predicted_at",
        ]].copy()
        tampil["compound_name"] = tampil["compound_name"].fillna("—")
        tampil["confidence"] = tampil["confidence"] * 100

        st.caption("Centang baris di kolom paling kiri untuk menghapusnya.")
        seleksi = st.dataframe(
            tampil, hide_index=True, width="stretch",
            on_select="rerun", selection_mode="multi-row",
            column_config={
                "id": None,
                "compound_name": st.column_config.TextColumn("Nama Senyawa"),
                "smiles": st.column_config.TextColumn("SMILES", width="medium"),
                "predicted_label": st.column_config.TextColumn("Prediksi"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", format="%.1f%%", min_value=0, max_value=100
                ),
                "mol_wt": st.column_config.NumberColumn("Berat Molekul", format="%.1f"),
                "log_p": st.column_config.NumberColumn("LogP", format="%.2f"),
                "tpsa": st.column_config.NumberColumn("TPSA", format="%.1f"),
                "predicted_at": st.column_config.TextColumn("Waktu Prediksi"),
            },
        )

        terpilih = seleksi.selection.rows if hasattr(seleksi, "selection") else []
        h1, h2 = st.columns(2)
        if h1.button(
            f"🗑️ Hapus {len(terpilih)} baris terpilih", disabled=not terpilih, width="stretch"
        ):
            delete_predictions(conn, tampil.iloc[terpilih]["id"].tolist())
            st.rerun()
        if h2.button("Kosongkan seluruh riwayat", width="stretch"):
            clear_predictions(conn, target_cfg["id"])
            st.rerun()

        st.divider()
        e1, e2 = st.columns(2)
        e1.download_button(
            "⬇️ Export CSV", filtered.to_csv(index=False).encode("utf-8"),
            "riwayat_prediksi.csv", "text/csv", width="stretch",
        )
        excel_buffer = io.BytesIO()
        filtered.to_excel(excel_buffer, index=False, engine="openpyxl")
        e2.download_button(
            "⬇️ Export Excel", excel_buffer.getvalue(), "riwayat_prediksi.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

# ---------------------------------------------------------------- SAR Insight
elif halaman == "SAR Insight":
    st.title("SAR Insight")
    st.caption("Memahami dasar keputusan model dan batas keandalannya (applicability domain).")

    st.subheader("Fitur Paling Berpengaruh")
    classifier = model.named_steps["clf"]
    importances = classifier.feature_importances_
    n_desc = len(DESCRIPTOR_NAMES)

    importance_rows = [
        {"Fitur": DESCRIPTOR_LABELS[name][0], "Kontribusi": float(importances[i])}
        for i, name in enumerate(DESCRIPTOR_NAMES)
    ]
    importance_rows.append({
        "Fitur": "Fingerprint (pola struktur)",
        "Kontribusi": float(importances[n_desc:].sum()),
    })
    importance_df = pd.DataFrame(importance_rows)

    chart = (
        alt.Chart(importance_df)
        .mark_bar(color="#16a34a")
        .encode(
            x=alt.X("Kontribusi:Q", title="Kontribusi terhadap prediksi"),
            y=alt.Y("Fitur:N", sort="-x", title=None),
            tooltip=["Fitur", alt.Tooltip("Kontribusi:Q", format=".4f")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Fingerprint struktur ditampilkan sebagai satu batang gabungan karena terdiri dari "
        "1.024 bit yang tidak bermakna bila dibaca satu per satu — nilainya merupakan total "
        "kontribusi seluruh pola struktur."
    )

    st.divider()
    st.subheader("Applicability Domain")
    pca_points = load_pca_points(target_cfg["id"])
    if pca_points is None:
        st.info("Data PCA training belum tersedia.")
    else:
        plot_df = pd.DataFrame({
            "PC1": pca_points["pc1"],
            "PC2": pca_points["pc2"],
            "Kelompok": pca_points["active"].map({1: "Training: Aktif", 0: "Training: Tidak Aktif"}),
        })

        history = pd.DataFrame(fetch_all(conn, target_id=target_cfg["id"]))
        if not history.empty:
            history = history.dropna(subset=["pc1", "pc2"])
            if not history.empty:
                plot_df = pd.concat([plot_df, pd.DataFrame({
                    "PC1": history["pc1"],
                    "PC2": history["pc2"],
                    "Kelompok": "Senyawa yang Anda cek",
                })], ignore_index=True)

        st.scatter_chart(plot_df, x="PC1", y="PC2", color="Kelompok", height=420)
        st.caption(
            "Grafik ini memproyeksikan senyawa ke ruang 2 dimensi berdasarkan sifat fisikokimianya. "
            "Senyawa yang posisinya jauh dari kerumunan data training berada di luar "
            "*applicability domain* model, sehingga prediksinya perlu ditafsirkan lebih hati-hati. "
            "Ini indikasi kasar, bukan ukuran presisi."
        )

# ---------------------------------------------------------------- Model Info
elif halaman == "Model Info":
    st.title("Model Info")
    st.caption("Transparansi metodologi, sumber data, dan batasan model.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Spesifikasi Model")
        st.markdown(f"""
| | |
|---|---|
| **Target organisme** | {metadata['target_label']} |
| **ChEMBL Target ID** | `{metadata['chembl_target_id']}` |
| **Sumber data** | ChEMBL (basis data bioaktivitas publik) |
| **Jumlah senyawa** | {metadata['n_compounds']:,} ({metadata['n_active']:,} aktif / {metadata['n_inactive']:,} tidak aktif) |
| **Definisi aktif** | MIC ≤ {metadata['active_threshold_um']} µM |
| **Algoritma** | Random Forest Classifier (scikit-learn) |
| **Fitur** | 9 descriptor fisikokimia RDKit + Morgan fingerprint 1.024 bit |
| **Terakhir dilatih** | {metadata['trained_at'][:19].replace('T', ' ')} UTC |
""")

    with c2:
        st.subheader("Cara Kerja")
        st.markdown("""
1. Struktur senyawa (SMILES) diurai dan divalidasi menggunakan RDKit.
2. Dihitung sifat fisikokimianya serta pola substruktur (Morgan fingerprint).
3. Model Random Forest yang telah dilatih dari data ChEMBL memperkirakan probabilitas
   senyawa tersebut aktif terhadap organisme target.
4. Hasil beserta sifat fisikokimianya disimpan di basis data lokal.

Seluruh proses prediksi berjalan **offline** di komputer ini — tidak ada data yang dikirim
ke layanan eksternal dan tidak diperlukan API key apa pun.
""")

    st.divider()
    st.subheader("Keterbatasan & Disclaimer")
    st.warning("""
- Model ini adalah **alat bantu skrining awal (in silico)**, bukan pengganti uji in vitro/in vivo. Hasil prediksi wajib diverifikasi secara eksperimental sebelum ditindaklanjuti.
- Data ChEMBL memiliki noise alami (variasi metode assay antar-laboratorium), dan ambang "aktif" (MIC ≤ 10 µM) adalah konvensi umum literatur QSAR, bukan kebenaran mutlak.
- Model hanya andal untuk senyawa yang secara struktural mirip dengan data training — periksa menu **SAR Insight** untuk menilai applicability domain.
- Ukuran dataset terbatas (ribuan, bukan jutaan senyawa). Performa model dilaporkan apa adanya di menu **Dashboard**.
""")
