# Dashboard Prediksi Customer Churn — Kelompok 16

Dashboard interaktif untuk menganalisis dan memprediksi **customer churn** pada perusahaan telekomunikasi menggunakan pendekatan **Exploratory Data Analysis (EDA)** dan **Machine Learning**.

## Tentang Proyek

Customer churn adalah kondisi ketika pelanggan berhenti menggunakan layanan dan beralih ke kompetitor. Proyek ini bertujuan membangun model prediksi churn untuk membantu perusahaan menyusun strategi retensi yang lebih efektif.

**Algoritma yang digunakan:**
- **Logistic Regression** — model baseline untuk memahami hubungan linear antar variabel
- **Random Forest** — model ensemble untuk menangkap pola non-linear

## Fitur Dashboard

| Tab | Deskripsi |
|-----|-----------|
| Overview | KPI utama (total pelanggan, churn rate, rata-rata biaya) dan distribusi churn |
| Distribusi | Histogram tenure dan monthly charges |
| Faktor Churn | Churn rate berdasarkan kontrak, internet service, dan metode pembayaran |
| Layanan | Efek protektif layanan tambahan (Online Security, Tech Support, dll.) |
| Demografi | Analisis gender, senior citizen, partner, dan dependents |
| Korelasi | Heatmap korelasi dan koefisien korelasi terhadap churn |
| Model ML | Tabel perbandingan performa (Accuracy, Precision, Recall, F1-Score, AUC-ROC) |
| Confusion Matrix | Visualisasi confusion matrix untuk kedua model |
| ROC Curve | Perbandingan ROC Curve Logistic Regression vs Random Forest |
| Feature Importance | Top 10 fitur terpenting dari kedua model |
| Kesimpulan | Model terbaik, kesimpulan analisis, dan rekomendasi strategi retensi |

## Dataset

- **Sumber:** Telco Customer Churn Dataset
- **Jumlah data:** 7,032 pelanggan
- **Jumlah fitur:** 21 (gender, tenure, contract, internet service, monthly charges, dll.)
- **Target:** Churn (Yes/No)

## Teknologi

- **Python 3.10+**
- **Streamlit** — framework dashboard
- **Plotly** — visualisasi interaktif
- **Scikit-learn** — machine learning (Logistic Regression, Random Forest)
- **Pandas & NumPy** — pengolahan data

## Cara Menjalankan Secara Lokal

```bash
# Clone repository
git clone https://github.com/Dimas-es/dashboard-churn-kelompok16.git
cd dashboard-churn-kelompok16

# Install dependencies
pip install -r requirements.txt

# Jalankan dashboard
streamlit run dashboard.py
```

Dashboard akan terbuka di `http://localhost:8501`.

## Hasil Utama

| Metrik | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 79.39% | 78.32% |
| Precision | 62.43% | 61.77% |
| Recall | **56.42%** | 48.40% |
| F1-Score | 59.27% | 54.27% |
| AUC-ROC | **83.45%** | 81.13% |

**Logistic Regression** dipilih sebagai model terbaik karena memiliki Recall lebih tinggi untuk menangkap pelanggan yang akan churn.

## Temuan Utama

1. Pelanggan dengan **tenure rendah (<6 bulan)** dan biaya bulanan tinggi memiliki risiko churn tertinggi
2. **Kontrak month-to-month** memiliki churn rate 42.7%, jauh di atas kontrak tahunan
3. Layanan **Online Security** dan **Tech Support** memberikan efek protektif signifikan terhadap churn
4. **Electronic check** sebagai metode pembayaran memiliki churn rate tertinggi (45.3%)

## Kelompok 16 — Sains Data

Prediksi Customer Churn pada Perusahaan Telekomunikasi Menggunakan Random Forest dan Logistic Regression untuk Meningkatkan Retensi Pelanggan.
