import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

st.set_page_config(
    page_title="Dashboard Prediksi Customer Churn — Kelompok 16",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1b4b 0%, #172554 100%);
    border: 1px solid rgba(59,130,246,.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: .78rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-weight: 800 !important; }
div[data-testid="stTabs"] button {
    font-weight: 600 !important; font-size: .85rem !important;
}
.insight-box {
    background: linear-gradient(135deg, rgba(59,130,246,.08), rgba(139,92,246,.05));
    border: 1px solid rgba(59,130,246,.2);
    border-radius: 12px; padding: 1.2rem 1.5rem; margin: .8rem 0;
}
.insight-box h4 { color: #60a5fa; font-size: .82rem; text-transform: uppercase; letter-spacing: .04em; margin-bottom: .4rem; }
.insight-box p, .insight-box li { color: #cbd5e1; font-size: .88rem; line-height: 1.7; }
.winner-card {
    background: linear-gradient(135deg, rgba(59,130,246,.08), rgba(139,92,246,.05));
    border: 1px solid rgba(59,130,246,.25); border-radius: 14px;
    padding: 1.5rem; margin-bottom: 1rem;
}
.cm-cell {
    text-align: center; padding: .8rem; border-radius: 8px;
    font-weight: 700; font-size: 1.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Data Loading & Caching ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("telco_churn_cleaned.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    return df

@st.cache_data
def train_models(df):
    dfc = df.copy()
    dfc["Churn_bin"] = dfc["Churn"].map({"Yes": 1, "No": 0})
    dfc.drop(["customerID", "Churn"], axis=1, inplace=True)

    cat_cols = dfc.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        dfc[col] = LabelEncoder().fit_transform(dfc[col])

    X = dfc.drop("Churn_bin", axis=1)
    y = dfc["Churn_bin"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    results = {
        "lr": {
            "acc": accuracy_score(y_test, lr_pred),
            "prec": precision_score(y_test, lr_pred),
            "rec": recall_score(y_test, lr_pred),
            "f1": f1_score(y_test, lr_pred),
            "auc": roc_auc_score(y_test, lr_proba),
            "cm": confusion_matrix(y_test, lr_pred),
            "fpr": roc_curve(y_test, lr_proba)[0],
            "tpr": roc_curve(y_test, lr_proba)[1],
            "coef": np.abs(lr.coef_[0]),
        },
        "rf": {
            "acc": accuracy_score(y_test, rf_pred),
            "prec": precision_score(y_test, rf_pred),
            "rec": recall_score(y_test, rf_pred),
            "f1": f1_score(y_test, rf_pred),
            "auc": roc_auc_score(y_test, rf_proba),
            "cm": confusion_matrix(y_test, rf_pred),
            "fpr": roc_curve(y_test, rf_proba)[0],
            "tpr": roc_curve(y_test, rf_proba)[1],
            "importance": rf.feature_importances_,
        },
        "features": feature_names,
    }
    return results

df = load_data()
model = train_models(df)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Dashboard Churn")
    st.markdown("**Kelompok 16** — Sains Data")
    st.divider()
    st.markdown(
        "Prediksi Customer Churn pada Perusahaan Telekomunikasi "
        "menggunakan **Random Forest** dan **Logistic Regression**."
    )
    st.divider()
    st.markdown("##### Dataset")
    st.metric("Total Pelanggan", f"{len(df):,}")
    st.metric("Jumlah Fitur", "21")

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("## 📊 Dashboard Prediksi Customer Churn")
st.caption(
    "Analisis EDA & Machine Learning pada dataset pelanggan telekomunikasi "
    "— Kelompok 16 Sains Data"
)

# ── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Overview",
    "📈 Distribusi",
    "🔍 Faktor Churn",
    "🛡 Layanan",
    "👤 Demografi",
    "🔗 Korelasi",
    "🤖 Model ML",
    "🎯 Confusion Matrix",
    "📉 ROC Curve",
    "⭐ Feature Importance",
    "📋 Kesimpulan",
])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 0 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    total = len(df)
    churn_yes = (df["Churn"] == "Yes").sum()
    churn_no = total - churn_yes
    churn_rate = churn_yes / total * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Pelanggan", f"{total:,}")
    c2.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_yes:,} churn", delta_color="inverse")
    c3.metric("Retained", f"{100 - churn_rate:.1f}%", delta=f"{churn_no:,} bertahan")
    c4.metric("Avg Monthly ($)", f"{df['MonthlyCharges'].mean():.2f}")
    c5.metric("Avg Tenure", f"{df['tenure'].mean():.1f} bln")
    c6.metric("Avg Total ($)", f"{df['TotalCharges'].mean():,.0f}")

    st.markdown("")
    col_left, col_right = st.columns(2)

    with col_left:
        fig_pie = px.pie(
            names=["Retained", "Churn"],
            values=[churn_no, churn_yes],
            color_discrete_sequence=["#3b82f6", "#ef4444"],
            hole=0.6,
        )
        fig_pie.update_layout(
            title="Distribusi Churn",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            legend=dict(orientation="h", y=-0.1),
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        mc_churn = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].mean()
        mc_no = df.loc[df["Churn"] == "No", "MonthlyCharges"].mean()
        t_churn = df.loc[df["Churn"] == "Yes", "tenure"].mean()
        t_no = df.loc[df["Churn"] == "No", "tenure"].mean()

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Churn", x=["Monthly Charges ($)", "Tenure (bulan)"],
            y=[mc_churn, t_churn], marker_color="#ef4444",
            text=[f"{mc_churn:.1f}", f"{t_churn:.1f}"], textposition="outside",
        ))
        fig_comp.add_trace(go.Bar(
            name="Retained", x=["Monthly Charges ($)", "Tenure (bulan)"],
            y=[mc_no, t_no], marker_color="#3b82f6",
            text=[f"{mc_no:.1f}", f"{t_no:.1f}"], textposition="outside",
        ))
        fig_comp.update_layout(
            title="Perbandingan Rata-rata: Churn vs Retained",
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, yaxis_title="Nilai",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DISTRIBUSI
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_t = px.histogram(
            df, x="tenure", color="Churn",
            color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"},
            nbins=12, barmode="stack",
        )
        fig_t.update_layout(
            title="Distribusi Tenure (Bulan)", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=400, xaxis_title="Tenure (bulan)", yaxis_title="Jumlah Pelanggan",
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with col_b:
        fig_m = px.histogram(
            df, x="MonthlyCharges", color="Churn",
            color_discrete_map={"No": "#8b5cf6", "Yes": "#ec4899"},
            nbins=12, barmode="stack",
        )
        fig_m.update_layout(
            title="Distribusi Monthly Charges", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=400, xaxis_title="Monthly Charges ($)", yaxis_title="Jumlah Pelanggan",
        )
        st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <h4>Temuan Distribusi</h4>
        <p>Distribusi tenure menunjukkan bahwa pelanggan baru (0–5 bulan) dan pelanggan loyal (66–71 bulan)
        mendominasi dataset. Pada MonthlyCharges, terdapat konsentrasi tinggi pada rentang $20–30 dan $70–100,
        menunjukkan adanya dua segmen layanan utama.</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FAKTOR CHURN
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    def churn_rate_by(col):
        grouped = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        grouped.columns = [col, "Churn Rate (%)"]
        return grouped

    col1, col2 = st.columns(2)
    with col1:
        cr_contract = churn_rate_by("Contract")
        fig_c = px.bar(
            cr_contract, y="Contract", x="Churn Rate (%)", orientation="h",
            color="Churn Rate (%)", color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            text=cr_contract["Churn Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_c.update_layout(
            title="Churn Rate per Tipe Kontrak", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, showlegend=False, coloraxis_showscale=False,
            xaxis=dict(range=[0, 50]),
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with col2:
        cr_inet = churn_rate_by("InternetService")
        fig_i = px.bar(
            cr_inet, y="InternetService", x="Churn Rate (%)", orientation="h",
            color="Churn Rate (%)", color_continuous_scale=["#10b981", "#06b6d4", "#ef4444"],
            text=cr_inet["Churn Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_i.update_layout(
            title="Churn Rate per Internet Service", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, showlegend=False, coloraxis_showscale=False,
            xaxis=dict(range=[0, 50]),
        )
        st.plotly_chart(fig_i, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        cr_pay = churn_rate_by("PaymentMethod").sort_values("Churn Rate (%)", ascending=True)
        fig_p = px.bar(
            cr_pay, y="PaymentMethod", x="Churn Rate (%)", orientation="h",
            color="Churn Rate (%)", color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            text=cr_pay["Churn Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_p.update_layout(
            title="Churn Rate per Metode Pembayaran", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, showlegend=False, coloraxis_showscale=False,
            xaxis=dict(range=[0, 50]),
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with col4:
        bins = list(range(0, 78, 6))
        labels = [f"{b}-{b+5}" for b in bins[:-1]]
        dft = df.copy()
        dft["tenure_bin"] = pd.cut(dft["tenure"], bins=bins, labels=labels, right=False)
        cr_tenure = dft.groupby("tenure_bin", observed=False)["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        cr_tenure.columns = ["Tenure", "Churn Rate (%)"]

        fig_tl = px.line(
            cr_tenure, x="Tenure", y="Churn Rate (%)",
            markers=True, color_discrete_sequence=["#ef4444"],
        )
        fig_tl.update_layout(
            title="Tenure vs Churn Rate", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=350, yaxis=dict(range=[0, 60]),
        )
        fig_tl.update_traces(fill="tozeroy", fillcolor="rgba(239,68,68,.1)")
        st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <h4>Temuan Kunci</h4>
        <ul>
            <li><strong>Kontrak Month-to-month</strong> memiliki churn rate tertinggi (~42.7%), jauh di atas One year (~11.3%) dan Two year (~2.8%).</li>
            <li><strong>Fiber optic</strong> menunjukkan churn rate ~41.9%, hampir 2x lipat dibanding DSL (~19.0%).</li>
            <li><strong>Electronic check</strong> sebagai metode pembayaran memiliki churn rate tertinggi (~45.3%).</li>
            <li>Pelanggan dengan tenure rendah (&lt;6 bulan) memiliki risiko churn paling tinggi.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — LAYANAN
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    services = ["OnlineSecurity", "TechSupport", "OnlineBackup",
                "DeviceProtection", "StreamingTV", "StreamingMovies"]

    svc_data = []
    for svc in services:
        yes_sub = df[df[svc] == "Yes"]
        no_sub = df[df[svc] == "No"]
        yes_rate = (yes_sub["Churn"] == "Yes").mean() * 100 if len(yes_sub) > 0 else 0
        no_rate = (no_sub["Churn"] == "Yes").mean() * 100 if len(no_sub) > 0 else 0
        svc_data.append({"Layanan": svc, "Dengan Layanan": yes_rate, "Tanpa Layanan": no_rate})

    svc_df = pd.DataFrame(svc_data)

    col_l, col_r = st.columns(2)
    with col_l:
        fig_svc = go.Figure()
        fig_svc.add_trace(go.Bar(
            name="Dengan Layanan", y=svc_df["Layanan"], x=svc_df["Dengan Layanan"],
            orientation="h", marker_color="#10b981",
            text=svc_df["Dengan Layanan"].apply(lambda x: f"{x:.1f}%"), textposition="outside",
        ))
        fig_svc.add_trace(go.Bar(
            name="Tanpa Layanan", y=svc_df["Layanan"], x=svc_df["Tanpa Layanan"],
            orientation="h", marker_color="#f87171",
            text=svc_df["Tanpa Layanan"].apply(lambda x: f"{x:.1f}%"), textposition="outside",
        ))
        fig_svc.update_layout(
            title="Churn Rate per Layanan Tambahan", barmode="group",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=450,
            xaxis_title="Churn Rate (%)", xaxis=dict(range=[0, 55]),
        )
        st.plotly_chart(fig_svc, use_container_width=True)

    with col_r:
        st.markdown("#### Efek Protektif Layanan")
        st.markdown("")
        for _, row in svc_df.iterrows():
            diff = row["Tanpa Layanan"] - row["Dengan Layanan"]
            col_a2, col_b2, col_c2 = st.columns([3, 1, 1])
            col_a2.markdown(f"**{row['Layanan']}**")
            col_b2.markdown(f"✅ {row['Dengan Layanan']:.1f}%")
            col_c2.markdown(f"❌ {row['Tanpa Layanan']:.1f}%")
            st.progress(min(diff / 30, 1.0))

    st.markdown("""
    <div class="insight-box">
        <h4>Temuan Layanan</h4>
        <p>Layanan <strong>Online Security</strong> dan <strong>Tech Support</strong> memberikan efek protektif
        terbesar terhadap churn. Pelanggan yang berlangganan layanan ini memiliki churn rate ~15%, dibandingkan
        ~42% tanpa layanan tersebut. Dukungan teknis dan keamanan menjadi faktor retensi yang sangat signifikan.</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DEMOGRAFI
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    col1, col2, col3 = st.columns(3)

    with col1:
        gender_df = df.groupby(["gender", "Churn"]).size().reset_index(name="Count")
        fig_g = px.bar(
            gender_df, x="gender", y="Count", color="Churn", barmode="stack",
            color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"},
        )
        fig_g.update_layout(
            title="Gender", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380,
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with col2:
        df_sr = df.copy()
        df_sr["Senior"] = df_sr["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior Citizen"})
        cr_sr = df_sr.groupby("Senior")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
        cr_sr.columns = ["Senior", "Churn Rate (%)"]
        fig_sr = px.bar(
            cr_sr, x="Senior", y="Churn Rate (%)",
            color="Churn Rate (%)", color_continuous_scale=["#3b82f6", "#ef4444"],
            text=cr_sr["Churn Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_sr.update_layout(
            title="Senior Citizen", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, showlegend=False, coloraxis_showscale=False,
            yaxis=dict(range=[0, 50]),
        )
        st.plotly_chart(fig_sr, use_container_width=True)

    with col3:
        items = []
        for feat in ["Partner", "Dependents"]:
            for val in ["Yes", "No"]:
                sub = df[df[feat] == val]
                rate = (sub["Churn"] == "Yes").mean() * 100
                label = f"{'Punya' if val == 'Yes' else 'Tanpa'} {feat}"
                items.append({"Kategori": label, "Churn Rate (%)": rate})
        pd_df = pd.DataFrame(items)
        fig_pd = px.bar(
            pd_df, x="Kategori", y="Churn Rate (%)",
            color="Churn Rate (%)", color_continuous_scale=["#10b981", "#ef4444"],
            text=pd_df["Churn Rate (%)"].apply(lambda x: f"{x:.1f}%"),
        )
        fig_pd.update_layout(
            title="Partner & Dependents", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, showlegend=False, coloraxis_showscale=False,
            yaxis=dict(range=[0, 40]),
        )
        st.plotly_chart(fig_pd, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <h4>Temuan Demografi</h4>
        <ul>
            <li><strong>Gender</strong> tidak berpengaruh signifikan — churn rate hampir sama (Male ~26.2%, Female ~27.0%).</li>
            <li><strong>Senior Citizen</strong> memiliki churn rate jauh lebih tinggi (~41.7%) dibanding non-senior (~23.7%).</li>
            <li>Pelanggan <strong>tanpa Partner</strong> (~33.0%) dan <strong>tanpa Dependents</strong> (~31.3%) lebih cenderung churn.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — KORELASI
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    dfc = df.copy()
    dfc["Churn_num"] = dfc["Churn"].map({"Yes": 1, "No": 0})
    numeric_df = dfc.select_dtypes(include=["int64", "float64", "int32", "float32"])
    corr = numeric_df.corr()

    fig_hm = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        aspect="auto", zmin=-1, zmax=1,
    )
    fig_hm.update_layout(
        title="Heatmap Korelasi Semua Fitur Numerik",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=500,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    churn_corr = corr["Churn_num"].drop("Churn_num").sort_values()
    fig_cc = px.bar(
        x=churn_corr.values, y=churn_corr.index, orientation="h",
        color=churn_corr.values, color_continuous_scale=["#10b981", "#94a3b8", "#ef4444"],
        labels={"x": "Koefisien Korelasi", "y": "Fitur"},
    )
    fig_cc.update_layout(
        title="Korelasi Fitur terhadap Churn",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=400,
        showlegend=False, coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cc, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tenure", "-0.35", delta="Protektif", delta_color="normal")
    c2.metric("TotalCharges", "-0.20", delta="Protektif", delta_color="normal")
    c3.metric("MonthlyCharges", "+0.19", delta="Risiko", delta_color="inverse")
    c4.metric("SeniorCitizen", "+0.15", delta="Risiko", delta_color="inverse")

    st.markdown("""
    <div class="insight-box">
        <h4>Interpretasi Korelasi</h4>
        <p><strong>Tenure</strong> memiliki korelasi negatif terkuat (-0.35) terhadap churn — semakin lama pelanggan
        berlangganan, semakin kecil kemungkinan churn. <strong>MonthlyCharges</strong> dan <strong>SeniorCitizen</strong>
        berkorelasi positif, menunjukkan bahwa biaya bulanan tinggi dan status senior meningkatkan risiko churn.</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — MODEL ML (Performa)
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### Tabel Perbandingan Performa")

    metrics_df = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        "Logistic Regression": [
            f"{model['lr']['acc']:.4f}",
            f"{model['lr']['prec']:.4f}",
            f"{model['lr']['rec']:.4f}",
            f"{model['lr']['f1']:.4f}",
            f"{model['lr']['auc']:.4f}",
        ],
        "Random Forest": [
            f"{model['rf']['acc']:.4f}",
            f"{model['rf']['prec']:.4f}",
            f"{model['rf']['rec']:.4f}",
            f"{model['rf']['f1']:.4f}",
            f"{model['rf']['auc']:.4f}",
        ],
    })

    lr_vals = [model["lr"]["acc"], model["lr"]["prec"], model["lr"]["rec"], model["lr"]["f1"], model["lr"]["auc"]]
    rf_vals = [model["rf"]["acc"], model["rf"]["prec"], model["rf"]["rec"], model["rf"]["f1"], model["rf"]["auc"]]
    best = ["✅ LR" if l > r else "✅ RF" for l, r in zip(lr_vals, rf_vals)]
    metrics_df["Terbaik"] = best

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)
    with col_l:
        labels_r = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[v * 100 for v in lr_vals] + [lr_vals[0] * 100],
            theta=labels_r + [labels_r[0]],
            name="Logistic Regression", fill="toself",
            line_color="#3b82f6", fillcolor="rgba(59,130,246,.15)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[v * 100 for v in rf_vals] + [rf_vals[0] * 100],
            theta=labels_r + [labels_r[0]],
            name="Random Forest", fill="toself",
            line_color="#10b981", fillcolor="rgba(16,185,129,.15)",
        ))
        fig_radar.update_layout(
            title="Perbandingan Metrik (Radar)",
            polar=dict(radialaxis=dict(range=[40, 90], showticklabels=True, color="#64748b")),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=420,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r:
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            name="Logistic Regression", x=labels_r, y=[v * 100 for v in lr_vals],
            marker_color="#3b82f6",
            text=[f"{v*100:.1f}%" for v in lr_vals], textposition="outside",
        ))
        fig_bars.add_trace(go.Bar(
            name="Random Forest", x=labels_r, y=[v * 100 for v in rf_vals],
            marker_color="#10b981",
            text=[f"{v*100:.1f}%" for v in rf_vals], textposition="outside",
        ))
        fig_bars.update_layout(
            title="Perbandingan Metrik (Bar)", barmode="group",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=420,
            yaxis=dict(range=[40, 92], title="Score (%)"),
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        <h4>Analisis Performa</h4>
        <p><strong>Logistic Regression</strong> mengungguli Random Forest pada semua metrik evaluasi.
        Dengan <strong>Recall {model['lr']['rec']*100:.2f}%</strong> (vs {model['rf']['rec']*100:.2f}%),
        Logistic Regression lebih mampu menangkap pelanggan yang benar-benar akan churn — metrik yang paling
        krusial untuk strategi retensi. AUC-ROC sebesar {model['lr']['auc']:.4f} menunjukkan kemampuan
        diskriminasi yang baik.</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 — CONFUSION MATRIX
# ════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    col_l, col_r = st.columns(2)

    for col, name, key, accent in [
        (col_l, "Logistic Regression", "lr", "#3b82f6"),
        (col_r, "Random Forest", "rf", "#10b981"),
    ]:
        with col:
            cm = model[key]["cm"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=[[cm[0, 0], cm[0, 1]], [cm[1, 0], cm[1, 1]]],
                x=["Predicted No", "Predicted Yes"],
                y=["Actual No", "Actual Yes"],
                text=[[f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"],
                       [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]],
                texttemplate="%{text}", textfont=dict(size=16, color="white"),
                colorscale=[[0, "#1e293b"], [0.5, accent], [1, accent]],
                showscale=False,
            ))
            fig_cm.update_layout(
                title=f"Confusion Matrix — {name}",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=380,
                xaxis_title="Predicted", yaxis_title="Actual",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("TP", cm[1, 1])
            mc2.metric("FN", cm[1, 0])
            mc3.metric("FP", cm[0, 1])
            mc4.metric("TN", cm[0, 0])

    st.markdown(f"""
    <div class="insight-box">
        <h4>Interpretasi Confusion Matrix</h4>
        <ul>
            <li><strong>Logistic Regression</strong> mendeteksi <strong>{model['lr']['cm'][1,1]} pelanggan churn</strong> (TP),
            sementara Random Forest hanya {model['rf']['cm'][1,1]}.</li>
            <li>LR memiliki <strong>False Negative lebih rendah ({model['lr']['cm'][1,0]} vs {model['rf']['cm'][1,0]})</strong>
            — lebih sedikit pelanggan churn yang terlewat.</li>
            <li>Untuk strategi retensi, <strong>meminimalkan False Negative lebih penting</strong> karena pelanggan
            churn yang tidak terdeteksi berarti kehilangan pendapatan.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 8 — ROC CURVE
# ════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=model["lr"]["fpr"], y=model["lr"]["tpr"],
        mode="lines", name=f"Logistic Regression (AUC={model['lr']['auc']:.4f})",
        line=dict(color="#3b82f6", width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,.08)",
    ))
    fig_roc.add_trace(go.Scatter(
        x=model["rf"]["fpr"], y=model["rf"]["tpr"],
        mode="lines", name=f"Random Forest (AUC={model['rf']['auc']:.4f})",
        line=dict(color="#10b981", width=2.5),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC=0.5)",
        line=dict(color="#64748b", width=1.5, dash="dash"),
    ))
    fig_roc.update_layout(
        title="Perbandingan ROC Curve — Logistic Regression vs Random Forest",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=500,
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Logistic Regression AUC", f"{model['lr']['auc']:.4f}", delta="Terbaik")
    c2.metric("Random Forest AUC", f"{model['rf']['auc']:.4f}")

    st.markdown("""
    <div class="insight-box">
        <h4>Interpretasi ROC Curve</h4>
        <p>Kedua model menunjukkan performa baik dengan kurva ROC jauh di atas garis diagonal.
        <strong>Logistic Regression</strong> memiliki AUC lebih tinggi, menunjukkan kemampuan yang lebih baik
        dalam membedakan pelanggan churn dan non-churn di berbagai threshold klasifikasi.</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 9 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    features = model["features"]

    col_l, col_r = st.columns(2)
    with col_l:
        rf_imp = pd.DataFrame({
            "Feature": features,
            "Importance": model["rf"]["importance"],
        }).sort_values("Importance", ascending=True).tail(10)

        fig_rf = px.bar(
            rf_imp, y="Feature", x="Importance", orientation="h",
            color="Importance", color_continuous_scale=["#64748b", "#3b82f6", "#8b5cf6"],
        )
        fig_rf.update_layout(
            title="Random Forest — Feature Importance (Top 10)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=450,
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_rf, use_container_width=True)

    with col_r:
        lr_coef = pd.DataFrame({
            "Feature": features,
            "|Coefficient|": model["lr"]["coef"],
        }).sort_values("|Coefficient|", ascending=True).tail(10)

        fig_lr = px.bar(
            lr_coef, y="Feature", x="|Coefficient|", orientation="h",
            color="|Coefficient|", color_continuous_scale=["#64748b", "#3b82f6", "#8b5cf6"],
        )
        fig_lr.update_layout(
            title="Logistic Regression — |Koefisien| (Top 10)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=450,
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_lr, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <h4>Temuan Feature Importance</h4>
        <ul>
            <li><strong>TotalCharges, MonthlyCharges, dan Tenure</strong> menjadi 3 fitur terpenting pada Random Forest.</li>
            <li>Pada Logistic Regression, <strong>Tenure</strong> memiliki pengaruh terbesar, diikuti MonthlyCharges dan Contract.</li>
            <li><strong>Contract</strong> konsisten berpengaruh tinggi di kedua model — kontrak jangka pendek sangat meningkatkan risiko churn.</li>
            <li>Layanan <strong>OnlineSecurity dan TechSupport</strong> juga muncul di top features, mengonfirmasi efek protektif dari analisis EDA.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 10 — KESIMPULAN
# ════════════════════════════════════════════════════════════════════════════
with tabs[10]:
    # Model terbaik highlight
    st.markdown(f"""
    <div class="winner-card">
        <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
            <div style="font-size:2.5rem;">🏆</div>
            <div style="flex:1;min-width:200px;">
                <div style="font-size:.75rem;color:#94a3b8;text-transform:uppercase;font-weight:600;">Model Terbaik</div>
                <div style="font-size:1.6rem;font-weight:800;color:#60a5fa;">Logistic Regression</div>
                <div style="font-size:.88rem;color:#cbd5e1;margin-top:.3rem;">
                    Dipilih karena memiliki <strong style="color:#f1f5f9;">Recall lebih tinggi
                    ({model['lr']['rec']*100:.2f}% vs {model['rf']['rec']*100:.2f}%)</strong> untuk menangkap
                    pelanggan yang akan churn, serta <strong style="color:#f1f5f9;">AUC-ROC terbaik
                    ({model['lr']['auc']:.4f})</strong>.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{model['lr']['acc']*100:.1f}%")
    c2.metric("F1-Score", f"{model['lr']['f1']*100:.1f}%")
    c3.metric("AUC-ROC", f"{model['lr']['auc']*100:.1f}%")

    st.divider()

    col_k, col_r = st.columns(2)
    with col_k:
        st.markdown("#### 📌 Kesimpulan")
        st.markdown("""
        1. **Logistic Regression** dipilih sebagai model terbaik karena mengungguli Random Forest pada semua metrik, terutama Recall.
        2. Pelanggan dengan **tenure rendah (<6 bulan)** dan **biaya bulanan tinggi** memiliki risiko churn tertinggi.
        3. **Kontrak month-to-month**, pembayaran **electronic check**, dan layanan **fiber optic** adalah indikator utama churn.
        4. Fitur **Tenure, MonthlyCharges, dan Contract** menjadi prediktor terpenting berdasarkan feature importance kedua model.
        5. Layanan **Online Security** dan **Tech Support** berperan sebagai faktor protektif signifikan.
        """)

    with col_r:
        st.markdown("#### 🎯 Rekomendasi Strategi Retensi")
        st.markdown("""
        - ▸ Berikan promo upgrade ke kontrak jangka panjang (1–2 tahun) pada pelanggan month-to-month baru.
        - ▸ Tawarkan bundling **Online Security & Tech Support** secara gratis di 6 bulan pertama.
        - ▸ Incentivize perpindahan dari **electronic check** ke automatic payment.
        - ▸ Tingkatkan kualitas layanan **Fiber Optic** dan buat program loyalty khusus senior citizen.
        - ▸ Implementasikan model Logistic Regression untuk **early warning system** — identifikasi pelanggan berisiko tinggi secara real-time.
        """)
