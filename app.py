"""
🛍️ Online Retail Customer Intelligence Dashboard
--------------------------------------------------
A fun, colorful, interactive Streamlit dashboard covering:
  - Exploratory Data Analysis (EDA)
  - Pareto (80/20) Revenue Analysis
  - RFM (Recency, Frequency, Monetary) Analysis
  - Customer Segmentation (K-Means, Hierarchical, DBSCAN)
  - Live Segment Predictor
  - Business Recommendations

Run with:  streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Retail Customer Intelligence 🛍️",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — colorful / fun / attractive theme
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Animated gradient header */
    .main-header {
        padding: 2rem 2.5rem;
        border-radius: 20px;
        background: linear-gradient(-45deg, #ff6a88, #ff9a8b, #6a82fb, #fc5c7d);
        background-size: 300% 300%;
        animation: gradientShift 10s ease infinite;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.25);
    }
    .main-header p {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.95;
    }

    /* KPI cards */
    .kpi-card {
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        transition: transform 0.25s ease;
    }
    .kpi-card:hover { transform: translateY(-6px) scale(1.02); }
    .kpi-value { font-size: 1.9rem; font-weight: 800; margin: 0; }
    .kpi-label { font-size: 0.9rem; font-weight: 600; opacity: 0.9; margin: 0; }

    .card1 { background: linear-gradient(135deg, #667eea, #764ba2); }
    .card2 { background: linear-gradient(135deg, #f093fb, #f5576c); }
    .card3 { background: linear-gradient(135deg, #4facfe, #00f2fe); }
    .card4 { background: linear-gradient(135deg, #43e97b, #38f9d7); }
    .card5 { background: linear-gradient(135deg, #fa709a, #fee140); }

    /* Section title */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d2d54;
        border-left: 8px solid #fc5c7d;
        padding-left: 12px;
        margin: 1.4rem 0 0.8rem 0;
    }

    /* Persona result card */
    .persona-card {
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff6a88, #6a82fb);
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d2d54, #1a1a2e);
    }
    section[data-testid="stSidebar"] * { color: #f1f1f1 !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #fc5c7d, #6a82fb);
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Bold

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🛍️ Online Retail Customer Intelligence Dashboard</h1>
        <p>EDA • Pareto Analysis • RFM • Customer Segmentation • Live Predictor 🎯✨</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR — data loading
# ============================================================
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Upload the retail CSV, or use the default file if present.")

uploaded_file = st.sidebar.file_uploader("📂 Upload `online_retail_II.csv`", type=["csv"])
default_path = "online_retail_II.csv"

country_default = "United Kingdom"


@st.cache_data(show_spinner="📦 Loading & cleaning transaction data...")
def load_raw(file_bytes_or_path):
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(file_bytes_or_path))
    else:
        df = pd.read_csv(file_bytes_or_path)
    return df


@st.cache_data(show_spinner="🧹 Cleaning & filtering dataset...")
def clean_data(df_raw, country):
    df = df_raw.copy()
    df = df[df["Country"] == country]
    df = df.dropna(subset=["Customer ID"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


data_source = None
if uploaded_file is not None:
    data_source = uploaded_file.getvalue()
else:
    try:
        with open(default_path, "rb") as f:
            data_source = f.read()
        st.sidebar.success("✅ Using default `online_retail_II.csv` found alongside app.py")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ No default file found — please upload the CSV to continue.")

if data_source is None:
    st.info("👈 Please upload **online_retail_II.csv** from the sidebar to light up the dashboard!")
    st.stop()

df_raw = load_raw(data_source)

countries = sorted(df_raw["Country"].dropna().unique().tolist())
country = st.sidebar.selectbox(
    "🌍 Country filter",
    countries,
    index=countries.index(country_default) if country_default in countries else 0,
)

df = clean_data(df_raw, country)

st.sidebar.markdown("---")
st.sidebar.metric("Rows after cleaning", f"{len(df):,}")
st.sidebar.metric("Unique customers", f"{df['Customer ID'].nunique():,}")
st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ using Streamlit + Plotly")

if st.sidebar.button("🎉 Celebrate!"):
    st.balloons()

# ============================================================
# TABS
# ============================================================
tab_overview, tab_eda, tab_pareto, tab_rfm, tab_cluster, tab_predict, tab_biz = st.tabs(
    [
        "🏠 Overview",
        "📊 EDA",
        "📈 Pareto 80/20",
        "💰 RFM Analysis",
        "🎯 Segmentation",
        "🔮 Predict Segment",
        "💡 Business Actions",
    ]
)

# ------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-title">📌 Key Metrics</div>', unsafe_allow_html=True)

    total_revenue = df["TotalPrice"].sum()
    total_orders = df["Invoice"].nunique()
    unique_customers = df["Customer ID"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0
    total_items = df["Quantity"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, "card1", f"£{total_revenue:,.0f}", "💷 Total Revenue"),
        (c2, "card2", f"{total_orders:,}", "🧾 Total Orders"),
        (c3, "card3", f"{unique_customers:,}", "👥 Unique Customers"),
        (c4, "card4", f"£{avg_order_value:,.1f}", "🛒 Avg Order Value"),
        (c5, "card5", f"{total_items:,.0f}", "📦 Items Sold"),
    ]
    for col, card_class, value, label in kpi_data:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card {card_class}">
                    <p class="kpi-value">{value}</p>
                    <p class="kpi-label">{label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">🌍 Top Countries (Full Dataset)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        top10 = df_raw["Country"].value_counts().head(10).reset_index()
        top10.columns = ["Country", "Orders"]
        fig = px.bar(
            top10.sort_values("Orders"),
            x="Orders",
            y="Country",
            orientation="h",
            color="Orders",
            color_continuous_scale="Sunset",
            template=PLOTLY_TEMPLATE,
            title="Top 10 Countries by Order Count",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig_pie = px.pie(
            top10,
            names="Country",
            values="Orders",
            hole=0.5,
            color_discrete_sequence=COLOR_SEQ,
            title="Share of Top 10",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-title">🗓️ Monthly Transaction Trend</div>', unsafe_allow_html=True)
    df_original_dated = df_raw.copy()
    df_original_dated["InvoiceDate"] = pd.to_datetime(df_original_dated["InvoiceDate"])
    df_original_dated["Month"] = df_original_dated["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_orders = df_original_dated.groupby("Month").size().reset_index(name="Transactions")
    fig_month = px.line(
        monthly_orders,
        x="Month",
        y="Transactions",
        markers=True,
        template=PLOTLY_TEMPLATE,
        title="Transactions Per Month (All Countries)",
    )
    fig_month.update_traces(line_color="#fc5c7d", line_width=3, marker=dict(size=8, color="#6a82fb"))
    st.plotly_chart(fig_month, use_container_width=True)

# ------------------------------------------------------------
# TAB 2 — EDA
# ------------------------------------------------------------
with tab_eda:
    st.markdown('<div class="section-title">📊 Distribution Explorer</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.histogram(
            df, x="Quantity", nbins=50, template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#667eea"], title="Quantity Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(
            df, x="Price", nbins=50, log_x=True, template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#f5576c"], title="Unit Price (log scale)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.histogram(
            df, x="TotalPrice", nbins=50, log_x=True, template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#fa8231"], title="Total Price per Line (log scale)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">👑 Top 10 Highest-Frequency Buyers</div>', unsafe_allow_html=True)
    customer_orders = df.groupby("Customer ID")["Invoice"].nunique().reset_index()
    customer_orders.columns = ["Customer ID", "OrderCount"]
    top10_buyers = customer_orders.sort_values("OrderCount", ascending=False).head(10)
    fig = px.bar(
        top10_buyers,
        x=top10_buyers["Customer ID"].astype(str),
        y="OrderCount",
        color="OrderCount",
        color_continuous_scale="Tealgrn",
        template=PLOTLY_TEMPLATE,
        title="Top 10 Highest Frequency Buyers",
        labels={"x": "Customer ID"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">💷 Distribution of Total Spend per Customer</div>', unsafe_allow_html=True)
    customer_spend = df.groupby("Customer ID")["TotalPrice"].sum().reset_index()
    customer_spend.columns = ["Customer ID", "TotalSpend"]
    fig = px.histogram(
        customer_spend, x="TotalSpend", nbins=50, log_x=True,
        template=PLOTLY_TEMPLATE, color_discrete_sequence=["#38f9d7"],
        title="Total Spend per Customer (log scale)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TAB 3 — PARETO
# ------------------------------------------------------------
with tab_pareto:
    st.markdown('<div class="section-title">📈 Pareto (80/20) Revenue Analysis</div>', unsafe_allow_html=True)

    customer_spend = df.groupby("Customer ID")["TotalPrice"].sum().reset_index()
    customer_spend.columns = ["Customer ID", "TotalSpend"]
    customer_spend = customer_spend.sort_values("TotalSpend", ascending=False).reset_index(drop=True)
    total_revenue = customer_spend["TotalSpend"].sum()
    customer_spend["CumulativeRevenue"] = customer_spend["TotalSpend"].cumsum()
    customer_spend["RevenuePercent"] = customer_spend["CumulativeRevenue"] / total_revenue * 100
    customers_80 = customer_spend[customer_spend["RevenuePercent"] <= 80]
    percent_customers = len(customers_80) / len(customer_spend) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{len(customer_spend):,}")
    c2.metric("Customers Driving 80% Revenue", f"{len(customers_80):,}")
    c3.metric("% of Customer Base", f"{percent_customers:.2f}%")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(customer_spend) + 1)),
            y=customer_spend["RevenuePercent"],
            mode="lines",
            line=dict(color="#6a82fb", width=3),
            name="Cumulative Revenue %",
            fill="tozeroy",
            fillcolor="rgba(106,130,251,0.15)",
        )
    )
    fig.add_hline(y=80, line_dash="dash", line_color="#fc5c7d", annotation_text="80% Revenue Line")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Cumulative Revenue vs Customers (Sorted by Spend)",
        xaxis_title="Customers (sorted by spend, descending)",
        yaxis_title="Cumulative Revenue (%)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**🏆 Top 10 Spenders**")
    st.dataframe(
        customer_spend.head(10).style.background_gradient(cmap="RdPu", subset=["TotalSpend"]),
        use_container_width=True,
    )

# ------------------------------------------------------------
# RFM COMPUTATION (shared by RFM tab & Segmentation tab)
# ------------------------------------------------------------
@st.cache_data(show_spinner="🧮 Computing RFM table...")
def compute_rfm(df):
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    recency = (reference_date - df.groupby("Customer ID")["InvoiceDate"].max()).dt.days
    frequency = df.groupby("Customer ID")["Invoice"].nunique()
    monetary = df.groupby("Customer ID")["TotalPrice"].sum()

    rfm_df = pd.DataFrame(
        {
            "CustomerID": recency.index,
            "Recency": recency.values,
            "Frequency": frequency.values,
            "Monetary": monetary.values,
        }
    )

    rfm_capped = rfm_df.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        Q1 = rfm_capped[col].quantile(0.25)
        Q3 = rfm_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 3 * IQR
        rfm_capped[col] = rfm_capped[col].clip(upper=upper_limit)

    rfm_capped["Frequency_Log"] = np.log1p(rfm_capped["Frequency"])
    rfm_capped["Monetary_Log"] = np.log1p(rfm_capped["Monetary"])

    scaler = StandardScaler()
    rfm_scaled = pd.DataFrame(
        scaler.fit_transform(rfm_capped[["Recency", "Frequency_Log", "Monetary_Log"]]),
        columns=["Recency", "Frequency", "Monetary"],
    )
    return rfm_df, rfm_capped, rfm_scaled, scaler


rfm_df, rfm_capped, rfm_scaled, rfm_scaler = compute_rfm(df)

# ------------------------------------------------------------
# TAB 4 — RFM ANALYSIS
# ------------------------------------------------------------
with tab_rfm:
    st.markdown('<div class="section-title">💰 RFM Table Preview</div>', unsafe_allow_html=True)
    st.dataframe(rfm_df.head(10), use_container_width=True)

    st.markdown('<div class="section-title">📦 Outlier Capping — Before vs After (IQR method)</div>', unsafe_allow_html=True)
    metric_choice = st.radio("Choose metric", ["Recency", "Frequency", "Monetary"], horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(rfm_df, y=metric_choice, template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=["#f5576c"], title=f"{metric_choice} — Before Capping")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(rfm_capped, y=metric_choice, template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=["#43e97b"], title=f"{metric_choice} — After Capping")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🔄 Log Transformation (Frequency & Monetary)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(rfm_capped, x="Frequency", nbins=30, template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=["#4facfe"], title="Frequency — Before Log")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.histogram(rfm_capped, x="Frequency_Log", nbins=30, template=PLOTLY_TEMPLATE,
                             color_discrete_sequence=["#00f2fe"], title="Frequency — After log1p")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig = px.histogram(rfm_capped, x="Monetary", nbins=30, template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=["#fa709a"], title="Monetary — Before Log")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.histogram(rfm_capped, x="Monetary_Log", nbins=30, template=PLOTLY_TEMPLATE,
                             color_discrete_sequence=["#fee140"], title="Monetary — After log1p")
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# CLUSTERING HELPERS (cached)
# ------------------------------------------------------------
@st.cache_data(show_spinner="🤖 Running K-Means across k=2..10...")
def kmeans_scan(rfm_scaled):
    wcss, sil_scores = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(rfm_scaled)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(rfm_scaled, labels))
    return wcss, sil_scores


@st.cache_data(show_spinner="🎯 Fitting final K-Means model...")
def kmeans_final(rfm_scaled, k):
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=500, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    return labels


@st.cache_data(show_spinner="🌳 Building hierarchical clustering...")
def hierarchical_scan(rfm_scaled, n_clusters):
    linkage_matrix = linkage(rfm_scaled, method="ward")
    results = []
    for link in ["ward", "complete", "average"]:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=link)
        labels = model.fit_predict(rfm_scaled)
        results.append((link, silhouette_score(rfm_scaled, labels)))
    best_link = max(results, key=lambda x: x[1])[0]
    best_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage=best_link).fit_predict(rfm_scaled)
    return linkage_matrix, results, best_link, best_labels


@st.cache_data(show_spinner="🔍 Tuning DBSCAN parameters...")
def dbscan_tuning(rfm_scaled):
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    min_samples_values = [3, 5, 8, 10]
    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            labels = db.fit_predict(rfm_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                score = np.nan
            else:
                mask = labels != -1
                if mask.sum() < 2 or len(set(labels[mask])) < 2:
                    score = np.nan
                else:
                    score = silhouette_score(np.asarray(rfm_scaled)[mask], labels[mask])
            rows.append({"eps": eps, "min_samples": min_samples, "n_clusters": n_clusters,
                          "n_noise": int((labels == -1).sum()), "Silhouette Score": score})
    results_df = pd.DataFrame(rows)
    best_row = results_df.loc[results_df["Silhouette Score"].idxmax()]
    best_eps, best_min_samples = best_row["eps"], int(best_row["min_samples"])
    best_labels = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric="euclidean").fit_predict(rfm_scaled)
    return results_df, best_eps, best_min_samples, best_labels


# ------------------------------------------------------------
# TAB 5 — SEGMENTATION
# ------------------------------------------------------------
with tab_cluster:
    st.markdown('<div class="section-title">🔎 K-Means: Choosing Optimal k</div>', unsafe_allow_html=True)

    wcss, sil_scores = kmeans_scan(rfm_scaled)
    k_range = list(range(2, 11))
    best_k = k_range[int(np.argmax(sil_scores))]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_range, y=wcss, mode="lines+markers",
                                  line=dict(color="#667eea", width=3), marker=dict(size=9)))
        fig.update_layout(template=PLOTLY_TEMPLATE, title="Elbow Method (WCSS)",
                           xaxis_title="k", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_range, y=sil_scores, mode="lines+markers",
                                  line=dict(color="#f5576c", width=3), marker=dict(size=9)))
        fig.add_vline(x=best_k, line_dash="dash", line_color="#43e97b")
        fig.update_layout(template=PLOTLY_TEMPLATE, title=f"Silhouette Score (best k = {best_k})",
                           xaxis_title="k", yaxis_title="Silhouette Score")
        st.plotly_chart(fig, use_container_width=True)

    chosen_k = st.slider("Select number of clusters (k) for final K-Means model", 2, 10, value=4)
    kmeans_labels = kmeans_final(rfm_scaled, chosen_k)
    rfm_df["KMeans_Cluster"] = kmeans_labels

    st.markdown('<div class="section-title">🎨 K-Means Cluster Visuals</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(rfm_df, x="Recency", y="Monetary", color=rfm_df["KMeans_Cluster"].astype(str),
                          color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
                          title="Clusters: Recency vs Monetary")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter_3d(rfm_df, x="Recency", y="Frequency", z="Monetary",
                             color=rfm_df["KMeans_Cluster"].astype(str),
                             color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
                             title="3D View: Recency / Frequency / Monetary")
        st.plotly_chart(fig, use_container_width=True)

    cluster_summary = (
        rfm_df.groupby("KMeans_Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean().round(2).sort_values("Monetary", ascending=False)
    )
    st.markdown("**📋 Cluster Averages (K-Means)**")
    st.dataframe(cluster_summary.style.background_gradient(cmap="PuBuGn"), use_container_width=True)

    st.markdown('<div class="section-title">🌳 Hierarchical Clustering</div>', unsafe_allow_html=True)
    linkage_matrix, link_results, best_link, hc_labels = hierarchical_scan(rfm_scaled, chosen_k)
    rfm_df["Agg_Cluster"] = hc_labels

    link_df = pd.DataFrame(link_results, columns=["Linkage", "Silhouette Score"])
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Best linkage method:** `{best_link}`")
        st.dataframe(link_df.style.background_gradient(cmap="Oranges", subset=["Silhouette Score"]),
                     use_container_width=True)
    with col2:
        fig = px.scatter(rfm_df, x="Recency", y="Monetary", color=rfm_df["Agg_Cluster"].astype(str),
                          color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
                          title="Hierarchical Clusters: Recency vs Monetary")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🔬 DBSCAN — Density-Based Clustering</div>', unsafe_allow_html=True)
    results_df, best_eps, best_min_samples, dbscan_labels = dbscan_tuning(rfm_scaled)
    rfm_df["DBSCAN_Cluster"] = dbscan_labels

    col1, col2 = st.columns(2)
    with col1:
        heatmap_data = results_df.pivot(index="min_samples", columns="eps", values="Silhouette Score")
        fig = px.imshow(heatmap_data, text_auto=".3f", color_continuous_scale="YlGnBu",
                         template=PLOTLY_TEMPLATE, title="DBSCAN Parameter Tuning (Silhouette Score)",
                         aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"🏆 Best params → eps = **{best_eps}**, min_samples = **{best_min_samples}**")
    with col2:
        color_labels = rfm_df["DBSCAN_Cluster"].astype(str).replace("-1", "Noise")
        fig = px.scatter(rfm_df, x="Recency", y="Monetary", color=color_labels,
                          color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
                          title="DBSCAN Clusters (grey/'Noise' = outliers)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🏁 Algorithm Comparison</div>', unsafe_allow_html=True)
    mask = rfm_df["DBSCAN_Cluster"] != -1
    metrics_rows = []
    for name, labels in [("K-Means", kmeans_labels), ("Hierarchical", hc_labels)]:
        metrics_rows.append({
            "Algorithm": name,
            "Silhouette Score": round(silhouette_score(rfm_scaled, labels), 3),
            "Davies-Bouldin Index": round(davies_bouldin_score(rfm_scaled, labels), 3),
            "Calinski-Harabasz Index": round(calinski_harabasz_score(rfm_scaled, labels), 1),
            "% Noise": "0%",
        })
    if mask.sum() > 1 and len(set(dbscan_labels[mask.values])) > 1:
        metrics_rows.append({
            "Algorithm": "DBSCAN",
            "Silhouette Score": round(silhouette_score(np.asarray(rfm_scaled)[mask.values], dbscan_labels[mask.values]), 3),
            "Davies-Bouldin Index": round(davies_bouldin_score(np.asarray(rfm_scaled)[mask.values], dbscan_labels[mask.values]), 3),
            "Calinski-Harabasz Index": round(calinski_harabasz_score(np.asarray(rfm_scaled)[mask.values], dbscan_labels[mask.values]), 1),
            "% Noise": f"{(~mask).mean()*100:.2f}%",
        })
    comparison_df = pd.DataFrame(metrics_rows)
    st.dataframe(comparison_df.style.background_gradient(cmap="RdPu", subset=["Silhouette Score"]),
                 use_container_width=True)

    fig = px.bar(comparison_df, x="Algorithm", y="Silhouette Score", color="Algorithm",
                 color_discrete_sequence=COLOR_SEQ, template=PLOTLY_TEMPLATE,
                 title="Silhouette Score Comparison Across Algorithms")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TAB 6 — PREDICT SEGMENT
# ------------------------------------------------------------
with tab_predict:
    st.markdown('<div class="section-title">🔮 Live Customer Segment Predictor</div>', unsafe_allow_html=True)
    st.write("Move the sliders to describe a customer, then hit **Predict** to see which segment they fall into! 🎉")

    persona_map = {
        0: ("Champions", "🏆"),
        1: ("Loyal Customers", "💎"),
        2: ("Regular Customers", "🙂"),
        3: ("At-Risk Customers", "⚠️"),
        -1: ("VIP / Outlier Customer", "🌟"),
    }

    col1, col2, col3 = st.columns(3)
    recency_in = col1.slider("📅 Recency (days since last purchase)", 0, 400, 30)
    frequency_in = col2.slider("🔁 Frequency (number of orders)", 1, 210, 10)
    monetary_in = col3.slider("💷 Monetary (total spend, £)", 0, 50000, 2000)

    if st.button("✨ Predict Segment ✨", use_container_width=True):
        cust_frame = pd.DataFrame({
            "Recency": [recency_in],
            "Frequency_Log": [np.log1p(frequency_in)],
            "Monetary_Log": [np.log1p(monetary_in)],
        })
        cust_scaled = rfm_scaler.transform(cust_frame)

        # Nearest cluster centroid from the current K-Means model (fast & stable for single-point prediction)
        km_model = KMeans(n_clusters=chosen_k, init="k-means++", n_init=20, max_iter=500, random_state=42)
        km_model.fit(rfm_scaled)
        cluster_id = int(km_model.predict(cust_scaled)[0])

        persona, emoji = persona_map.get(cluster_id, ("Unknown Segment", "❓"))

        st.markdown(
            f"""
            <div class="persona-card">
                {emoji} Predicted Segment: <br> Cluster {cluster_id} — {persona}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.balloons()

        st.caption(
            "Note: persona labels (Champions/Loyal/Regular/At-Risk) are illustrative and map to the "
            "cluster ordering learned from this dataset — always re-check the cluster-mean table on the "
            "Segmentation tab before using labels in production."
        )

# ------------------------------------------------------------
# TAB 7 — BUSINESS ACTIONS
# ------------------------------------------------------------
with tab_biz:
    st.markdown('<div class="section-title">💡 Customer Segments & Marketing Actions</div>', unsafe_allow_html=True)

    seg_cards = [
        ("🏆", "Champions", "Frequent buyers with high spending and recent purchases.",
         "Invite to an exclusive VIP loyalty programme with early access to sales.", "#43e97b", "#38f9d7"),
        ("💎", "Loyal Customers", "Purchase regularly and contribute consistent revenue.",
         "Offer reward points, cashback, and personalized recommendations.", "#4facfe", "#00f2fe"),
        ("🙂", "Regular Customers", "Average purchase frequency and spending.",
         "Encourage higher spending via bundle offers & cross-selling.", "#667eea", "#764ba2"),
        ("⚠️", "At-Risk Customers", "Haven't purchased recently — may churn.",
         "Send a 20% discount coupon with a 7-day expiry + re-engagement email.", "#fa709a", "#fee140"),
        ("🌟", "Noise / VIP Outliers", "Unusual purchasing behaviour — often extreme high-value.",
         "Assign dedicated support & exclusive premium offers.", "#ff6a88", "#6a82fb"),
    ]

    cols = st.columns(len(seg_cards))
    for col, (emoji, name, desc, action, c1, c2) in zip(cols, seg_cards):
        with col:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {c1}, {c2});
                            border-radius: 16px; padding: 1rem; color: white;
                            min-height: 260px; box-shadow: 0 6px 16px rgba(0,0,0,0.2);">
                    <div style="font-size:2rem;">{emoji}</div>
                    <div style="font-weight:800; font-size:1.05rem; margin:0.3rem 0;">{name}</div>
                    <div style="font-size:0.85rem; opacity:0.95; margin-bottom:0.6rem;">{desc}</div>
                    <hr style="border-color: rgba(255,255,255,0.4);">
                    <div style="font-size:0.85rem; font-weight:600;">🎯 Action:</div>
                    <div style="font-size:0.82rem;">{action}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">🚀 Recommended Model for Deployment</div>', unsafe_allow_html=True)
    st.success(
        "**DBSCAN** is recommended for production deployment — unlike K-Means and Hierarchical Clustering, "
        "it automatically flags outlier/VIP customers as noise instead of forcing them into a regular cluster, "
        "giving more realistic, actionable segments for the marketing team."
    )

    st.markdown('<div class="section-title">🧩 Ideas to Improve Future Segmentation</div>', unsafe_allow_html=True)
    st.write(
        """
        - 👤 Customer demographics (age, gender, location)
        - 🛍️ Product categories purchased & browsing history
        - 📱 Device type (mobile / desktop)
        - ⭐ Ratings & reviews
        - 🎟️ Coupon / promo usage and payment method preference
        - 🔁 Return & refund history and support interactions
        - 🪪 Membership / loyalty status
        """
    )

st.markdown("---")
st.caption("🛍️ Retail Customer Intelligence Dashboard — built with Streamlit, Plotly & scikit-learn ✨")
