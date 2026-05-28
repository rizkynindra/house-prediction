import streamlit as st
import joblib
import numpy as np
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="centered",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .hero-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.55);
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .section-label {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        margin-top: 1.2rem;
    }

    .result-box {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.15), rgba(96, 165, 250, 0.15));
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeInUp 0.5s ease;
    }

    .result-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .result-price {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #34d399, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .error-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        color: #fca5a5;
        font-size: 0.9rem;
        margin-top: 0.8rem;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Override Streamlit widget labels */
    label, .stSelectbox label, .stNumberInput label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* Input fields */
    input[type="number"], .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #1d4ed8);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.4);
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model", "svm_model.pkl")
    return joblib.load(model_path)


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ─── Helper ───────────────────────────────────────────────────────────────────
def format_rupiah(value: float) -> str:
    """Format angka menjadi format Rupiah Indonesia."""
    val = int(round(value))
    # Format dengan titik sebagai pemisah ribuan
    s = f"{val:,}".replace(",", ".")
    return f"Rp. {s}"


# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🏠 Prediksi Harga Rumah</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Estimasi harga properti berbasis Machine Learning (SVM)</p>', unsafe_allow_html=True)

if not model_loaded:
    st.markdown(f'<div class="error-box">❌ <strong>Gagal memuat model:</strong> {model_error}</div>', unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="main-card">', unsafe_allow_html=True)

# ── Row 1: Area & Tahun Dibangun ──────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="section-label">📐 Luas Area (m²)</p>', unsafe_allow_html=True)
    area_input = st.text_input(
        label="Area",
        value="",
        placeholder="Contoh: 150",
        key="area",
        label_visibility="collapsed",
    )

with col2:
    st.markdown('<p class="section-label">📅 Tahun Dibangun</p>', unsafe_allow_html=True)
    year_input = st.text_input(
        label="Tahun Dibangun",
        value="",
        placeholder="Contoh: 2010",
        key="year",
        label_visibility="collapsed",
    )

# ── Row 2: Kamar Tidur, Kamar Mandi, Lantai ───────────────────────────────────
col3, col4, col5 = st.columns(3)
with col3:
    st.markdown('<p class="section-label">🛏 Kamar Tidur</p>', unsafe_allow_html=True)
    bedrooms_input = st.text_input(
        label="Bedrooms",
        value="",
        placeholder="Contoh: 3",
        key="bedrooms",
        label_visibility="collapsed",
    )

with col4:
    st.markdown('<p class="section-label">🚿 Kamar Mandi</p>', unsafe_allow_html=True)
    bathrooms_input = st.text_input(
        label="Bathrooms",
        value="",
        placeholder="Contoh: 2",
        key="bathrooms",
        label_visibility="collapsed",
    )

with col5:
    st.markdown('<p class="section-label">🏗 Jumlah Lantai</p>', unsafe_allow_html=True)
    floors_input = st.text_input(
        label="Floors",
        value="",
        placeholder="Contoh: 2",
        key="floors",
        label_visibility="collapsed",
    )

# ── Row 3: Dropdown ───────────────────────────────────────────────────────────
col6, col7, col8 = st.columns(3)
with col6:
    st.markdown('<p class="section-label">📍 Lokasi</p>', unsafe_allow_html=True)
    location = st.selectbox(
        label="Location",
        options=[
            ("0 — Downtown", 0),
            ("1 — Suburban", 1),
            ("2 — Urban", 2),
            ("3 — Rural", 3),
        ],
        format_func=lambda x: x[0],
        key="location",
        label_visibility="collapsed",
    )

with col7:
    st.markdown('<p class="section-label">✨ Kondisi Rumah</p>', unsafe_allow_html=True)
    condition = st.selectbox(
        label="Condition",
        options=[
            ("0 — Excellent", 0),
            ("1 — Good", 1),
            ("2 — Fair", 2),
            ("3 — Poor", 3),
        ],
        format_func=lambda x: x[0],
        key="condition",
        label_visibility="collapsed",
    )

with col8:
    st.markdown('<p class="section-label">🚗 Garasi</p>', unsafe_allow_html=True)
    garage = st.selectbox(
        label="Garage",
        options=[
            ("0 — Tidak Ada", 0),
            ("1 — Ada", 1),
        ],
        format_func=lambda x: x[0],
        key="garage",
        label_visibility="collapsed",
    )

st.markdown("</div>", unsafe_allow_html=True)

# ─── Predict Button ───────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Prediksi Harga Rumah", use_container_width=True)

if predict_btn:
    errors = []

    # Validasi: semua input teks harus angka
    def parse_number(raw: str, label: str, is_int: bool = True):
        raw = raw.strip()
        if raw == "":
            errors.append(f"**{label}** tidak boleh kosong.")
            return None
        try:
            return int(raw) if is_int else float(raw)
        except ValueError:
            errors.append(f"**{label}** harus berupa angka (bukan teks).")
            return None

    area      = parse_number(area_input,      "Luas Area",      is_int=False)
    year      = parse_number(year_input,      "Tahun Dibangun", is_int=True)
    bedrooms  = parse_number(bedrooms_input,  "Kamar Tidur",    is_int=True)
    bathrooms = parse_number(bathrooms_input, "Kamar Mandi",    is_int=True)
    floors    = parse_number(floors_input,    "Jumlah Lantai",  is_int=True)

    # Nilai-nilai tambahan dari selectbox (sudah pasti angka)
    loc_val  = location[1]
    cond_val = condition[1]
    gar_val  = garage[1]

    if errors:
        err_html = "<br>".join([f"• {e}" for e in errors])
        st.markdown(f'<div class="error-box">⚠️ Perbaiki input berikut:<br>{err_html}</div>', unsafe_allow_html=True)
    else:
        # Susun fitur sesuai urutan: Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage
        features = np.array([[area, bedrooms, bathrooms, floors, year, loc_val, cond_val, gar_val]])

        try:
            predicted_price = model.predict(features)[0]
            harga_fmt = format_rupiah(predicted_price)

            st.markdown(f"""
            <div class="result-box">
                <p class="result-label">Estimasi Harga Rumah</p>
                <p class="result-price">{harga_fmt}</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as ex:
            st.markdown(f'<div class="error-box">❌ <strong>Error saat prediksi:</strong> {ex}</div>', unsafe_allow_html=True)
