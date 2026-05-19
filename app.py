import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# LOAD MODEL
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")

# AKURASI MODEL KAMU
akurasi_model = 74

# CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef4ff, #f8fbff);
}

.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}

.title {
    font-size: 44px;
    font-weight: 800;
    color: #1e3a8a;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    color: #64748b;
    text-align: center;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(135deg, #dbeafe, #eff6ff);
    padding: 24px;
    border-radius: 22px;
    text-align: center;
    border: 1px solid #bfdbfe;
}

.metric-number {
    font-size: 38px;
    font-weight: 800;
    color: #1d4ed8;
}

.metric-label {
    color: #475569;
    font-size: 15px;
}

div.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 15px;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: 700;
    border: none;
}

.result-positive {
    background: #fee2e2;
    color: #991b1b;
    padding: 25px;
    border-radius: 20px;
    font-size: 24px;
    font-weight: 800;
    text-align: center;
}

.result-negative {
    background: #dcfce7;
    color: #166534;
    padding: 25px;
    border-radius: 20px;
    font-size: 24px;
    font-weight: 800;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "🩺 Prediction", "📊 About Model"]
)

# ================= HOME =================
if menu == "🏠 Home":
    st.markdown('<div class="title">🩺 Diabetes Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Aplikasi prediksi risiko diabetes berbasis Machine Learning</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Selamat Datang 👋</h3>
        <p>
        Website ini dibuat untuk membantu melakukan prediksi awal terhadap risiko diabetes 
        berdasarkan beberapa indikator kesehatan seperti kadar glukosa, tekanan darah, BMI, insulin, umur, 
        dan riwayat diabetes keluarga.
        </p>
        <p>
        Sistem ini menggunakan model <b>Support Vector Machine (SVM)</b> yang telah dilatih menggunakan 
        dataset diabetes. Hasil prediksi yang diberikan berupa kategori apakah seseorang berisiko diabetes 
        atau tidak berisiko diabetes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{akurasi_model}%</div>
            <div class="metric-label">Akurasi Model</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">8</div>
            <div class="metric-label">Variabel Input</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">SVM</div>
            <div class="metric-label">Metode Machine Learning</div>
        </div>
        """, unsafe_allow_html=True)

# ================= PREDICTION =================
elif menu == "🩺 Prediction":
    st.markdown('<div class="title">Diabetes Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Masukkan data pasien pada form berikut</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Data Pasien")

        pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=25)

        tombol = st.button("🔍 Predict Diabetes Risk")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if tombol:
            data_baru = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                   insulin, bmi, pedigree, age]])

            data_scaled = scaler.transform(data_baru)
            hasil = model.predict(data_scaled)[0]

            if hasil == 1:
                st.markdown('<div class="result-positive">Berisiko Diabetes</div>', unsafe_allow_html=True)
                st.write("""
                Berdasarkan data yang dimasukkan, sistem memprediksi bahwa pasien memiliki indikasi 
                risiko diabetes. Hasil ini bukan diagnosis medis, tetapi dapat digunakan sebagai 
                gambaran awal untuk meningkatkan kewaspadaan.
                """)
            else:
                st.markdown('<div class="result-negative">Tidak Berisiko Diabetes</div>', unsafe_allow_html=True)
                st.write("""
                Berdasarkan data yang dimasukkan, sistem memprediksi bahwa pasien tidak menunjukkan 
                indikasi risiko diabetes berdasarkan pola data yang telah dipelajari model.
                """)

            st.info("Catatan: Hasil prediksi ini hanya bersifat edukatif dan bukan pengganti diagnosis dokter.")
        else:
            st.write("Silakan isi data pasien terlebih dahulu, lalu klik tombol prediksi.")

        st.markdown('</div>', unsafe_allow_html=True)

# ================= ABOUT MODEL =================
elif menu == "📊 About Model":
    st.markdown('<div class="title">About Dataset & Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Informasi data dan metode yang digunakan</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <h3>Dataset</h3>
        <p>
        Dataset yang digunakan adalah dataset diabetes yang berisi beberapa indikator kesehatan pasien.
        Variabel input terdiri dari Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI,
        Diabetes Pedigree Function, dan Age.
        </p>

        <h3>Metode</h3>
        <p>
        Model yang digunakan adalah <b>Support Vector Machine (SVM)</b>. Metode ini digunakan untuk 
        melakukan klasifikasi apakah pasien termasuk dalam kategori berisiko diabetes atau tidak.
        </p>

        <h3>Akurasi Model</h3>
        <p>
        Berdasarkan hasil pengujian pada data testing, model memperoleh akurasi sebesar 
        <b>{akurasi_model}%</b>.
        </p>

        <h3>Tujuan Aplikasi</h3>
        <p>
        Aplikasi ini dibuat sebagai sistem prediksi berbasis web yang dapat digunakan untuk menampilkan
        hasil klasifikasi secara interaktif dan mudah dipahami oleh pengguna.
        </p>
    </div>
    """, unsafe_allow_html=True)