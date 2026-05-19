import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="DiaPredict AI",
    page_icon="🩺",
    layout="wide"
)

model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")
AKURASI = joblib.load("accuracy.pkl")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 45%, #fdf2f8 100%);
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1180px;
}

.hero {
    padding: 45px;
    border-radius: 32px;
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    color: white;
    box-shadow: 0 18px 45px rgba(30, 64, 175, 0.28);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 52px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    opacity: 0.95;
}

.card {
    background: rgba(255,255,255,0.92);
    padding: 28px;
    border-radius: 26px;
    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

.small-card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.07);
    border: 1px solid #e5e7eb;
}

.num {
    font-size: 36px;
    font-weight: 900;
    color: #1d4ed8;
}

.label {
    color: #64748b;
    font-size: 15px;
}

.badge {
    display: inline-block;
    padding: 9px 16px;
    border-radius: 999px;
    background: #dcfce7;
    color: #166534;
    font-weight: 800;
    margin-bottom: 14px;
}

.warning-badge {
    display: inline-block;
    padding: 9px 16px;
    border-radius: 999px;
    background: #fee2e2;
    color: #991b1b;
    font-weight: 800;
    margin-bottom: 14px;
}

div.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 16px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-weight: 800;
    border: none;
    font-size: 17px;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    color: white;
}

.result-good {
    padding: 28px;
    border-radius: 24px;
    background: linear-gradient(135deg, #dcfce7, #f0fdf4);
    color: #166534;
    font-size: 26px;
    font-weight: 900;
    text-align: center;
    border: 1px solid #86efac;
}

.result-bad {
    padding: 28px;
    border-radius: 24px;
    background: linear-gradient(135deg, #fee2e2, #fff1f2);
    color: #991b1b;
    font-size: 26px;
    font-weight: 900;
    text-align: center;
    border: 1px solid #fca5a5;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 25px 0;">
        <h1 style="font-size:48px; margin-bottom:0;">🩺</h1>
        <h1 style="font-size:36px; font-weight:800; color:#1e293b; margin-top:0;">
            Let's go Check<br>Diabetes
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # Card informasi model
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.65);
        padding: 20px;
        border-radius: 20px;
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.5);
    ">
        <h4 style="margin-top:0;">🤖 Model Information</h4>
        <p style="font-size:16px; margin-bottom:0;">
            Model menggunakan <b>Support Vector Machine (SVM)</b>
            untuk memprediksi risiko diabetes berdasarkan
            indikator kesehatan pasien.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Card penjelasan
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.65);
        padding: 20px;
        border-radius: 20px;
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,0.5);
    ">
        <h4 style="margin-top:0;">💡 About This App</h4>
        <p style="font-size:16px; margin-bottom:0;">
            Website ini membantu melakukan prediksi awal
            risiko diabetes secara cepat, interaktif,
            dan mudah dipahami.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 Menu")

    menu = st.radio(
         "🌐 Menu Navigasi",[
            "🏠 Home",
            "📘 Guide",
            "📊 About Model",
            "🧮 BMI Calculator",
            "🩺 Prediction"
        ]
    )

if menu == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <div class="badge">Machine Learning Web App</div>
        <h1>DiaPredict AI 🩺</h1>
        <p>
        Sistem prediksi risiko diabetes berbasis Support Vector Machine.
        Website ini dibuat untuk membantu pengguna memahami potensi risiko diabetes
        berdasarkan indikator kesehatan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="small-card"><div class="num">{AKURASI:2.f}%</div><div class="label">Akurasi Model</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="small-card"><div class="num">8</div><div class="label">Variabel Input</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="small-card"><div class="num">SVM</div><div class="label">Algoritma Utama</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>✨ Tentang Aplikasi</h3>
        <p>
        Aplikasi ini menggunakan data diabetes dengan beberapa variabel kesehatan seperti
        jumlah kehamilan, glukosa, tekanan darah, insulin, BMI, riwayat diabetes keluarga,
        dan umur. Data tersebut diproses menggunakan model Machine Learning untuk menghasilkan
        prediksi awal berupa <b>berisiko diabetes</b> atau <b>tidak berisiko diabetes</b>.
        </p>
        <p>
        Website ini cocok digunakan sebagai demo proyek Machine Learning karena sudah memiliki
        input interaktif, model prediksi, tampilan hasil, dan informasi performa model.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif menu == "🩺 Prediction":
    st.markdown("""
    <div class="hero">
        <h1>Prediksi Risiko Diabetes</h1>
        <p>Masukkan data kesehatan pasien, lalu klik tombol prediksi untuk melihat hasil klasifikasi.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 Masukkan Data Pasien")

        pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=None, placeholder="Masukkan Jumlah Kehamilan")
        glucose = st.number_input("Glucose / Kadar Glukosa", min_value=0, max_value=300, value=None, placeholder="Masukkan Kadar Glukosa")
        blood_pressure = st.number_input("Blood Pressure / Tekanan Darah", min_value=0, max_value=200, value=None, placeholder="Masukkan Tekanan Darah")
        skin_thickness = st.number_input("Skin Thickness / Ketebalan Kulit", min_value=0, max_value=100, value=None, placeholder="Masukkan Ketebalan Kulit")
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=None, placeholder="Masukkan Kadar Insulin")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=None,placeholder="Masukkan nilai BMI")
        pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=None, placeholder="Masukkan Nilai Pedigree")
        age = st.number_input("Age / Umur", min_value=1, max_value=120, value=None, placeholder="Masukkan Umur")

        tombol = st.button("🔍 Prediksi Sekarang")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Hasil Prediksi")

        if tombol:
            if None in [
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, pedigree, age
            ]:
                st.warning("⚠️ Mohon isi seluruh data pasien terlebih dahulu.")
            else:
                data_baru = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                       insulin, bmi, pedigree, age]])
    
                data_scaled = scaler.transform(data_baru)
                hasil = model.predict(data_scaled)[0]
    
                probabilitas = model.predict_proba(data_scaled)[0]
                peluang_diabetes = probabilitas[1] * 100
                peluang_tidak_diabetes = probabilitas[0] * 100
                
            if hasil == 1:
                st.markdown('<div class="result-bad">⚠️ Berisiko Diabetes</div>', unsafe_allow_html=True)
                st.write("""
                Berdasarkan data yang dimasukkan, model memprediksi bahwa pasien memiliki indikasi
                risiko diabetes. Hasil ini dapat menjadi peringatan awal untuk pengguna agar pengguna lebih memperhatikan
                pola hidup dengan menjaga pola makan, rutin berolahraga, dan berkonsultasi dengan tenaga medis untuk pemeriksaan lebih lanjut.
                """)
                # Tampilkan probabilitas
                st.markdown("### 📊 Probabilitas Prediksi")
                st.write(f"**Peluang Berisiko Diabetes:** {peluang_diabetes:.2f}%")
                st.write(f"**Peluang Tidak Berisiko Diabetes:** {peluang_tidak_diabetes:.2f}%")

                # Progress bar berdasarkan peluang diabetes
                st.progress(min(max(peluang_diabetes / 100, 0), 1))
                st.markdown('<div class="warning-badge">Saran: lakukan konsultasi medis</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-good">✅ Tidak Berisiko Diabetes</div>', unsafe_allow_html=True)
                st.write("""
                Berdasarkan data yang dimasukkan, model memprediksi bahwa pasien tidak menunjukkan
                indikasi risiko diabetes berdasarkan pola data yang dipelajari. Meskipun begitu, tetap disarankan untuk
                menjaga pola hidup sehat dengan makan yang bergizi, melakukan aktivitas fisik yang bermanfaat,
                dan melakukan pemeriksaan kesehatan secara berkala.
                """)
                 # Tampilkan probabilitas
                st.markdown("### 📊 Probabilitas Prediksi")
                st.write(f"**Peluang Tidak Berisiko Diabetes:** {peluang_tidak_diabetes:.2f}%")

                # Progress bar berdasarkan peluang diabetes
                st.progress(min(max(peluang_diabetes / 100, 0), 1))
                st.markdown('<div class="badge">Saran: tetap jaga pola hidup sehat</div>', unsafe_allow_html=True)

            st.info("Catatan: Hasil prediksi ini hanya untuk tujuan edukasi dan bukan pengganti diagnosis dokter.")
        else:
            st.write("Belum ada hasil prediksi. Isi data pasien terlebih dahulu, lalu klik tombol prediksi.")
            st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=180)

        st.markdown('</div>', unsafe_allow_html=True)

elif menu == "📊 About Model":
    st.markdown("""
    <div class="hero">
        <h1>About Dataset & Model</h1>
        <p>Informasi mengenai data, metode, dan performa model yang digunakan.</p>
    </div>
    """, unsafe_allow_html=True)

    # Membuka card putih
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Dataset
    st.markdown("### 📌 Dataset")
    st.markdown("""
Dataset yang digunakan adalah dataset diabetes yang berisi indikator kesehatan pasien.

Variabel input terdiri dari:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
""")

    # Metode
    st.markdown("### 🤖 Metode")
    st.markdown("""
Algoritma yang digunakan adalah **Support Vector Machine (SVM)**.

Model ini bekerja dengan mencari batas pemisah terbaik antara kelas pasien
yang berisiko diabetes dan tidak berisiko diabetes.
""")

    # Performa Model
    st.markdown("### 📈 Performa Model")
    st.markdown(f"""
Berdasarkan pengujian pada data testing, model memperoleh akurasi sebesar
**{AKURASI:2.f}%**.

Nilai ini menunjukkan bahwa model mampu mengklasifikasikan sebagian besar
data uji dengan cukup baik.
""")

    # Tujuan
    st.markdown("### 🎯 Tujuan")
    st.markdown("""
Tujuan aplikasi ini adalah menampilkan hasil prediksi diabetes dalam bentuk
website interaktif yang mudah digunakan dan mudah dipahami oleh pengguna.
""")

    # Menutup card putih
    st.markdown('</div>', unsafe_allow_html=True)
    
elif menu == "📘 Guide":
    st.markdown("""
    <div class="hero">
        <h1>Panduan Penggunaan</h1>
        <p>Ikuti langkah berikut untuk menggunakan aplikasi prediksi diabetes.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🧭 Cara Menggunakan Aplikasi</h3>
        <ol>
            <li>Buka menu <b>Prediction</b>.</li>
            <li>Masukkan nilai pada setiap variabel kesehatan pasien.</li>
            <li>Klik tombol <b>Prediksi Sekarang</b>.</li>
            <li>Hasil prediksi akan muncul di sebelah kanan.</li>
            <li>Baca interpretasi hasil dan catatan yang diberikan.</li>
            <li>Untuk memeriksa nilai BMI, Buka Menu BMI.</li>
        </ol>

        st.markdown("### ⚠️ Catatan Penting")
        st.markdown("""
                Aplikasi ini tidak digunakan untuk menggantikan diagnosis dokter.
                Hasil prediksi hanya bersifat edukatif dan sebagai contoh penerapan
                Machine Learning dalam bidang kesehatan.
                """)
        st.markdown('</div>', unsafe_allow_html=True)

elif menu == "🧮 BMI Calculator":
    st.markdown("""
    <div class="custom-card">

    <h1>🧮 BMI Calculator</h1>

    <p class="center-text">
    Hitung BMI (Body Mass Index) Anda
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        tinggi = st.number_input(
            "Tinggi Badan (cm)",
            min_value=1.0
        )
    with col2:

        berat = st.number_input(
            "Berat Badan (kg)",
            min_value=1.0
        )
    bmi_result = berat / ((tinggi / 100) ** 2)

    st.metric(
        "Hasil BMI",
        f"{bmi_result:.2f}"
    )

    if bmi_result < 18.5:

        st.warning("Kategori: Berat badan kurang")

    elif bmi_result < 25:

        st.success("Kategori: Berat badan normal")

    elif bmi_result < 30:

        st.warning("Kategori: Kelebihan berat badan")
    else:
        st.error("Kategori: Obesitas")

    st.markdown("""
    <div class="custom-card">

    <h3>📋 Interpretasi BMI</h3>
    <ul>
    <li><b>< 18.5</b> : Berat badan kurang</li>
    <li><b>18.5 - 24.9</b> : Berat badan normal</li>
    <li><b>25 - 29.9</b> : Kelebihan berat badan</li>
    <li><b>>= 30</b> : Obesitas</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

