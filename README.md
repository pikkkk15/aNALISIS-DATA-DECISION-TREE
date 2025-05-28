Analisis Data Decision Tree - HR Employee Attrition

Proyek ini bertujuan untuk menganalisis data kepegawaian menggunakan algoritma Decision Tree guna memprediksi kemungkinan karyawan mengalami attrition (berhenti bekerja). Analisis ini membantu dalam memahami faktor-faktor yang memengaruhi keputusan karyawan untuk tetap atau meninggalkan perusahaan.

Dataset

Dataset yang digunakan adalah WA_Fn-UseC_-HR-Employee-Attrition.csv, yang berisi informasi terkait karyawan, termasuk:

Umur

Jenis kelamin

Jabatan

Departemen

Tingkat pendidikan

Masa kerja

Status attrition (Ya/Tidak)

Struktur Proyek

├── analisis.py
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── decision_tree_pegawai.png
└── confusion_matrix_pegawai.png

analisis.py: Skrip Python utama untuk memuat data, melakukan preprocessing, membangun model Decision Tree, dan mengevaluasi kinerjanya.

WA_Fn-UseC_-HR-Employee-Attrition.csv: Dataset yang digunakan untuk analisis.

decision_tree_pegawai.png: Visualisasi struktur pohon keputusan yang dihasilkan.

confusion_matrix_pegawai.png: Matriks kebingungan yang menunjukkan performa model.

Cara Menjalankan

Klon repositori:

git clone https://github.com/pikkkk15/aNALISIS-DATA-DECISION-TREE.git
cd aNALISIS-DATA-DECISION-TREE

Instal dependensi:

Pastikan Anda memiliki Python 3.x dan pip terinstal. Kemudian, instal pustaka yang diperlukan:

pip install pandas scikit-learn matplotlib

Jalankan skrip analisis:

python analisis.py

Skrip akan memuat data, membangun model Decision Tree, mengevaluasi kinerjanya, dan menyimpan visualisasi hasil ke file PNG.

Hasil Analisis

Visualisasi Pohon Keputusan:



Matriks Kebingungan:



Matriks kebingungan menunjukkan jumlah prediksi yang benar dan salah yang dibuat oleh model, membantu dalam mengevaluasi akurasi dan kesalahan klasifikasi.
