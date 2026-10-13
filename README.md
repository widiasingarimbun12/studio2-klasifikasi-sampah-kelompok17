# Studio 2 — Klasifikasi Sampah

## Identitas Kelompok
- **Nama Kelompok:** Kelompok 17
- **Mata Kuliah:** Deep Learning (IF25-40401)
- **Program Studi:** Teknik Informatika, Institut Teknologi Sumatera

## Anggota Kelompok
- Nama: Widia Salsalina Br Singarimbun — NIM: 123140035
- Nama: Audy Olivya Br Gurusinga — NIM: 123140025
- Nama: Jesika Filosovi Br Perangin-angin — NIM: 123140044

## Deskripsi
Proyek ini bertujuan membangun model deep learning berbasis CNN untuk mengklasifikasikan citra sampah ke dalam lima kelas: botol plastik, gelas plastik, kertas/kardus, saset kemasan, dan styrofoam.

## Pendekatan
Model awal menggunakan ResNet-18 pretrained dan akan dievaluasi menggunakan Macro-F1 sebagai metrik utama.

## Struktur Proyek
- `evaluate.py`: skrip evaluasi model.
- `train_baseline.py`: skrip pelatihan baseline.
- `submission/model.py`: definisi model untuk evaluasi.
- `submission/model.pth`: bobot model hasil pelatihan.

## Dataset
Dataset disediakan oleh tim pengajar mata kuliah Deep Learning ITERA. Dataset tidak disertakan dalam repository ini.

## Status
Persiapan repository dan implementasi baseline.
