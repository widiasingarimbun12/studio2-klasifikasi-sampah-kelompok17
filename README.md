# Studio 2 — Klasifikasi Sampah · Kelompok 17

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  |Widia Salsalina Br Singarimbun|123140035     |                  |
| 2  |Audy Olivya Br Gurusinga              |123140025     |                  |
| 3  |Jesika Filosovi Br Perangin-angin              |123140044      |                 |

# Deskripsi
Proyek ini bertujuan membangun model deep learning berbasis CNN untuk mengklasifikasikan citra sampah ke dalam lima kelas: botol plastik, gelas plastik, kertas/kardus, saset kemasan, dan styrofoam.

## Pendekatan
Model awal menggunakan ResNet-18 pretrained dan akan dievaluasi menggunakan Macro-F1 sebagai metrik utama.

## Ringkasan Pendekatan

- **Arsitektur / backbone**: (mis. ResNet-50 pretrained ImageNet, CNN from scratch, ...)
- **Strategi pelatihan**: (freezing, learning rate, scheduler, epoch, augmentasi, ...)
- **Data tambahan**: (tidak ada / sebutkan sumber dan jumlahnya)
- **Ukuran input & preprocessing**:

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | | | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py ...            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
...                 # kode/notebook pelatihan kelompok
```
