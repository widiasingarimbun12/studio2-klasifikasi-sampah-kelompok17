# Studio 2 — Klasifikasi Sampah · Kelompok <NOMOR/NAMA KELOMPOK>

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Widia Salsalina Br Singarimbun             |123140035     |                  |
| 2  |Audy Olivya Br Gurusinga              |123140025     |                  |
| 3  |Jesika Filosovi Br Perangin-angin             |123140044      |                 |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: EfficientNet-B0 pretrained pada ImageNet, dengan lapisan klasifikasi (classifier) yang disesuaikan untuk memprediksi lima kelas sampah: botol_plastik, gelas_plastik, kertas_kardus, saset_kemasan, dan styrofoam.
- **Strategi pelatihan**: Menggunakan transfer learning dengan bobot pretrained ImageNet. Model dilatih selama 10 epoch menggunakan optimizer AdamW dengan learning rate awal (3 \times 10^{-4}), weight decay (10^{-4}), Cosine Annealing Learning Rate Scheduler, dan Cross Entropy Loss dengan label smoothing 0,05. Pelatihan menggunakan batch size 32 dan memilih checkpoint terbaik berdasarkan nilai Macro-F1 pada data validasi.
- **Data tambahan**: Tidak ada data tambahan 
- **Ukuran input & preprocessing**: Gambar diubah ukurannya menjadi 256 × 256 piksel, kemudian dilakukan Center Crop menjadi 224 × 224 piksel. Selanjutnya, gambar dikonversi menjadi tensor dan dinormalisasi menggunakan mean [0.485, 0.456, 0.406] serta standard deviation [0.229, 0.224, 0.225]. Selama pelatihan, augmentasi yang digunakan meliputi Random Resized Crop dengan skala 0,7–1,0, Random Horizontal Flip, dan Color Jitter dengan parameter 0,15.

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
