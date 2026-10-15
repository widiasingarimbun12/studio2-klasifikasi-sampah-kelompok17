# Studio Proyek 2 — Klasifikasi Sampah

## Deskripsi
Proyek ini membandingkan tiga arsitektur CNN untuk klasifikasi gambar
sampah ke dalam lima kelas: botol plastik, gelas plastik, kertas/kardus,
saset kemasan, dan styrofoam.

Model yang dibandingkan:
- ResNet-18
- EfficientNet-B0
- MobileNetV3-Large

## Model Final
EfficientNet-B0 dipilih berdasarkan hasil evaluasi pada data validasi.

| Metrik | Hasil |
|---|---:|
| Accuracy | 94.92% |
| Macro-F1 | 94.53% |
| Ukuran bobot model | 15.60 MB |
| Jumlah gambar validasi | 413 |

## Perbandingan Model

| Model | Accuracy | Macro-F1 | Ukuran bobot |
|---|---:|---:|---:|
| ResNet-18 | 93.46% | 93.20% | 42.71 MB |
| EfficientNet-B0 | 94.92% | 94.53% | 15.60 MB |
| MobileNetV3-Large | 94.43% | 94.11% | 16.25 MB |

Hasil berasal dari evaluasi pada data validasi yang sama. Karena data ini
juga digunakan untuk pemilihan model, hasil tersebut bukan estimasi
performa pada test set independen.

## Struktur Proyek

- `submission/model.py` — definisi arsitektur dan transformasi input.
- `submission/model.pth` — bobot EfficientNet-B0 terpilih.
- `evaluate.py` — skrip evaluasi.
- `train_model.py` — skrip pelatihan baseline.
- `train_baseline.py` — skrip eksperimen baseline.
- `train_efficientnet.py` — skrip eksperimen EfficientNet-B0.
- `train_mobilenet.py` — skrip eksperimen MobileNetV3-Large.
- `train_augmentasi.py` — skrip eksperimen augmentasi.
- `requirements.txt` — dependensi Python.
- `experiments/` — hasil eksperimen dan evaluasi.

## Menjalankan Evaluasi

Pastikan dataset sudah tersedia secara lokal dan mengikuti struktur
folder yang diharapkan oleh `evaluate.py`.

```bash
python evaluate.py --submission submission --data PATH_DATASET_VALIDASI --batch-size 32 --out evaluation_result.json
```

Ganti `PATH_DATASET_VALIDASI` dengan lokasi folder validasi yang berisi
lima subfolder kelas.

## Catatan Dataset
Dataset tidak disertakan dalam repositori ini. Siapkan dataset secara
terpisah dan patuhi ketentuan lisensi serta penggunaan dari sumbernya.

## Lingkungan
Dependensi utama tercantum di `requirements.txt`. Hasil waktu evaluasi
dapat berbeda bergantung pada perangkat keras dan lingkungan eksekusi.
