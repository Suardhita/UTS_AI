# Laporan Ujian Tengah Semester

# Artificial Intelligence

## Sistem Deteksi Penggunaan Alat Pelindung Diri (APD/PPE)

**Menggunakan YOLO11 Nano (YOLO11n) oleh Ultralytics**

*Deep Learning • Computer Vision • Object Detection*

### Disusun oleh

- I Nyoman Budiarta (2501010074)
- I Made Suardhita Kusuma (2501010075)
- Putri Rahmawati Ambarwangi (2501010059)

---

## 1. LATAR BELAKANG

Pengawasan penggunaan Alat Pelindung Diri (APD) di lingkungan kerja umumnya dilakukan secara langsung oleh petugas keselamatan. Namun, pengawasan manual memiliki keterbatasan, terutama ketika jumlah pekerja cukup banyak dan area kerja yang harus dipantau sangat luas. Selain itu, proses pemantauan sangat bergantung pada ketelitian serta konsentrasi petugas sehingga berpotensi menimbulkan kesalahan atau terlewatnya pelanggaran penggunaan APD.

Untuk membantu mengatasi keterbatasan tersebut, diperlukan pendekatan yang mampu mendeteksi penggunaan APD secara otomatis melalui analisis gambar. Salah satu teknologi yang dapat dimanfaatkan adalah Artificial Intelligence (AI), khususnya metode Deep Learning dalam bidang Computer Vision. Teknologi ini memungkinkan komputer mempelajari pola visual dari data gambar dan mengenali objek berdasarkan karakteristik yang dipelajari selama proses pelatihan.

Pada proyek ini, model yang digunakan adalah You Only Look Once versi 11 Nano (YOLO11n) dari Ultralytics. Model ini dipilih karena relatif ringan, dirancang untuk deteksi objek yang efisien, dan mampu mengenali lebih dari satu objek dalam satu gambar sekaligus memperkirakan posisi objek melalui bounding box. Penerapan model YOLO11n diharapkan dapat menjadi dasar pengembangan sistem yang membantu pemantauan penggunaan maupun ketidakgunaan APD.

## 2. RUMUSAN MASALAH

- 2.1  Bagaimana proses pengolahan dan persiapan dataset APD/PPE, mulai dari pemeriksaan struktur dan penghitungan jumlah data, pembersihan data (data cleaning), hingga pengecekan kembali data train, validation, dan test?

- 2.2  Bagaimana penerapan metode Deep Learning menggunakan model YOLO11 Nano untuk mendeteksi dan mengidentifikasi berbagai jenis APD/PPE pada gambar?

- 2.3  Bagaimana proses pelatihan model YOLO11 Nano menggunakan dataset PPE dengan memanfaatkan Google Colab dan GPU untuk mendukung komputasi?

- 2.4  Bagaimana performa model YOLO11 Nano berdasarkan metrik Precision, Recall, mAP50, mAP50–95, serta Confusion Matrix?

- 2.5  Bagaimana kemampuan model YOLO11 Nano dalam mendeteksi penggunaan maupun ketidakgunaan APD melalui pengujian dengan gambar dari dataset dan gambar baru di luar dataset?

## DATASET

Dataset yang digunakan dalam proyek ini adalah **Personal Protective Equipment (PPE)** yang diperoleh dari Roboflow Universe. Dataset ini berisi gambar dan anotasi objek APD yang digunakan sebagai data untuk proses pelatihan, validasi, dan pengujian model YOLO11n.

**Sumber dataset:** [Personal Protective Equipment (PPE) – Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/dataset/8)

## MODEL YANG DIGUNAKAN

Model yang digunakan dalam proyek ini adalah **YOLO11n (You Only Look Once versi 11 Nano)** yang dikembangkan oleh Ultralytics. YOLO11n merupakan model kecerdasan buatan yang digunakan untuk mengenali dan mendeteksi objek secara otomatis pada gambar maupun video. Model ini dapat menemukan objek sekaligus memperkirakan letaknya menggunakan *bounding box*.

Versi Nano dirancang dengan ukuran model yang relatif ringan agar proses deteksi dapat berjalan dengan cepat dan efisien. Dalam proyek ini, YOLO11n digunakan untuk mendeteksi berbagai kategori objek Alat Pelindung Diri (APD/PPE) berdasarkan gambar pada dataset yang telah disiapkan.

## METODE YANG DIGUNAKAN

Metode yang digunakan dalam proyek ini adalah **Deep Learning**. Deep Learning merupakan bagian dari *machine learning* yang menggunakan jaringan saraf tiruan dengan banyak lapisan untuk mempelajari pola dan ciri-ciri dari data secara otomatis.

Pada proyek ini, metode Deep Learning diterapkan untuk melatih model YOLO11n menggunakan gambar dan anotasi dari dataset PPE. Melalui proses pelatihan, model mempelajari ciri visual dari setiap kategori APD sehingga nantinya dapat mengenali objek dan menentukan lokasinya pada gambar atau video yang diuji.

## 3. TAHAPAN PENGERJAAN PROYEK

### Persiapan Awal: Library dan Pemeriksaan GPU

Kode berikut dijalankan sebelum pengolahan dataset. Seluruh potongan kode pada Bab 3 bersumber dari dokumentasi notebook Colab proyek PPE sebelumnya; nomor sel asli dipertahankan sebagai referensi.

```python
!pip -q install ultralytics
import hashlib
import shutil
import zipfile
from pathlib import Path
import torch
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from google.colab import drive
from ultralytics import YOLO
print('GPU aktif:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Nama GPU:', torch.cuda.get_device_name(0))
```

### 3.1 Pengambilan dan Ekstraksi Dataset

Tahap awal dilakukan dengan menghubungkan Google Drive ke Google Colab agar dataset dapat diakses dari lingkungan notebook. Setelah lokasi file arsip dataset ditemukan dan diperiksa, seluruh isi arsip diekstraksi ke folder kerja untuk digunakan pada tahap persiapan berikutnya.

```python
from google.colab import drive
drive.mount('/content/drive')
file_zip_drive = Path('/content/drive/MyDrive/Dataset/Dataset.zip')
if not file_zip_drive.exists():
    raise FileNotFoundError('File ZIP tidak ditemukan. Periksa path Google Drive.')
folder_ekstrak = Path('dataset')
folder_ekstrak.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(file_zip_drive, 'r') as file_zip:
    file_zip.extractall(folder_ekstrak)
print('Dataset diekstrak ke:', folder_ekstrak)
```

### 3.2 Pemeriksaan Struktur Dataset

Sistem memeriksa keberadaan dataset serta susunan folder train, valid, dan test. Pemeriksaan bertujuan memastikan gambar beserta anotasinya tersusun sesuai format YOLO sehingga dapat digunakan untuk pelatihan, validasi, dan pengujian model.

```python
folder_dataset = None
for file_yaml in folder_ekstrak.rglob('data.yaml'):
    periksa_folder = file_yaml.parent
    if (periksa_folder / 'train' / 'images').is_dir() and \
       (periksa_folder / 'valid' / 'images').is_dir() and \
       (periksa_folder / 'test' / 'images').is_dir():
        folder_dataset = periksa_folder
        break
if folder_dataset is None:
    raise FileNotFoundError('Struktur dataset YOLO tidak ditemukan. Cek isi ZIP.')
with open(folder_dataset / 'data.yaml', 'r') as file:
    data_yaml = yaml.safe_load(file)
nama_kelas = data_yaml['names']
if isinstance(nama_kelas, dict):
    nama_kelas = [nama_kelas[k] for k in sorted(nama_kelas, key=lambda k: int(k))]
print('Jumlah kelas:', len(nama_kelas))
for nomor, nama in enumerate(nama_kelas):
    print(nomor, '-', nama)
ekstensi_gambar = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
jumlah_gambar = {}
for bagian in ['train', 'valid', 'test']:
    lokasi_gambar = folder_dataset / bagian / 'images'
    jumlah = 0
    for gambar in lokasi_gambar.iterdir():
        if gambar.is_file() and gambar.suffix.lower() in ekstensi_gambar:
            jumlah += 1
    jumlah_gambar[bagian] = jumlah
    print(bagian, ':', jumlah, 'gambar')
print('Total gambar:', sum(jumlah_gambar.values()))
```

### 3.3 Pembersihan Data (Data Cleaning)

Proses data cleaning dilakukan dengan membuat salinan dataset yang telah dibersihkan tanpa mengubah berkas dataset sumber. Gambar yang teridentifikasi identik disimpan satu kali dengan mempertahankan kemunculan pertama berdasarkan urutan pemeriksaan train, valid, kemudian test. Jumlah data duplikat yang dikeluarkan dicatat sebagai bagian dari dokumentasi proses pembersihan.

```python
folder_bersih = Path('/content/dataset_bersih')
if folder_bersih.exists():
    shutil.rmtree(folder_bersih)
for bagian in ['train', 'valid', 'test']:
    (folder_bersih / bagian / 'images').mkdir(parents=True)
    (folder_bersih / bagian / 'labels').mkdir(parents=True)
hash_terpakai = {}
jumlah_bersih = {'train': 0, 'valid': 0, 'test': 0}
tanpa_label = 0
label_tidak_valid = 0
gambar_duplikat = 0
duplikat_label_berbeda = 0
label_kosong = 0
for bagian in ['train', 'valid', 'test']:
    folder_gambar = folder_dataset / bagian / 'images'
    folder_label = folder_dataset / bagian / 'labels'
    if not folder_label.is_dir():
        raise FileNotFoundError(f'Folder label tidak ditemukan: {folder_label}')
    for gambar in sorted(folder_gambar.iterdir()):
        if gambar.suffix.lower() not in ekstensi_gambar:
            continue
        label = folder_label / (gambar.stem + '.txt')
        if not label.exists():
            tanpa_label += 1
            continue
        isi_label = label.read_text(encoding='utf-8').splitlines()
        isi_label = [baris.strip() for baris in isi_label if baris.strip()]
        valid = True
        for baris in isi_label:
            kolom = baris.split()
            if len(kolom) != 5:
                valid = False
                break
            try:
                nilai = [float(angka) for angka in kolom]
                id_kelas = nilai[0]
                x, y, lebar, tinggi = nilai[1:]
                if not (id_kelas.is_integer() and 0 <= id_kelas < len(nama_kelas)
                        and 0 <= x <= 1 and 0 <= y <= 1
                        and 0 < lebar <= 1 and 0 < tinggi <= 1):
                    valid = False
                    break
            except ValueError:
                valid = False
                break
        if not valid:
            label_tidak_valid += 1
            continue
        if len(isi_label) == 0:
            label_kosong += 1
        kode_hash = hashlib.sha256(gambar.read_bytes()).hexdigest()
        isi_anotasi = tuple(sorted(isi_label))
        if kode_hash in hash_terpakai:
            gambar_duplikat += 1
            if isi_anotasi != hash_terpakai[kode_hash]:
                duplikat_label_berbeda += 1
            continue
        hash_terpakai[kode_hash] = isi_anotasi
        gambar_baru = folder_bersih / bagian / 'images' / (kode_hash + gambar.suffix.lower())
        label_baru = folder_bersih / bagian / 'labels' / (kode_hash + '.txt')
        shutil.copy2(gambar, gambar_baru)
        label_baru.write_text('\n'.join(isi_label) + ('\n' if isi_label else ''), encoding='utf-8')
        jumlah_bersih[bagian] += 1
print('HASIL CLEANING')
print('Jumlah setelah cleaning:', jumlah_bersih)
print('Gambar tanpa label:', tanpa_label)
print('Label tidak valid:', label_tidak_valid)
print('Gambar duplikat:', gambar_duplikat)
print('Duplikat dengan anotasi berbeda:', duplikat_label_berbeda)
print('Label kosong yang dipertahankan:', label_kosong)
```

### 3.4 Pemeriksaan Jumlah Data dan Penyusunan Konfigurasi Dataset 

### Pembagian Dataset (Data Split)

Dataset yang kami gunakan sudah dibagi dari sumbernya menjadi tiga bagian, yaitu **train**, **valid**, dan **test**. Jadi, pada proyek ini kami tidak melakukan proses pembagian dataset atau membagi ulang gambar ke dalam ketiga bagian tersebut.

Kami hanya **mengkalkulasikan atau menghitung jumlah gambar** yang terdapat pada masing-masing folder train, valid, dan test. Penghitungan ini dilakukan untuk mengetahui jumlah data pada setiap bagian. Setelah proses data cleaning, jumlah gambar juga diperiksa dan dihitung kembali untuk mengetahui jumlah data yang tersisa dan memastikan dataset siap digunakan untuk proses pelatihan, validasi, dan pengujian model.
 
Setelah pembersihan, sistem memeriksa jumlah data pada setiap bagian dataset untuk memastikan data yang tersedia memenuhi persyaratan minimum pada notebook. Selanjutnya, file konfigurasi dataset disusun kembali dengan mengacu pada struktur dataset bersih sehingga lokasi data dan informasi kelas dapat dibaca dengan benar oleh YOLO11n.

```python
total_gambar = sum(jumlah_bersih.values())
if total_gambar == 0:
    raise ValueError('Tidak ada gambar yang tersisa. Periksa hasil cleaning.')
for bagian in ['train', 'valid', 'test']:
    if jumlah_bersih[bagian] == 0:
        raise ValueError(f'Data {bagian} kosong. Periksa dataset.')
    persen = jumlah_bersih[bagian] / total_gambar * 100
    print(f'{bagian}: {jumlah_bersih[bagian]} gambar ({persen:.2f}%)')
print('Total gambar bersih:', total_gambar)
konfigurasi_yolo = {
    'path': str(folder_bersih),
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'names': nama_kelas
}
file_yaml_bersih = folder_bersih / 'data.yaml'
with open(file_yaml_bersih, 'w') as file:
    yaml.safe_dump(konfigurasi_yolo, file, allow_unicode=True, sort_keys=False)
print('File konfigurasi:', file_yaml_bersih)
```

### 3.5 Pemilihan Model

Model yang digunakan adalah YOLO11 Nano (YOLO11n), dengan bobot awal pretrained dari file yolo11n.pt. Pendekatan ini disebut transfer learning karena bobot hasil pelatihan sebelumnya dimanfaatkan sebagai titik awal, kemudian disesuaikan dengan dataset PPE yang memiliki 14 kategori objek.

Parameter pada notebook sumber: 50 epoch, batch size 16, dan ukuran input 640 piksel. Parameter pelatihan dapat diubah jika tersedia sumber daya GPU dan kebutuhan eksperimen berbeda.

```python
if not torch.cuda.is_available():
    raise RuntimeError('GPU belum aktif.')
nama_model = 'yolo11n.pt'
epoch = 50
batch_size = 16
ukuran_gambar = 640
model = YOLO(nama_model)
print('Model:', nama_model)
print('Epoch:', epoch)
print('Batch size:', batch_size)
print('Ukuran input:', ukuran_gambar)
```

### 3.6 Pelatihan Model

Pelatihan dilakukan melalui fungsi train() dari pustaka Ultralytics menggunakan konfigurasi yang telah ditentukan pada notebook. Mixed precision diaktifkan dengan amp=True, augmentasi Mosaic dinonaktifkan melalui mosaic=0.0, jumlah worker diatur menjadi workers=0, dan pembuatan grafik hasil pelatihan diaktifkan melalui plots=True.

Hasil pelatihan diarahkan ke direktori /content/PPE_Runs/training_yolo11. Setelah proses pelatihan selesai, notebook menyiapkan lokasi bobot model terbaik, yaitu folder weights/best.pt, yang digunakan untuk tahap evaluasi dan prediksi.

Catatan hasil sementara. Pada cuplikan output yang tersedia, epoch 1 dan 2 sudah menampilkan metrik validasi, sedangkan epoch 3 masih berupa log proses yang belum selesai. Karena itu, cuplikan tersebut belum cukup untuk menyimpulkan hasil akhir seluruh pelatihan.

```python
hasil_training = model.train(
    amp=True,
    data=str(file_yaml_bersih),
    epochs=epoch,
    batch=batch_size,
    imgsz=ukuran_gambar,
    device=0,
    mosaic=0.0,
    workers=0,
    plots=True,
    project='/content/PPE_Runs',
    name='training_yolo11',
    exist_ok=True
)
folder_training = Path(model.trainer.save_dir)
file_model_terbaik = folder_training / 'weights' / 'best.pt'
print('File model terbaik:', file_model_terbaik)
```

### 3.7 Visualisasi Hasil Pelatihan

Notebook menyiapkan grafik perbandingan training loss dan validation loss untuk tiga komponen, yaitu box loss, classification loss, dan Distribution Focal Loss (DFL). Selain itu, dibuat grafik perkembangan metrik Precision, Recall, mAP50, dan mAP50–95 sepanjang epoch pelatihan.

Seluruh grafik menggunakan data dari file results.csv yang tersimpan pada direktori hasil training. Visualisasi tersebut digunakan untuk mengamati perkembangan proses pembelajaran serta membantu mengidentifikasi indikasi overfitting atau underfitting. Penilaian akhir tetap memerlukan pemeriksaan metrik dan grafik secara menyeluruh.

```python
file_riwayat = folder_training / 'results.csv'
riwayat = pd.read_csv(file_riwayat)
riwayat.columns = riwayat.columns.str.strip()
for jenis in ['box', 'cls', 'dfl']:
    kolom_train = 'train/' + jenis + '_loss'
    kolom_valid = 'val/' + jenis + '_loss'
    if kolom_train not in riwayat.columns or kolom_valid not in riwayat.columns:
        print('Kolom tidak ditemukan:', jenis)
        continue
    plt.figure(figsize=(8, 4))
    plt.plot(riwayat['epoch'], riwayat[kolom_train], label='Training')
    plt.plot(riwayat['epoch'], riwayat[kolom_valid], label='Validation')
    plt.title('Training vs Validation ' + jenis.upper() + ' Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder_training / ('grafik_loss_' + jenis + '.png'))
    plt.show()
metrik_kolom = {
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall',
    'metrics/mAP50(B)': 'mAP50',
    'metrics/mAP50-95(B)': 'mAP50-95'
}
plt.figure(figsize=(8, 4))
for nama_kolom, nama_metrik in metrik_kolom.items():
    if nama_kolom in riwayat.columns:
        plt.plot(riwayat['epoch'], riwayat[nama_kolom], label=nama_metrik)
plt.title('Perkembangan Metrik Validation')
plt.xlabel('Epoch')
plt.ylabel('Nilai metrik')
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(folder_training / 'grafik_metrik.png')
plt.show()
```

### 3.8 Evaluasi Model

Kode evaluasi memuat bobot terbaik best.pt dan menjalankan fungsi val() pada bagian test. Metrik yang dihitung meliputi Precision, Recall, mAP50, serta mAP50–95. Hasil metrik agregat kemudian ditampilkan dalam bentuk grafik batang agar lebih mudah dibandingkan.

Tahap tambahan menghitung rata-rata F1-score per kelas dan menyajikan seluruh metrik evaluasi dalam persentase. Evaluasi ini bertujuan menilai kemampuan model dalam mengenali objek APD pada data yang tidak digunakan untuk memperbarui bobot saat pelatihan.

```python
model_terbaik = YOLO(str(file_model_terbaik))
hasil_evaluasi = model_terbaik.val(
    data=str(file_yaml_bersih),
    split='test',
    device=0,
    batch=batch_size,
    plots=True,
    project='/content/PPE_Runs',
    name='evaluasi_test',
    exist_ok=True
)
folder_evaluasi = Path('/content/PPE_Runs/evaluasi_test')
precision = float(hasil_evaluasi.box.mp)
recall = float(hasil_evaluasi.box.mr)
map50 = float(hasil_evaluasi.box.map50)
map5095 = float(hasil_evaluasi.box.map)
print('Precision:', round(precision, 4))
print('Recall:', round(recall, 4))
print('mAP50:', round(map50, 4))
print('mAP50-95:', round(map5095, 4))
plt.figure(figsize=(8, 4))
plt.bar(['Precision', 'Recall', 'mAP50', 'mAP50-95'],
        [precision, recall, map50, map5095])
plt.title('Performa Deteksi PPE pada Data Test')
plt.ylabel('Nilai (0-1)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(folder_evaluasi / 'grafik_performa_test.png')
plt.show()
if 'hasil_evaluasi' not in globals():
    print('Jalankan sel Evaluasi di atas terlebih dahulu.')
else:
    metrik = hasil_evaluasi.box
    precision_persen = float(metrik.mp) * 100
    recall_persen = float(metrik.mr) * 100
    map50_persen = float(metrik.map50) * 100
    map5095_persen = float(metrik.map) * 100
    if len(metrik.f1) > 0:
        f1_persen = float(metrik.f1.mean()) * 100
        print(f'F1 rata-rata kelas: {f1_persen:.2f}%')
    else:
        print('F1 belum tersedia. Periksa label pada data test.')
    print(f'Precision         : {precision_persen:.2f}%')
    print(f'Recall            : {recall_persen:.2f}%')
    print(f'mAP50             : {map50_persen:.2f}%')
    print(f'mAP50-95          : {map5095_persen:.2f}%')
    print('mAP50: evaluasi pada IoU 0.50.')
    print('mAP50-95: rata-rata pada IoU 0.50 sampai 0.95.')
    print('P, R, dan F1 mengikuti titik confidence pilihan evaluator.')
    print('Laporkan nama metriknya; jangan mengganti mAP menjadi accuracy.')
```

### 3.9 Confusion Matrix dan Evaluasi per Kelas

Confusion Matrix ditampilkan melalui dua file hasil evaluasi, yaitu confusion_matrix.png dan confusion_matrix_normalized.png. Visualisasi ini digunakan untuk melihat pola prediksi benar dan kesalahan klasifikasi antar kategori.

Kode berikutnya mengekstraksi metrik untuk setiap kelas dan menyimpannya sebagai tabel CSV. Dengan demikian, performa kelas yang relatif baik maupun kelas yang masih lemah dapat diperiksa dan dibandingkan secara lebih rinci.

```python
for nama_file in ['confusion_matrix.png', 'confusion_matrix_normalized.png']:
    lokasi_gambar = folder_evaluasi / nama_file
    if lokasi_gambar.exists():
        print(nama_file)
        display(Image.open(lokasi_gambar))
    else:
        print('Belum ditemukan:', lokasi_gambar)
hasil_per_kelas = []
for urutan, nomor_kelas in enumerate(hasil_evaluasi.box.ap_class_index):
    p, r, ap50, ap5095 = hasil_evaluasi.box.class_result(urutan)
    hasil_per_kelas.append({
        'Kelas': nama_kelas[int(nomor_kelas)],
        'Precision': round(float(p), 3),
        'Recall': round(float(r), 3),
        'mAP50': round(float(ap50), 3),
        'mAP50-95': round(float(ap5095), 3)
    })
if hasil_per_kelas:
    tabel_kelas = pd.DataFrame(hasil_per_kelas)
    display(tabel_kelas)
    tabel_kelas.to_csv(folder_evaluasi / 'hasil_per_kelas.csv', index=False)
else:
    print('Tidak ada kelas yang dapat dihitung. Periksa data test.')
```

### 3.10 Pengujian dengan Gambar dari Dataset

Salah satu gambar dari direktori testing dipilih sebagai sampel pengujian. Model terbaik digunakan untuk menjalankan fungsi predict() dengan confidence threshold sebesar 0,05. Hasil prediksi ditampilkan pada gambar menggunakan bounding box, nama kelas objek, dan nilai confidence dari setiap deteksi.

```python
folder_test = folder_bersih / 'test' / 'images'
gambar_uji = None
for gambar in folder_test.iterdir():
    if gambar.suffix.lower() in ekstensi_gambar:
        gambar_uji = gambar
        break
if gambar_uji is None:
    raise FileNotFoundError('Gambar test tidak ditemukan.')
prediksi = model_terbaik.predict(source=str(gambar_uji), conf=0.05)
print('Gambar yang diuji:', gambar_uji.name)
display(Image.fromarray(prediksi[0].plot()[..., ::-1]))
```

### 3.11 Pengujian dengan Gambar Baru di Luar Dataset

Tahap akhir  pengujian menggunakan gambar baru yang diunggah dari komputer. Gambar dikonversi ke format RGB, kemudian diproses menggunakan ukuran input 1280 piksel dengan confidence threshold sebesar 0,010.

```python
from google.colab import files
from PIL import Image
from IPython.display import display
# Membuka tombol untuk memilih gambar dari komputer.
gambar_upload = files.upload()
# Melakukan prediksi untuk setiap gambar yang diunggah.
for nama_file, isi_file in gambar_upload.items():
    gambar = Image.open(nama_file).convert('RGB')
    hasil_prediksi = model_terbaik.predict(
        source=gambar,
        imgsz=1280,
        conf=0.010,
        classes=None,
        agnostic_nms=False,
        max_det=100,
        save=True,
        project='/content/PPE_Runs',
        name='testing_local',
        exist_ok=True
    )
    print('Nama gambar:', nama_file)
    # Mengubah warna BGR menjadi RGB agar tampil dengan benar.
    gambar_hasil = hasil_prediksi[0].plot()
    display(Image.fromarray(gambar_hasil[..., ::-1]))
```

Jalankan sel secara berurutan di Google Colab. Sesuaikan path file ZIP di Google Drive dan pastikan runtime GPU aktif. Hasil pelatihan dan evaluasi dalam laporan tidak boleh dianggap sudah tersedia hanya karena kode ditampilkan; hasil sebenarnya harus diperoleh dengan menjalankan notebook.
