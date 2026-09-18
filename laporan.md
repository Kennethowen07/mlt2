# Laporan Proyek Machine Learning - Kenneth Owen Gozali

## Project Overview

Katalog buku tumbuh jauh lebih cepat daripada kemampuan pembaca menyaringnya. Data International Publishers Association mencatat Indonesia menerbitkan 107.856 buku ber-ISBN sepanjang 2022 dan menempati peringkat sembilan dunia, sementara rata-rata konsumsi bacaan per orang hanya 5,91 judul per tahun. Selisih antara jumlah pilihan dan jumlah yang benar-benar dibaca itulah yang menjadi persoalan, bukan ketersediaan buku.

Iyengar dan Lepper [4] menunjukkan bahwa memperbanyak pilihan justru menurunkan kemungkinan seseorang mengambil keputusan. Pada percobaan mereka, meja pajangan berisi 24 varian selai menarik lebih banyak pengunjung tetapi menghasilkan pembelian jauh lebih sedikit dibanding meja berisi 6 varian. Toko buku daring menghadapi bentuk yang sama dalam skala lebih besar, dan pengguna yang tidak menemukan titik masuk akan berhenti menelusuri.

Sistem rekomendasi menjawab persoalan ini dengan menyempitkan katalog menjadi daftar pendek yang disesuaikan. Ricci dkk. [2] mencatat empat manfaat utamanya, yaitu menaikkan jumlah item yang terjual, memperluas keragaman item yang terjual termasuk yang tidak populer, menaikkan kepuasan pengguna, dan memberi penjual pemahaman lebih baik tentang preferensi pelanggan. Manfaat kedua penting khusus untuk buku, karena tanpa rekomendasi penjualan cenderung menumpuk pada segelintir bestseller sementara sebagian besar katalog tidak pernah ditemukan siapa pun.

Proyek ini membangun sistem rekomendasi buku dari data Book-Crossing, kumpulan rating yang dikumpulkan Ziegler dkk. [1] lewat crawling komunitas Book-Crossing pada 2004. Dua pendekatan dibangun berdampingan supaya kelemahan masing-masing terlihat, content based filtering yang bekerja dari atribut buku dan collaborative filtering yang bekerja dari pola penilaian antar pengguna.

## Business Understanding

### Problem Statements

- Bagaimana merekomendasikan buku lain kepada pengguna yang baru selesai membaca satu judul tertentu, tanpa bergantung pada riwayat penilaiannya?
- Bagaimana memprediksi seberapa besar kemungkinan seorang pengguna menyukai buku yang belum pernah ia nilai, sehingga sistem dapat menyusun daftar top-N yang dipersonalisasi?
- Seberapa besar sebenarnya sumbangan pemodelan terhadap kualitas rekomendasi dibanding tebakan sederhana berbasis rata-rata?

### Goals

- Membangun sistem yang menerima satu judul buku sebagai masukan dan mengembalikan lima buku paling mirip berdasarkan atribut kontennya.
- Membangun sistem yang memprediksi rating untuk seluruh pasangan pengguna dan buku, lalu menyusun sepuluh rekomendasi teratas dari buku yang belum pernah dinilai pengguna tersebut.
- Mengukur kedua sistem terhadap tolok ukur dasar, sehingga peningkatan yang diperoleh dapat dinyatakan sebagai angka dan bukan sebagai klaim.

### Solution statements

Dua pendekatan diajukan, masing-masing menjawab problem statement yang berbeda.

**Content based filtering dengan TF-IDF dan cosine similarity.** Judul, penulis, dan penerbit setiap buku digabung menjadi satu dokumen teks, diubah menjadi vektor berbobot TF-IDF, lalu kemiripan antar buku dihitung dengan cosine similarity. Pendekatan ini menjawab problem statement pertama karena tidak membutuhkan riwayat pengguna sama sekali, cukup satu judul sebagai titik awal.

**Collaborative filtering dengan matrix factorization.** Matriks pengguna kali buku diuraikan memakai Truncated SVD setelah bias pengguna dan bias buku dikeluarkan, lalu hasil rekonstruksinya dipakai untuk mengisi sel yang kosong. Pendekatan ini menjawab problem statement kedua karena menghasilkan skor untuk buku yang belum pernah disentuh pengguna.

Untuk problem statement ketiga, hasil collaborative filtering dibandingkan terhadap dua tolok ukur, yaitu menebak rata-rata global untuk semua orang dan menebak rata-rata ditambah bias tanpa faktorisasi. Hasil content based filtering dibandingkan terhadap pemilihan lima buku secara acak.

## Data Understanding

Dataset yang digunakan adalah Book-Crossing dari [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset), dikumpulkan Cai-Nicolas Ziegler pada Agustus sampai September 2004 lewat crawling komunitas Book-Crossing selama empat minggu dengan izin pengelolanya. Data terdiri atas tiga berkas terpisah.

| Berkas | Baris | Kolom |
|---|---|---|
| `Ratings.csv` | 1.149.780 | 3 |
| `Books.csv` | 271.360 | 8 |
| `Users.csv` | 278.858 | 3 |

Pembacaan berkas memerlukan `encoding='latin-1'` karena judul buku memuat karakter di luar UTF-8, dan `on_bad_lines='skip'` karena beberapa baris rusak format sejak sumber aslinya.

### Variabel pada dataset

Variabel pada `Ratings.csv`

- `User-ID`, nomor identitas pengguna yang sudah dianonimkan menjadi bilangan bulat.
- `ISBN`, nomor identitas buku sepanjang sepuluh karakter, sebagian memuat huruf.
- `Book-Rating`, penilaian pengguna terhadap buku pada skala 1 sampai 10, dengan nilai 0 sebagai penanda interaksi implisit.

Variabel pada `Books.csv`

- `ISBN`, kunci penghubung ke berkas rating.
- `Book-Title`, judul buku.
- `Book-Author`, nama penulis, hanya satu nama meski buku ditulis beberapa orang.
- `Year-Of-Publication`, tahun terbit edisi bersangkutan.
- `Publisher`, nama penerbit.
- `Image-URL-S`, `Image-URL-M`, `Image-URL-L`, tautan sampul buku dalam tiga ukuran.

Variabel pada `Users.csv`

- `User-ID`, kunci penghubung ke berkas rating.
- `Location`, lokasi pengguna dalam format kota, wilayah, negara pada satu kolom teks.
- `Age`, usia pengguna.

### Exploratory Data Analysis

**Sebaran rating dan makna nilai nol.** Nilai 0 muncul 716.109 kali atau 62,3 persen dari seluruh baris. Dokumentasi Ziegler menyatakan nilai ini menandai interaksi implisit, bukan penilaian rendah, sehingga tidak berada pada skala yang sama dengan 1 sampai 10. Setelah nilai 0 dikeluarkan tersisa 433.671 penilaian dengan rata-rata 7,601.

![Sebaran rating eksplisit](https://raw.githubusercontent.com/USERNAME/REPO/main/images/sebaran-rating.png)

Sebarannya miring tajam ke kanan. Puncaknya pada nilai 8 sebanyak 103.736, sementara nilai 1 hanya muncul 1.770 kali. Nilai 1 sampai 4 digabung pun belum mencapai 20.000 dari 433.671 baris. Rentang efektif rating pada praktiknya sekitar 5 sampai 10, dan model yang dilatih di sini akan sulit mengenali buku yang benar-benar tidak disukai karena contohnya terlalu sedikit.

**Kepadatan interaksi.** Dari 77.805 pengguna yang memberi rating eksplisit, 45.382 di antaranya hanya menilai satu buku. Dari 185.973 buku yang dinilai, 129.621 hanya menerima satu rating. Median kedua sisi sama-sama satu.

![Sebaran jumlah rating per pengguna dan per buku](https://raw.githubusercontent.com/USERNAME/REPO/main/images/kepadatan-interaksi.png)

Sumbu tegak kedua grafik memakai skala logaritmik, karena tanpa itu batang selain yang pertama tidak terlihat sama sekali. Persentil ke-90 pengguna berada di 9 rating dan persentil ke-99 di 73 rating, sementara satu pengguna terekam menilai 8.524 buku. Di sisi buku, persentil ke-99 berada di 22 rating dengan nilai tertinggi 707. Kepadatan matriksnya 0,003 persen.

Bentuk ekor panjang ini menjadi temuan paling menentukan pada proyek, karena collaborative filtering bekerja dengan membandingkan pola penilaian dan pengguna dengan satu rating tidak memiliki pola untuk dibandingkan.

**Anomali pada `Year-Of-Publication`.** Tiga baris memuat nama penerbit di kolom tahun, yaitu `DK Publishing Inc` dan `Gallimard`, akibat tanda kutip dalam judul yang menggeser seluruh kolom saat data aslinya dibentuk. Terdapat 4.618 baris bertahun 0 dan 23 baris bertahun setelah 2004 padahal data dikumpulkan pada tahun itu, salah satunya tertulis 2026. Batas bawahnya juga janggal, terdapat entri bertahun 1376.

![Sebaran tahun terbit](https://raw.githubusercontent.com/USERNAME/REPO/main/images/tahun-terbit.png)

Pada rentang wajar 1950 sampai 2004 terkumpul 266.374 judul dengan median 1996 dan tahun terbanyak 2002. Sebanyak 74,5 persen terbit sejak 1990, sehingga koleksinya condong ke terbitan baru pada masa data dikumpulkan. Ekornya menjulur ke kiri sampai dekade 1950-an tetapi tipis.

**Kondisi kolom `Age`.** Sebanyak 110.762 dari 278.858 baris kosong, hampir empat puluh persen. Dari yang terisi, 1.248 berada di luar rentang wajar dengan nilai tertinggi 244 tahun.

![Sebaran umur pengguna](https://raw.githubusercontent.com/USERNAME/REPO/main/images/umur-pengguna.png)

Sisanya berpusat pada median 32 tahun dengan rata-rata 34,7 dan puncak di usia 24. Kolom ini tidak dipakai sebagai fitur pada kedua pendekatan, karena content based bersandar pada atribut buku dan collaborative pada pola rating.

**Entitas HTML yang belum ter-decode.** Sebanyak 15.791 nama penerbit dan 4.870 judul pada `Books.csv` masih memuat entitas HTML mentah, tersimpan sejak proses crawling dan tidak pernah dikembalikan ke karakter aslinya. Contohnya sebagai berikut.

| Book-Title | Publisher |
|---|---|
| The Mummies of Urumchi | W. W. Norton &amp;amp; Company | 
| New Vegetarian, Bold and Beautiful Recipes for Every Occasion | Ryland Peters &amp;amp; Small Ltd |
| To Kill a Mockingbird | Little Brown &amp;amp; Company |
| Turning Thirty | Hodder &amp;amp; Stoughton General Division |
| Decipher | Simon &amp;amp; Schuster (Trade Division) |

Pada judul polanya sama, `Angels &amp;amp; Demons` seharusnya `Angels & Demons`. Tanpa pembersihan, TF-IDF akan memperlakukan `amp` sebagai kata tersendiri. Karena token itu muncul di ribuan dokumen, dua penerbit yang tidak berhubungan akan terlihat mirip semata karena keduanya memakai ampersand.

**Buku dan penulis terbanyak.** Agatha Christie memimpin dengan 632 judul, disusul William Shakespeare 567 dan Stephen King 524. Penerbit terbanyak adalah Harlequin dengan 7.535 judul dan Silhouette dengan 4.220, keduanya penerbit roman, yang berarti nama penerbit membawa sinyal genre secara tidak langsung. Buku paling banyak dinilai adalah `The Lovely Bones` dengan 707 rating, disusul `Wild Animus` 581 dan `The Da Vinci Code` 487.

**Judul kosong pada tabel buku terpopuler.** Baris kelima tabel tersebut menampilkan `NaN` pada kolom `Book-Title`, yaitu ISBN `0679781587` dengan 333 rating. Penelusuran menunjukkan `Book-Title` pada `Books.csv` sebenarnya tidak memiliki satu pun nilai kosong. `NaN` tersebut muncul dari operasi penggabungan, bukan dari data aslinya, karena ISBN itu ada di `Ratings.csv` tetapi tidak memiliki padanan di `Books.csv` sehingga `merge` dengan `how='left'` mengisinya dengan `NaN`.

Kasusnya tidak tunggal. Terdapat 49.829 rating yang merujuk 36.137 ISBN di luar `Books.csv`, sekitar 11,5 persen dari seluruh rating eksplisit. Penanganannya dilakukan pada tahap 2 Data Preparation, seluruh baris tersebut dibuang, bukan diisi nilai pengganti. Pengisian dengan teks kosong atau penanda seragam seperti `unknown` justru berbahaya, karena TF-IDF akan menganggap semua buku tanpa judul saling mirip dan sistem akan merekomendasikannya satu sama lain. Pemeriksaan setelah tahap 2 memastikan tidak ada lagi `Book-Title` kosong pada data yang dipakai pemodelan.

## Data Preparation

Tahapan berikut dijalankan berurutan, sama persis dengan urutan pada notebook.

**1. Membuang rating implisit.** Seluruh 716.109 baris bernilai 0 dibuang, menyisakan 433.671 baris. Tahapan ini diperlukan karena nilai 0 dan nilai 1 sampai 10 berasal dari dua satuan yang berbeda. Mempertahankan keduanya dalam satu kolom membuat model membaca 0 sebagai penilaian yang jauh lebih buruk daripada 1, padahal 0 justru menandakan ketiadaan penilaian.

**2. Membuang rating untuk buku tanpa metadata.** Sebanyak 49.829 rating atau 11,5 persen merujuk 36.137 ISBN yang tidak ada pada `Books.csv`. Baris ini dibuang, bukan diisi nilai pengganti, karena content based filtering membutuhkan judul, penulis, dan penerbit untuk membandingkan buku. Mengisinya dengan penanda seragam akan membuat seluruh buku tanpa judul terlihat saling mirip di mata TF-IDF. Tahapan ini sekaligus menyelesaikan kemunculan `NaN` pada tabel buku terpopuler di Data Understanding, dan pemeriksaan setelahnya memastikan tidak ada lagi `Book-Title` kosong. Tersisa 383.842 rating dari 68.091 pengguna terhadap 149.836 buku.

**3. Menyaring pengguna dan buku dengan interaksi minimum.** Pengguna dengan kurang dari 10 rating dan buku dengan kurang dari 10 rating dibuang. Penyaringan diulang sampai jumlah baris berhenti berubah, dan pada data ini konvergen setelah 13 putaran. Pengulangan diperlukan karena membuang pengguna berinteraksi rendah dapat menjatuhkan jumlah rating sebuah buku ke bawah ambang, dan sebaliknya, sehingga satu putaran saja meninggalkan sisa yang belum memenuhi syarat.

Ambang 10 ditetapkan setelah membandingkan tiga pilihan.

| Ambang | Baris | Pengguna | Buku | Kepadatan | Matriks kemiripan |
|---|---|---|---|---|---|
| 3 | 183.538 | 14.519 | 22.658 | 0,056% | sekitar 4.100 MB |
| 5 | 115.219 | 6.851 | 9.085 | 0,185% | sekitar 660 MB |
| 10 | 41.456 | 1.820 | 2.030 | 1,122% | sekitar 33 MB |

Ambang 3 mempertahankan data terbanyak tetapi kepadatannya terlalu rendah untuk dipelajari, dan matriks kemiripan antar bukunya melampaui memori yang tersedia di Google Colab. Ambang 10 menaikkan kepadatan dua puluh kali lipat dan menurunkan kebutuhan memori menjadi 33 MB. Pemangkasan yang terjadi memang besar, dari 383.842 menjadi 41.456 baris, tetapi yang hilang sebagian besar adalah baris tunggal yang tidak menyumbang sinyal apa pun kepada collaborative filtering.

**4. Membersihkan metadata buku.** Tiga kolom URL sampul dibuang karena tidak dipakai pemodelan. Entitas HTML di-decode memakai `html.unescape`, lalu spasi berlebih di ujung teks dibuang. Pada 2.030 buku yang tersisa setelah penyaringan, entitas ini masih menyisa di 26 judul dan 37 nama penerbit. Pembersihan ini diperlukan agar TF-IDF tidak membentuk token semu seperti `amp` yang membuat penerbit tak berhubungan terlihat mirip. Setelah penyaringan pada tahap sebelumnya, tidak ada lagi nilai kosong pada `Book-Author` maupun `Publisher`, dan ketiga baris dengan tahun berisi nama penerbit ikut tersingkir dengan sendirinya.

**5. Membuang ISBN ganda untuk judul yang sama.** Sebanyak 183 judul memiliki lebih dari satu ISBN karena terbit dalam beberapa edisi, misalnya `Bridget Jones's Diary` yang muncul empat kali. Satu ISBN dipertahankan per judul, menyisakan 1.847 buku. Tahapan ini diperlukan karena tanpa penghapusan, content based filtering akan merekomendasikan edisi lain dari buku yang sedang dilihat pengguna, dan itu pengulangan, bukan rekomendasi. Penghapusan hanya berlaku pada data content based, sedangkan collaborative filtering tetap memakai seluruh 2.030 ISBN karena yang dipelajari di sana adalah pola penilaian per item.

**6. Menyusun fitur konten.** Judul, penulis, dan penerbit digabung menjadi satu kolom teks bernama `konten`. Penggabungan diperlukan karena TF-IDF bekerja pada satu dokumen per item. Ketiga atribut dipilih karena membawa sinyal yang berbeda, penulis paling kuat menandai kemiripan selera, penerbit menangkap genre secara tidak langsung, dan kata dalam judul menangkap seri serta tema. `Year-Of-Publication` tidak diikutkan karena angka tahun sebagai token teks tidak membawa makna kemiripan.

**7. Menghitung matriks TF-IDF.** Kolom `konten` diubah menjadi matriks berukuran 1.847 kali 3.430 memakai `TfidfVectorizer` dengan `stop_words='english'`. Pembuangan stop word diperlukan agar dua buku tidak dianggap mirip semata karena judulnya sama-sama diawali `The`.

**8. Menyandikan identitas untuk collaborative filtering.** `User-ID` dan `ISBN` dipetakan menjadi indeks bilangan bulat berurutan mulai dari nol, beserta kamus baliknya. Penyandian diperlukan karena indeks baris dan kolom matriks harus rapat dan berurutan, sedangkan `ISBN` berupa teks dan `User-ID` berupa angka yang melompat sampai ratusan ribu.

**9. Membagi data latih dan validasi.** Data diacak dengan `random_state=42` lalu dibagi 80 banding 20, menghasilkan 33.164 baris latih dan 8.292 baris validasi. Pengacakan diperlukan karena urutan asli data mengelompok per pengguna, sehingga tanpa pengacakan data validasi akan berisi pengguna yang tidak pernah muncul di data latih. Rating dibiarkan pada skala aslinya 1 sampai 10 agar galat model terbaca dalam satuan yang sama dengan penilaian pengguna.

## Modeling

### Content Based Filtering

Kemiripan antar buku dihitung dengan cosine similarity terhadap matriks TF-IDF, menghasilkan matriks 1.847 kali 1.847 berisi nilai 0 sampai 1. Fungsi rekomendasi mengambil kolom kemiripan untuk judul masukan, memilih lima nilai tertinggi selain dirinya sendiri, lalu mengembalikan judul beserta penulis dan penerbitnya.

Contoh keluaran untuk masukan `Pet Sematary` karya Stephen King.

| Book-Title | Book-Author | Publisher |
|---|---|---|
| Christine | Stephen King | Signet Book |
| Desperation | Stephen King | Signet Book |
| Firestarter | Stephen King | Signet Book |
| The Tommyknockers | Stephen King | Signet Book |
| Carrie | Stephen King | Signet Book |

Kelima hasilnya karya penulis yang sama. Nama penulis mendominasi karena bobot TF-IDF-nya tinggi, token `king` hanya muncul pada sedikit dokumen sehingga diperlakukan sebagai kata pembeda.

**Kelebihan.** Tidak membutuhkan data rating sama sekali, sehingga buku yang baru masuk katalog dapat langsung direkomendasikan begitu metadatanya tersedia. Bebas dari cold start pada sisi item. Hasilnya juga dapat dijelaskan, karena alasan kemiripan dapat ditelusuri sampai ke kata mana yang berbobot besar.

**Kekurangan.** Rekomendasinya terkurung pada penulis dan penerbit yang sama, sehingga pengguna tidak akan menemukan nama baru. Kualitasnya sepenuhnya bergantung pada kekayaan metadata, dan Book-Crossing hanya menyediakan tiga atribut tanpa genre maupun sinopsis. Pendekatan ini juga tidak mengenal siapa penggunanya, semua orang yang membuka buku yang sama menerima daftar yang persis sama.

### Collaborative Filtering

Matriks pengguna kali buku berukuran 1.820 kali 2.030 disusun dari data latih, dengan sel tanpa rating diisi nol. Tiga suku bias dihitung lebih dahulu, yaitu rata-rata global sebesar 7,946, bias tiap pengguna terhadap rata-rata itu, dan bias tiap buku. Truncated SVD dengan 50 komponen kemudian dijalankan pada residu, yaitu selisih rating terhadap tebakan bias, dan menjelaskan 25,19 persen ragamnya.

Pemisahan bias sebelum faktorisasi diperlukan agar SVD tidak menghabiskan kapasitasnya untuk mempelajari kecenderungan yang sudah bisa dihitung langsung dengan rata-rata. Prediksi akhir adalah penjumlahan rata-rata global, kedua bias, dan hasil rekonstruksi, lalu dijepit ke rentang 1 sampai 10 untuk pelaporan galat. Untuk penyusunan peringkat top-N, skor sebelum penjepitan yang dipakai, karena 3,2 persen prediksi melebihi 10 dan penjepitan akan membuat banyak buku bernilai sama persis sehingga urutannya menjadi acak.

Pengujian dilakukan pada **User-ID 29526**, yang menempati indeks 175 setelah penyandian dan memiliki 28 rating. Lima buku dengan penilaian tertinggi dari pengguna tersebut adalah `Divine Secrets of the Ya-Ya Sisterhood`, `Prodigal Summer`, `The Red Tent`, `Pope Joan`, dan satu judul lain, seluruhnya diberi nilai 10. Kelimanya fiksi dewasa bertema perempuan.

Sepuluh rekomendasi teratas untuk User-ID 29526, disaring dari buku yang belum pernah ia nilai.

| Skor | Book-Title | Book-Author |
|---|---|---|
| 11,66 | Dilbert, A Book of Postcards | Scott Adams |
| 11,52 | Harry Potter and the Chamber of Secrets Postcard Book | J. K. Rowling |
| 11,47 | The Secret Garden | Frances Hodgson Burnett |
| 11,46 | Fox in Socks | Dr. Seuss |
| 11,44 | More Than Complete Hitchhiker's Guide | Douglas Adams |
| 11,41 | The Giving Tree | Shel Silverstein |
| 11,37 | The Authoritative Calvin and Hobbes | Bill Watterson |
| 11,35 | Calvin and Hobbes | Bill Watterson |

Daftar ini didominasi buku bergambar dan komik yang hampir selalu menerima rating tinggi dari siapa pun, sementara riwayat User-ID 29526 seluruhnya fiksi dewasa. Ketidakcocokan itu menunjukkan bias buku memegang peran lebih besar daripada faktor selera.

**Kelebihan.** Mampu merekomendasikan buku yang tidak memiliki kemiripan atribut apa pun dengan riwayat pengguna, sehingga pengguna dapat menemukan penulis dan genre baru. Tidak membutuhkan metadata sama sekali, cukup pola rating. Hasilnya juga bersifat personal, tiap pengguna menerima daftar berbeda.

**Kekurangan.** Menderita cold start pada dua sisi sekaligus, buku baru tanpa rating dan pengguna baru tanpa riwayat sama-sama tidak dapat dilayani. Kualitasnya sangat bergantung pada kepadatan data, dan pada dataset ini penyaringan agresif harus dilakukan sampai tersisa 1,1 persen data awal. Hasilnya juga sulit dijelaskan, karena tidak ada atribut yang bisa ditunjuk sebagai alasan rekomendasi.

## Evaluation

Kedua pendekatan diukur dengan metrik yang berbeda karena keluarannya berbeda. Collaborative filtering menghasilkan prediksi rating berupa angka, sehingga diukur dengan metrik galat. Content based filtering tidak memprediksi rating sama sekali, sehingga diukur dengan metrik relevansi daftar.

### Metrik untuk Collaborative Filtering

**RMSE (Root Mean Squared Error)** adalah akar dari rata-rata kuadrat selisih antara prediksi dan nilai sebenarnya.

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Keterangan simbol.

- $n$, jumlah pasangan pengguna dan buku pada data validasi, pada proyek ini 8.292.
- $i$, penomoran pasangan, berjalan dari 1 sampai $n$.
- $y_i$, rating sebenarnya yang diberikan pengguna pada pasangan ke-$i$, bernilai 1 sampai 10.
- $\hat{y}_i$, dibaca y-topi, yaitu rating hasil prediksi model untuk pasangan yang sama.
- $\sum_{i=1}^{n}$, penjumlahan seluruh suku dari pasangan pertama sampai pasangan ke-$n$.
- $\sqrt{\ }$, akar kuadrat, dipakai untuk mengembalikan hasil ke satuan poin rating.

Cara kerjanya, tiap selisih dikuadratkan terlebih dahulu sehingga selisih besar memberi sumbangan jauh lebih berat daripada selisih kecil. Selisih 4 poin menyumbang 16, sedangkan empat selisih 1 poin hanya menyumbang 4 secara total. Pengakaran di akhir mengembalikan hasilnya ke satuan asli, yaitu poin rating. RMSE dipilih sebagai metrik utama karena pada sistem rekomendasi satu prediksi yang meleset jauh lebih merugikan daripada beberapa prediksi yang meleset tipis, sebab buku yang salah direkomendasikan akan langsung terasa mengganggu bagi pengguna.

**MAE (Mean Absolute Error)** adalah rata-rata nilai mutlak selisih antara prediksi dan nilai sebenarnya.

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

Keterangan simbol $n$, $i$, $y_i$, dan $\hat{y}_i$ sama dengan RMSE. Tambahannya, $|\ |$ berarti nilai mutlak, sehingga prediksi yang kelebihan 2 poin dan yang kekurangan 2 poin dihitung sama beratnya.

Cara kerjanya, setiap selisih dihitung tanpa memperhatikan arah lalu dirata-ratakan, sehingga semua kesalahan diperlakukan setara. MAE disertakan sebagai pendamping RMSE karena angkanya langsung terbaca sebagai rata-rata meleset berapa poin, tanpa distorsi dari pengkuadratan.

### Metrik untuk Content Based Filtering

**Precision@K** adalah proporsi item relevan di antara K item teratas yang direkomendasikan.

$$Precision@K = \frac{|\{\text{item relevan}\} \cap \{\text{K item teratas}\}|}{K}$$

Keterangan simbol.

- $K$, jumlah rekomendasi teratas yang diukur, pada proyek ini 5 karena fungsi rekomendasi mengembalikan lima buku.
- Tanda $@$, dibaca "pada", sehingga Precision@5 berarti ketepatan yang dihitung hanya dari lima rekomendasi teratas dan mengabaikan sisa katalog.
- $\{\text{item relevan}\}$, himpunan buku yang dianggap relevan terhadap buku masukan.
- $\{\text{K item teratas}\}$, himpunan K buku dengan nilai kemiripan tertinggi.
- $\cap$, irisan dua himpunan, yaitu buku yang sekaligus relevan dan masuk daftar teratas.
- $|\ |$, pada rumus ini berarti jumlah anggota himpunan, bukan nilai mutlak.

Cara kerjanya, setiap rekomendasi dinilai relevan atau tidak, lalu jumlah yang relevan dibagi jumlah rekomendasi yang diberikan. Metrik ini dipilih karena pengguna hanya melihat beberapa item teratas dan tidak peduli pada sisa katalog, sehingga yang perlu diukur adalah ketepatan daftar pendek, bukan cakupan keseluruhan.

Karena Book-Crossing tidak menyediakan label genre, relevansi didefinisikan sebagai kesamaan penulis atau kesamaan penerbit antara buku masukan dan buku hasil rekomendasi. Definisi ini merupakan pendekatan pengganti dan bukan penilaian selera pengguna sebenarnya, keterbatasan yang perlu dicatat saat membaca hasilnya.

### Hasil Evaluasi

Hasil collaborative filtering pada 8.292 baris data validasi, dibandingkan terhadap dua tolok ukur.

| Metode | RMSE | MAE |
|---|---|---|
| Tebak rata-rata global | 1,7215 | 1,3422 |
| Rata-rata ditambah bias | 1,5091 | 1,1331 |
| Matrix factorization (SVD 50 komponen) | 1,5066 | 1,1303 |

Hasil content based filtering, dihitung atas seluruh 1.847 buku dengan K sama dengan 5.

| Definisi relevan | Precision@5 |
|---|---|
| Pemilihan acak sebagai tolok ukur | 0,0299 |
| Penulis sama | 0,5225 |
| Penerbit sama | 0,5541 |
| Penulis atau penerbit sama | 0,7238 |

Sebanyak 861 dari 1.847 buku memperoleh Precision@5 sempurna, sedangkan 115 buku tidak memperoleh satu pun rekomendasi relevan.

### Pembahasan Hasil

**Problem statement 1, merekomendasikan buku lain tanpa bergantung pada riwayat penilaian pengguna.**

Tujuan ini tercapai. Content based filtering mencapai Precision@5 sebesar 0,7238, sekitar dua puluh empat kali lipat dibanding tolok ukur pemilihan acak yang hanya 0,0299. Sistem hanya membutuhkan satu judul sebagai masukan dan tidak menyentuh data rating sama sekali.

Batasannya terletak pada jenis kemiripan yang ditangkap. Sistem sangat kuat mengenali karya penulis yang sama, dan itu terlihat pada 861 buku yang memperoleh nilai sempurna. Namun sistem tidak memiliki cara mengenali kemiripan tema antar penulis berbeda, sehingga 115 buku sama sekali tidak mendapat rekomendasi relevan. Buku-buku itu umumnya karya penulis yang hanya muncul sekali dalam data dan diterbitkan penerbit kecil, sehingga tidak ada jangkar untuk dicocokkan. Menambahkan genre atau sinopsis akan mengatasi hal ini, tetapi Book-Crossing tidak menyediakannya.

**Problem statement 2, memprediksi kemungkinan pengguna menyukai buku yang belum pernah dinilai.**

Tujuan ini tercapai secara teknis. Sistem menghasilkan skor untuk seluruh 3.694.600 pasangan pengguna dan buku, lalu menyusun sepuluh rekomendasi teratas dari buku yang belum pernah dinilai pengguna bersangkutan. RMSE pada data validasi sebesar 1,5066 dan MAE sebesar 1,1303, yang berarti prediksi rata-rata meleset sekitar 1,13 poin dari penilaian sebenarnya.

Besar galat tersebut perlu dibaca terhadap sebaran ratingnya. Rentang efektif rating pada data ini sekitar 5 sampai 10, jadi kesalahan 1,13 poin cukup besar untuk mengaburkan perbedaan antara buku yang disukai biasa dan buku yang sangat disukai. Sistem ini layak dipakai untuk menyaring kandidat, tetapi belum layak dijadikan dasar tunggal penyusunan urutan.

**Problem statement 3, seberapa besar sumbangan pemodelan dibanding tebakan sederhana.**

Jawabannya kecil, dan ini temuan terpenting proyek. Menebak rata-rata global menghasilkan RMSE 1,7215. Menambahkan bias pengguna dan bias buku menurunkannya menjadi 1,5091, sebuah perbaikan sebesar 0,2124. Menambahkan matrix factorization di atasnya hanya menurunkan lagi sebesar 0,0025, menjadi 1,5066. Pola yang sama terlihat pada MAE.

Artinya, hampir seluruh daya prediksi berasal dari mengetahui siapa penilai yang murah hati dan buku mana yang umumnya disukai orang, bukan dari mencocokkan selera antar pengguna. Faktorisasi yang seharusnya menjadi inti collaborative filtering praktis tidak memberi tambahan apa pun. Daftar top-N yang dihasilkan memperkuat kesimpulan ini, karena isinya didominasi buku bergambar dan komik yang hampir selalu menerima rating tinggi dari siapa pun, bukan buku yang cocok dengan selera spesifik pengguna tersebut.

Penyebab yang paling mungkin adalah kepadatan data. Meski sudah disaring sampai ambang 10, matriksnya hanya terisi 1,122 persen, dan 50 komponen SVD hanya menjelaskan 25,19 persen ragam residu. Pada tingkat kelangkaan seperti ini pola selera bersama antar pengguna terlalu tipis untuk dipisahkan dari derau. Menaikkan jumlah komponen tidak memperbaiki galat validasi, tanda bahwa tambahan komponen hanya menangkap derau.

**Catatan atas pencapaian keseluruhan.**

Perbandingan terhadap tolok ukur adalah bagian yang paling menentukan pada evaluasi ini. RMSE 1,5066 jika berdiri sendiri terdengar seperti model yang bekerja, padahal 98,8 persen perbaikannya berasal dari perhitungan rata-rata sederhana yang tidak memerlukan pembelajaran mesin sama sekali. Tanpa pembanding, angka itu akan menyesatkan.

Peningkatan lebih lanjut kemungkinan besar tidak datang dari pergantian algoritma. Yang dibutuhkan adalah data yang lebih padat, misalnya dengan memakai umpan balik implisit yang jumlahnya 716.109 baris dan sengaja dibuang di awal, atau metadata yang lebih kaya seperti genre dan sinopsis untuk membangun sistem hibrida yang menggabungkan kedua pendekatan.

## Referensi

[1] C.-N. Ziegler, S. M. McNee, J. A. Konstan, dan G. Lausen, "Improving Recommendation Lists Through Topic Diversification," dalam *Proceedings of the 14th International Conference on World Wide Web (WWW '05)*, Chiba, Jepang, 2005, hlm. 22-32.

[2] F. Ricci, L. Rokach, B. Shapira, dan P. B. Kantor, *Recommender Systems Handbook*. Boston, MA, USA, Springer, 2011.

[3] Y. Koren, R. Bell, dan C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," *Computer*, vol. 42, no. 8, hlm. 30-37, Agustus 2009.

[4] S. S. Iyengar dan M. R. Lepper, "When Choice is Demotivating, Can One Desire Too Much of a Good Thing?," *Journal of Personality and Social Psychology*, vol. 79, no. 6, hlm. 995-1006, 2000.

[5] International Publishers Association, *IPA Annual Report, ISBN Statistics by Country*, 2022.
