# Laporan Proyek Machine Learning -Bayuzen Ahmad
---
## Domain Proyek
---
<p align="justify">Sistem rekomendasi buku adalah jenis sistem rekomendasi di mana kita harus merekomendasikan buku serupa kepada pembaca berdasarkan minatnya dan juga merekomendasikan buku berdasarkan rating tertinggi.
Dengan adanya sistem rekomenasi selain mempermudah para customer mencari buku sesuai keinginan dan buku yang memiliki rating tertinggi, sistem rekomendasi juga akan meningkatkan kepuasaan user sehingga berdampak pada profit yang akan didapatkan.<br>

---
# Business Understanding
---
Pengembangan dan pengambungan sistem rekomendasi berdasarkan content based filtering dan model rekomendasi K-NN dengan memberikan rokemendasi buku berdasarkan penulis dengan rating tertinggi dan berdasarkan catatan pencarian buku user dimasa lalu.<br>
## Problem Statements
1.Bagaimana perseberan rating buku ? <br>
2.Buku apa saja yang memiliki rating tertinggi berdasarkan penulis buku sehingga dapat membangun sistem rekomendasi yang menampilkan buku rating tertinggi dari setiap penulis buku ?<br>
## Goals
1.Mengetahui persebaran rating buku. <br>
2.Membangun sistem rekomendasi berdasarkan rating tertinggi setiap buku dari setiap penulis buku.<br>

## Solution
  Menggunakan algoritma content based filtering Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.<br>

 ---
 
 # Data Understanding
 ---
  Masalah yang dijelaskan pada dataset mengharuskan dilakukan cleansing karena dataset masih belum bersih, dilakukan penggabungan data karena data terdiri dari tiga dataset yaitu book.csv,Rating.csv, dan User.csv, dan perlu dilakukan transformasi pada dataset untuk mendapatkan model rekomendasi yang baik. [Sumber Dataset](https://www.kaggle.com/arashnic/book-recommendation-dataset) <br>
 - variabel pada dataset : <br>
  **User** <br>
Berisi pengguna. Perhatikan bahwa ID pengguna (User-ID) telah dianonimkan dan dipetakan ke bilangan bulat. Data demografis disediakan (Lokasi, Usia) jika tersedia. Jika tidak, bidang ini berisi nilai NULL.<br>
**Book** <br>
Buku diidentifikasi oleh ISBN masing-masing. ISBN yang tidak valid telah dihapus dari set data. Selain itu, beberapa informasi berbasis konten diberikan (Judul Buku, Penulis Buku, Tahun Penerbitan, Penerbit), diperoleh dari Amazon Web Services. Perhatikan bahwa dalam kasus beberapa penulis, hanya yang pertama yang disediakan. URL yang menautkan ke gambar sampul juga diberikan, muncul dalam tiga rasa berbeda (Image-URL-S, Image-URL-M, Image-URL-L), yaitu kecil, sedang, besar. URL ini mengarah ke situs web Amazon.<br>
**Rating** <br>
Berisi informasi peringkat buku. Peringkat (Penilaian Buku) bersifat eksplisit, dinyatakan dalam skala 1-10 (nilai yang lebih tinggi menunjukkan apresiasi yang lebih tinggi), atau implisit, yang dinyatakan dengan 0. <br>
  
**Data Overview**
  melihat 5 baris pertama pada tiap data set. <br>
  **books.csv** <br>
  ![image](https://user-images.githubusercontent.com/88529383/147850812-f662236c-87b5-476c-8a94-92d93fa68adb.png) <br>
  
 **Rating.csv** <br>
  ![image](https://user-images.githubusercontent.com/88529383/147850832-3670db93-eec7-4e94-9787-074e43bafaef.png) <br>
 
  **User.csv** <br>
  ![image](https://user-images.githubusercontent.com/88529383/147850856-c8044710-b186-4054-aaf2-1ac938c23ed3.png) <br>
  pada proses pembuatan sistem rekomendasi variabel yang digunakan adalah ISBN,Book-Title,Book-Author,Year-Of-Publication dan Publisher,user-id, dan location. <br>
  
  # Eksplonatory Data Analysis
  ---
  - Persebaran rating buku <br>
  ![image](https://user-images.githubusercontent.com/88529383/147850977-68fbadb6-2e0c-4f19-a06f-f889dabd1c3e.png) <br>
  banyak buku yang memiliki rating 0 dan lumayan sedikit buku yang memiliki rating dengan rentang 7 hingga 10. <br>
  
  - Publikasi Terbanyak Berdasarkan Tahun Terbitan <br>
  ![image](https://user-images.githubusercontent.com/88529383/147851003-67c1c481-a0fe-416c-b972-deef5d44a809.png) <br>
  Buku terbitan tahun 1996 merupakan yang terbanyak memiliki publikasi disusul dengan tahun 2002 dan 1999.<br>
  
  - Penulis Buku Dengan Publikasi Terbanyak <br>
  ![image](https://user-images.githubusercontent.com/88529383/147851015-fdd9e552-8b54-437e-8d4b-ab142f6d54f3.png) <br>
  Agatha Cristie merupakan penulis yang paling banyak menerbitkan buku disusul dengan William Shakespeare dan Stephen King. <br>
  
  - Publisher Dengan Publikasi Terbanyak <br>
  ![image](https://user-images.githubusercontent.com/88529383/147851030-67f45c80-7685-41a4-bc68-1c7ce427f90d.png) <br>
Ballatine Books adalah publisher dengan jumlah publikasi terbanyak disusul dengan publisher pocket <br>
  
  - Buku Dengan Penjualan Terbanyak <br>
  ![image](https://user-images.githubusercontent.com/88529383/147851062-8f5a8a64-bb72-4d41-80e4-1d9cc7512797.png) <br>
  Buku yang paling banyak terjual adalah buku Wild Animus dengan total penjulan mencapai 2500 buku. <br>
  
  - Negara Dengan User Terbanyak <br>
  ![image](https://user-images.githubusercontent.com/88529383/147851081-443f218d-bb99-4c49-8c97-dbe707834865.png) <br>
  Amerika memiliki pengguna terbanyak dengan total pengguna mencapai sekitar 140000 <br>
  
  # Sistem Rekomendasi
## Rekomendasi Berdasarkan Author
  ```
  @interact
  def recommend_books_on_author(author_name = list(df['Book-Author'].value_counts().index)):
  a = df[df['Book-Author']==author_name][['Book-Title','Book-Rating']]
  a = a.sort_values(by = 'Book-Rating',ascending=False)
  return a.head(10)
  ```
  ![image](https://user-images.githubusercontent.com/88529383/147851154-3ffdd7aa-15b1-4ba5-803a-f66b942768f4.png) <br>
  sistem rekomendasi menunjukkan bahwa jika mencari penulis buku stephen king maka akan direkomendasikan buku rating tertinggi dari stephen king. <br>
  
  ## Rekomendasi Berdasarkan K-NN
  model rekomendasi K-NN memiliki beberapa keuntungan yaitu Algoritma ML paling sederhana,Non-parametrik dan lazy in nature, Non-parametrik- tidak ada asumsi untuk distribusi,tidak memerlukan titik data training, data training digunakan dalam fase test model. <br>
pada model ini menggunakan metrik berupa judul buku untuk mencari kesamaan antara buku dan juga rating buku. <br>
  ![image](https://user-images.githubusercontent.com/88529383/147851380-e2c20287-9542-4de2-8271-31cb0ae6ed72.png)
diatas adalah tampilan data ketika telah diprocessing. <br>
  ```
  model = neighbors.NearestNeighbors(n_neighbors=6,algorithm='brute',metric='cosine')
model.fit(feature_scale)
dist, idlist = model.kneighbors(feature_scale)
  def book_recommender(book_name=list(df['Book-Title'].value_counts().index)):
  book_list_name = []
  book_id = df[df['Book-Title']==book_name].index
  book_id = book_id[0]
  for newid in idlist[book_id]:
    book_list_name.append(df.loc[newid]['Book-Title'])
  return book_list_name
  book_recommender("There's a Bat in Bunk Five")
  ```
  Rekomendasi Model K-NN akan menampilkan rekomdasi berdasarkan tingkat kesamaan antara buku satu dengan buku lainnya dengan memanfaatkan catatan buku yang paling sering dicari oleh para customer. Jika customer membeli buku dimasa lalu dengan judul **Wild Animus** maka sistem akan memberikan rekomendasi buku **Strega** karena memiliki kemiripan dengan buku **Wild Animus**














