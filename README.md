# 2024-2025-Bahar-D-nemi-YZM212-Makine-renmesi-Dersi-III.-Laboratuvar-De-erlendirmesi
# Makine Öğrenmesinde Matris Manipülasyonu ve Özdeğer-Özvektör Analizi

### ✅ Matris operasyonları ve eigen-analiz yöntemlerinin makine öğrenmesindeki uygulamalarının incelenmesi

---

## 1. Temel Kavramlar

### 1.1 Matris Manipülasyonu
Veri setleriyle ve model parametreleriyle çalışırken elimizdeki tüm sayıları satır–sütun düzeni (vektör/matris) hâline getiriyoruz. 
Bu matrisler üzerinde toplama, çarpma, transpoz, tersini alma gibi işlemler yapıyor; ayrıca SVD veya özdeğer-özvektör ayrıştırmasıyla "iç yapıyı" dekompoze ediyoruz.

### 1.2 Özdeğer (\( \lambda \))
Bir A matrisinin, belirli bir \( \vec{v} \ne 0 \) vektörünü yalnızca ölçeklendirerek (yönünü değiştirmeden)
\[
A \vec{v} = \lambda \vec{v}
\]
eşitliğini sağladığı skaler sayıdır.

### 1.3 Özvektör (\(\vec{v}\))
Yukarıdaki eşitliği sağlayan, yani matris çarpımında sadece \(\lambda\) katsayısı kadar uzayı genişleten veya daraltan vektördür.

---

## 2. Makine Öğrenmesindeki Rolleri

### 2.1 Boyut İndirme
- **PCA**: Verinin kovaryans matrisini eigendecomposition'a sokup en büyük varyansı taşıyan eksenleri (ana bileşenleri) seçer.
- **Kernel PCA**: Doğrusal olarak ayrılamayan veriyi yüksek boyutlu bir alana geçirip orada PCA uygulayarak doğrusal olmayan ilişkileri yakalar.

### 2.2 Gürültü Azaltma ve Özellik Çıkarımı
- **SVD**: Veri matrisini \( U \Sigma V^T \) formuna ayırır; \( \Sigma \)'daki küçük tekil değerleri atarak gereksiz detayları (gürültüyü) elemeyi sağlar.

### 2.3 Graf Tabanlı Kümeleme
- **Spektral Kümeleme**: Verileri ilişkililik matrisine, oradan da Laplace operatörüne dönüştürür; en küçük özdeğerlere karşılık gelen özvektörleri alıp klasik kümeleme algoritmalarını uygular.

### 2.4 Derin Öğrenmede Hesap Verimliliği
Backpropagation ve ağırlık güncellemeleri, büyük boyutlu matris çarpma/toplamaları üzerinden ilerler; GPU hızlandırmasıyla bu işlemler pratik hale gelir.

### 2.5 Yüz Tanıma (Eigenfaces)
- Eğitim görüntülerinden oluşturulan yüz piksel matrisinin kovaryansını PCA’ya sokar, elde edilen temel yüz bileşenlerini ("eigenface") çıkarır.
- Yeni bir yüz, bu bileşenler uzayında küçük boyutlu bir vektör olarak temsil edilir ve tanıma benzerlik ölçütleriyle yapılır.

---

## 3. Yöntemler ve Uygulama Alanları

| Yöntem             | Temel İşlem                                | Nerede Kullanılır                        |
|---------------------|-----------------------------------------------|---------------------------------------------|
| **PCA**             | Kovaryans matrisini ayrıştırmak              | Boyut indirme, görselleştirme              |
| **Kernel PCA**      | Veriyi karmaşık bir alana taşıyıp ayrıştırmak   | Doğrusal olmayan veri yapıları            |
| **SVD**             | Matrisi temel parçalara (tekil değerlere) ayırmak | Gürültü filtresi, low-rank approx.         |
| **Spektral Kümeleme** | Graf Laplace operatörünü ayrıştırmak        | Karmaşık ağlarda küme bulma              |
| **Eigenfaces**      | Yüz kovaryans matrisine PCA uygulamak         | Yüz tanıma, kimlik doğrulama              |

---

## 4. NumPy `linalg.eig` Fonksiyonunun Dokümantasyonu

### 1. Ne İşe Yarar?
\( A \vec{v}_i = \lambda_i \vec{v}_i \) eşitliğini sağlayan:
- Özdeğerler \(\lambda_i\)
- Sağ özvektörler \(\vec{v}_i\)
için hesaplama yapar.

### 2. Fonksiyon İmzası
```python
eigenvalues, eigenvectors = np.linalg.eig(a)
```

- **Parametre:**
  - `a` : Kare matris
- **Dönüş:**
  - `eigenvalues`: \( \lambda_1, \ldots, \lambda_M \)
  - `eigenvectors`: Her sütun bir \( \vec{v}_i \)
- **Hata:** Yakınsamama durumunda `LinAlgError`

### 3. Python Seviyesindeki Adımlar
1. **Girdi Hazırlığı:** `makearray(a)` 
2. **Karelik Kontrolü:** `assert_stacked_square(a)` 
3. **Signature Seçimi:** `commonType`
4. **Ufunc Çağrısı:** `_umath_linalg.eig`
5. **Fortran Entegrasyonu:** `dgeev`/`zgeev`
6. **Sonuç Paketleme:** `EigResult` ile wrap

---

## 5. Saf Python Özdeğer Hesaplama Adımları

- `get_dimensions(matrix)`
- `find_determinant(matrix)` — kofaktör açılımı
- `list_multiply()` ve `list_add()` — karakteristik denklem
- `identity_matrix()` — birim matris üretimi
- `characteristic_equation()` — \(A - \lambda I\) 
- `determinant_equation()` — karakteristik polinom
- `find_eigenvalues()` — \(numpy.roots\) ile çözüm

---

## 6. Karşılaştırma
NumPy ve saf Python aynı özdeğer kümesini döndürmüş, hesaplama farklılıkları analiz edilmiştir.

---

## 7. Kaynakça
- Burden, R.L. & Faires, J.D. (2011). *Numerical Analysis*. Cengage Learning.
- Strang, G. (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
- Murphy, K.P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
- https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
