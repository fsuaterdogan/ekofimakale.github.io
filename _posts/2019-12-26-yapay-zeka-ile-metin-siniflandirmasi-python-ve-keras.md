---
title: "Yapay Zeka ile Metin Duygu Analizi - Python ve Keras"
categories:
  - Yazılım
header:
  teaser: https://images.unsplash.com/photo-1531715047058-33b6c9df7897?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1267&q=80
---
![Yapay Zeka ile Metin Duygu Analizi](https://images.unsplash.com/photo-1531715047058-33b6c9df7897?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1267&q=80)

İçerikler
-
- Veri Kümesi Seçme
- Temel Model Tanımlama
- (Derin) Yapay Sinir Ağları Üzerine Bir Astar
- Keras Tanıtımı
- Keras'ı Yükleme
- İlk Keras Modeliniz
- Sözcük Yerleştirme Nedir?
- Tek Sıcak Kodlama
- Word Yerleştirme
- Keras Gömme Katmanı
- Ön Hazır Word Gömülerini Kullanma
- Evrişimli Sinir Ağları (CNN)
- Hiperparametreler Optimizasyonu
- Sonuç

Projenizdeki insanların ruh halini bildiğinizi düşünün. Bu öğreticiden sonra, bunu yapacak donanıma sahip olacaksınız. Bunu yaparken, (derin) sinir ağlarının mevcut gelişmelerini ve metne nasıl uygulanabileceğini kavrayacaksınız.

Makine öğrenimi ile metinden ruh halini okumaya duyarlılık analizi (sentiment analysis) denir ve metin sınıflandırmasında öne çıkan kullanım durumlarından biridir. Bu, doğal dil işlemenin (NLP) çok aktif bir araştırma alanına girmektedir. Metin sınıflandırmasının diğer yaygın kullanım örnekleri arasında spam tespiti, müşteri sorgularının otomatik olarak etiketlenmesi ve metnin tanımlı konulara ayrılması yer alır. Peki bunu nasıl yapabilirsiniz?

Veri Kümesi Seçmek
-
Maalesef, Türkçe bir veri seti yok. İngilizce olarak UCI Machine Learning Depository'u indirebilirsiniz. Veri kümesi IMDb, Amazon ve Yelp'den etiketli incelemeler içeriyor. Olumsuz bir duygu için 0 ve olumlu bir duygu için 1 ile işaretlendi.

Biz bu yazıda sadece IMDb verilerini ele aldık.

Dosyayı bir data klasörüne aktarın ve [Pandas](https://pandas.pydata.org/) ile çağırın:
```
import pandas as pd

sozluk = {'yelp':   'data/sentiment_analysis/yelp_labelled.txt',
                 'amazon': 'data/sentiment_analysis/amazon_cells_labelled.txt',
                 'imdb':   'data/sentiment_analysis/imdb_labelled.txt'}

df_list = []
for kaynak, dosyayolu in sozluk.items(): #Sözlükteki verilerin kaynak ve dosya yolu için
    df = pd.read_csv(dosyayolu, names=['sentence', 'label'], sep='\t') #Dosya yolunu oku ve sütunları isimlendir, \t yani boşluk noktasında ayır
    df['source'] = source  # Kaynak adlı sütun ekler
    df_list.append(df) #df_list'e df yi ekle

df = pd.concat(df_list)
print(df.iloc[0])
```
print(df.iloc[0]) çıktısı:

```
sentence    A very, very, very slow-moving, aimless movie ...
label                                                       0
Name: 0, dtype: object
```

Bu veri seti ile, bir cümlenin duygu analizini tahmin etmek için bir model eğitebilirsiniz. Bunu yapmanın bir yolu, her cümledeki her kelimenin sıklığını saymak ve bu sayımı veri kümesindeki tüm sözcük kümesine geri bağlamaktır. Verileri alarak ve tüm cümlelerdeki tüm kelimelerden bir kelime hazinesi oluşturarak başlayacaksınız. Metin koleksiyonuna NLP'de korpus da denir.

Hemen örnek verelim. Aşağıdaki iki cümleniz olduğunu düşünün:

```
sentences = ['John likes ice cream', 'John hates chocolate.']
```
Cümleleri vektörleştirmek için scikit-learn kütüphanesi tarafından sağlanan CountVectorizer'ı kullanacağız. Her cümlenin kelimelerini alır ve cümledeki tüm benzersiz kelimelerin bir sözlüğünü oluşturur. Bu kelime daha sonra kelimelerin sayımının bir özellik vektörü oluşturmak için kullanılabilir:

```
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.vocabulary_ 
{'John': 0, 'chocolate': 1, 'cream': 2, 'hates': 3, 'ice': 4, 'likes': 5} #Çıktı
```

Önceki iki cümleyi alıp CountVectorizer ile dönüştürdüğünüzde, cümlenin her kelimesinin sayısını temsil eden bir vektör elde edersiniz:

```
vectorizer.transform(sentences).toarray()
array([[1, 0, 1, 0, 1, 1],
       [1, 1, 0, 1, 0, 0]])
```

Örneğin, yukarıdaki ilk öğeye bakarsanız, her iki vektörün de 1 olduğunu görebilirsiniz. Bu, iki cümlenin de John oluşumunu taşıdığı anlamına gelir.

Bu, Kelime Torbası (Bag-of-words) modeli olarak kabul edilir. Her belge bir vektör olarak temsil edilir. Bu vektörleri artık bir makine öğrenme modeli için özellik vektörleri olarak kullanabilirsiniz. Bu, bizi bir temel model belirleyerek bir sonraki bölümümüze götürür.

Temel Model Tanımlama
-
Makine öğrenimi ile çalışırken, önemli bir adım temel bir model tanımlamaktır. <u>Temel modeli, daha sonra test etmek istediğiniz daha gelişmiş modellerle karşılaştırma olarak kullanacaksınız.</u>

İlk olarak, verileri doğruluğu değerlendirmenize ve modelinizin iyi genelleşip genelleşmediğine bakmanıza olanak tanıyan bir eğitim ve test setine bölebilirsiniz. Bu, modelin daha önce görmediği veriler üzerinde iyi performans gösterip gösteremeyeceği anlamına gelir.

Aşırı uydurma, bir modelin eğitim verileri üzerinde çok iyi eğitilmiş olmasıdır. Aşırı uyumdan kaçınmak istersiniz, <u>çünkü bu modelin çoğunlukla eğitim verilerini ezberlediği anlamına gelir</u>. Bu, egzersiz verilerinde büyük bir doğruluk, ancak test verilerinde düşük bir doğruluk anlamına gelir.

Birleştirilmiş veri setimizden çıkardığımız Yelp veri setini alarak başlıyoruz. Oradan cümleleri ve etiketleri alıyoruz. .Values, bu bağlamda çalışması daha kolay olan bir Pandas Series nesnesi yerine bir NumPy dizisi döndürür:

```
from sklearn.model_selection import train_test_split

df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
```

Burada cümleleri vektörleştirmek için önceki BOW modelinde tekrar kullanacağız. Bu görev için CountVectorizer'ı tekrar kullanabilirsiniz. Egzersiz sırasında test verileri bulunmayabileceğinden, yalnızca egzersiz verilerini kullanarak kelime haznesi oluşturabilirsiniz. Bu kelimeleri kullanarak eğitim ve test setinin her bir cümlesine yönelik özellik vektörleri oluşturabilirsiniz:

```
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
```

Ortaya çıkan özellik vektörlerinin, tren testi bölünmesinden sonra sahip olduğumuz eğitim örneği sayısı olan 750 örneğe sahip olduğunu görebilirsiniz. Her örnek, kelime büyüklüğünde 1714 boyuta sahiptir. Ayrıca, seyrek bir matris aldığımızı görebilirsiniz. Bu, yalnızca sıfır olmayan elemanlara sahip matrisler için optimize edilmiş ve sadece sıfır olmayan elemanların kaydını yükleyen hafıza yükünü azaltan bir veri türüdür.

CountVectorizer, cümleleri daha önce kelime bilgisinde gördüğünüz gibi bir dizi jetona ayıran tokenizasyon gerçekleştirir. Ayrıca noktalama işaretlerini ve özel karakterleri kaldırır ve her kelimeye başka ön işlemler uygulayabilir. İsterseniz, CountVectorizer ile NLTK kitaplığından özel bir belirteç kullanabilir veya modelinizin performansını artırmak için keşfedebileceğiniz herhangi bir sayıda özelleştirmeyi kullanabilirsiniz.

Kullanacağımız sınıflandırma modeli, matematiksel olarak aslında girdi özelliği vektörüne dayanarak 0 ile 1 arasında bir gerileme biçimini ifade eden basit ama güçlü bir doğrusal model olan lojistik regresyon. Bir kesme değeri (varsayılan olarak 0,5) belirtilerek, sınıflandırma için regresyon modeli kullanılır. LogisticRegression sınıflandırıcısını sağlayan scikit-learn kütüphanesini tekrar kullanabilirsiniz:

```
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score)
Accuracy: 0.796
```

Lojistik regresyonun% 79,6'lık etkileyici bir seviyeye ulaştığını görebilirsiniz, ancak bu modelin sahip olduğumuz diğer veri kümelerinde nasıl performans gösterdiğine bir göz atalım. Bu komut dosyasında, sahip olduğumuz her veri kümesi için tüm işlemi gerçekleştirir ve değerlendiririz:

```
for source in df['source'].unique():
    df_source = df[df['source'] == source]
    sentences = df_source['sentence'].values
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print('{} verisi için doğruluk payı: {:.4f}'.format(source, score))
```

İşte sonuçlar:

```
yelp verisi için doğruluk payı: 0.7960
amazon verisi için doğruluk payı: 0.7960
imdb verisi için doğruluk payı: 0.7487
```

(Derin) Yapay Sinir Ağları Üzerine Bir Astar
-
