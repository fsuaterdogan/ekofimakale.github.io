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

(Derin) Yapay Sinir Ağları Üzerine Bir Sentez
-

AI araştırmacıları, AI'nın insan seviyesindeki performansı aşacağında birbirleriyle anlaşamadıklarını kabul ettiler. Bu makaleye göre hala biraz zamanımız olmalı.

Böylece sinir ağlarının nasıl çalıştığını merak ediyor olabilirsiniz. Sinir ağlarına zaten aşinaysanız, Keras'ı içeren bölümlere atlamaktan çekinmeyin.

Bu makalede, tekillik konusunda endişelenmenize gerek yok, ancak (derin) sinir ağları yapay zekadaki son gelişmelerde çok önemli bir rol oynuyor. Her şey 2012'de ünlü ImageNet Challenge'daki önceki tüm modellerden daha iyi performans gösteren Geoffrey Hinton ve ekibi tarafından ünlü bir kağıtla başladı.

Geoffrey Hinton ve ekibi, bu eğitimde de ele alacağımız evrişimli bir sinir ağı (CNN) kullanarak önceki modelleri yenmeyi başardı.

O zamandan beri sinir ağları sınıflandırma, regresyon ve hatta üretken modelleri içeren çeşitli alanlara taşındı. En yaygın alanlar bilgisayarla görme, ses tanıma ve doğal dil işlemeyi (NLP) içerir.

Sinir ağları veya bazen yapay sinir ağı (YSA) veya ileri beslemeli sinir ağı olarak adlandırılan, insan beynindeki sinir ağlarından belirsiz bir şekilde esinlenen hesaplama ağlarıdır. Aşağıdaki grafikte olduğu gibi bağlı nöronlardan (düğüm olarak da adlandırılır) oluşurlar.

Özellik vektörlerinizde beslediğiniz bir giriş nöronları katmanı ile başlarsınız ve değerler daha sonra gizli bir katmana iletilir. Her bağlantıda, değeri ileri doğru besliyorsunuz, değer bir ağırlıkla çarpılır ve değere bir sapma eklenir. Bu her bağlantıda olur ve sonunda bir veya daha fazla çıkış düğümü olan bir çıkış katmanına ulaşırsınız.

İkili bir sınıflandırmaya sahip olmak istiyorsanız, bir düğümü kullanabilirsiniz, ancak birden fazla kategoriniz varsa, her kategori için birden çok düğüm kullanmalısınız:

![Yapay sinir ağları](https://files.realpython.com/media/ANN.a2284c5d07a3.png)

İstediğiniz kadar gizli katmanınız olabilir. Aslında, birden fazla gizli katmanı olan bir sinir ağı, derin bir sinir ağı olarak kabul edilir. Endişelenmeyin: Burada sinir ağları ile ilgili matematiksel derinliklere girmeyeceğiz. Ancak, ilgili matematik hakkında sezgisel bir görsel anlayış elde etmek istiyorsanız, Grant Sanderson'ın YouTube Oynatma Listesine göz atabilirsiniz.

Karşınızda Keras
-
Keras, François Chollet'in Tensorflow (Google), Theano veya CNTK (Microsoft) üzerinde çalışabilen derin bir öğrenme ve sinir ağları API'ıdır. Python ile Derin Öğrenme François Chollet'in harika kitabını alıntılamak için:

Keras, derin öğrenme modelleri geliştirmek için üst düzey yapı taşları sağlayan, model düzeyinde bir kütüphanedir. Tensör manipülasyonu ve farklılaşması gibi düşük seviyeli işlemleri ele almaz. Bunun yerine, Keras'ın arka uç motoru olarak hizmet veren özel, iyi optimize edilmiş bir tensör kütüphanesine güvenir.

Her katmanı ve parçayı kendi başınıza uygulamak zorunda kalmadan sinir ağlarını denemeye başlamak için harika bir yoldur. Örneğin, Tensorflow harika bir makine öğrenimi kütüphanesidir, ancak bir modelin çalışması için çok sayıda kaynak plakası uygulamanız gerekir.

Keras Kurulumu
-
```
$ pip install keras
```

İlk Keras Modelimiz
-
```
from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

Modelin eğitimine başlamadan önce, öğrenme sürecini yapılandırmanız gerekir. Bu işlem, .compile() yöntemi ile yapılır. Bu yöntem, optimize edici ve kayıp işlevini belirtir.

Ayrıca, daha sonra değerlendirme için kullanılabilecek bir metrik listesi ekleyebilirsiniz, ancak bunlar eğitimi etkilemez. Bu durumda, ikili çapraz entropiyi ve Adam optimizatörünü kullanmak istiyoruz. Keras ayrıca modele ve eğitim için mevcut parametrelerin sayısına genel bir bakış sağlamak için kullanışlı bir .summary() işlevi içerir:

```
model.compile(loss='binary_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy'])
model.summary()

Sonuç:
_________________________________________________________________
Layer (type)                 Output Shape          Param #   
=================================================================
dense_1 (Dense)              (None, 10)            17150     
_________________________________________________________________
dense_2 (Dense)              (None, 1)             11        
=================================================================
Total params: 17,161
Trainable params: 17,161
Non-trainable params: 0
_________________________________________________________________
```

İlk katman için 8575, bir sonrakinde 6 başka parametremizin olduğunu fark edebilirsiniz. Bunlar nereden geldi?

Her özellik vektörü için 1714 boyutumuz var ve sonra 5 düğümümüz var. Her özellik boyutu ve 1714 * 5 = 8570 parametrelerini oluşturan her düğüm için ağırlıklara ihtiyacımız var ve sonra her düğüm için 5 kat daha fazla bir önyargıya sahibiz, bu da bize 8575 parametrelerini getirir. Son düğümde, başka bir 5 ağırlığımız ve bir önyargı var, bu da bizi 6 parametreye götürüyor.

Temiz! Neredeyse bitti. Şimdi egzersizinize .fit() işlevi ile başlamanın zamanı geldi.

Sinir ağlarındaki eğitim yinelemeli bir süreç olduğundan, eğitim tamamlandıktan sonra durmaz. Modelin eğitim almasını istediğiniz yineleme sayısını belirtmeniz gerekir. Tamamlanan yinelemelere genellikle çağ (epochs) denir. Her çağdan sonra eğitim kaybı ve doğruluğunun nasıl değiştiğini görebilmek için 100 dönem boyunca çalıştırmak istiyoruz.

Seçiminiz için bir diğer parametre toplu iş boyutudur. Parti boyutu bir dönemde kaç örnek kullanmak istediğimizden sorumludur, yani bir ileri / geri geçişte kaç örnek kullanılır. Bu, daha az çağa ihtiyaç duyduğu için hesaplama hızını arttırır, ancak daha fazla belleğe ihtiyaç duyar ve model daha büyük toplu boyutlarla bozulabilir. Küçük bir eğitim setimiz olduğundan, bunu düşük bir parti boyutuna bırakabiliriz:

```
history = model.fit(X_train, y_train,
                     epochs=100,
                     verbose=False,
                     validation_data=(X_test, y_test)
                     batch_size=10)
```

Şimdi modelin doğruluğunu ölçmek için .evaluate() yöntemini kullanabilirsiniz. Bunu hem egzersiz verileri hem de test verileri için yapabilirsiniz. Egzersiz verilerinin test verileri için olduğundan daha yüksek bir doğruluğa sahip olmasını bekliyoruz. Daha uzun bir sinir ağı eğitirseniz, aşırı takılmaya başlaması daha olasıdır.

.fit() yöntemini yeniden çalıştırırsanız, önceki eğitimden hesaplanan ağırlıklarla başlayacağınızı unutmayın. Modeli tekrar eğitmeye başlamadan önce modeli tekrar derlediğinizden emin olun. Şimdi doğruluk modelini değerlendirelim:

```
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

Sonuç:
Training Accuracy: 1.0000
Testing Accuracy:  0.7960
```

Modelin, eğitim seti için %100 doğruluğa ulaştığından fazla takıldığını görebilirsiniz. Ancak, bu model için epoch sayısı oldukça fazla olduğu için bu bekleniyordu. Bununla birlikte, test setinin doğruluğu, ilerlememiz açısından büyük bir adım olan BOW modeliyle önceki lojistik Regresyonumuzu zaten aştı.

Hayatınızı kolaylaştırmak için, Geçmiş geri aramasına dayalı eğitim ve test verilerinin kaybını ve doğruluğunu görselleştirmek için bu küçük yardımcı işlevini kullanabilirsiniz. Her Keras modeline otomatik olarak uygulanan bu geri arama, .fit() yöntemine eklenebilecek kaybı ve ek metrikleri kaydeder. Bu durumda, sadece doğrulukla ilgileniyoruz. Bu yardımcı işlev matplotlib çizim kütüphanesini kullanır:

```
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
```

Bu işlevi kullanmak için, geçmiş sözlüğü içinde toplanan doğruluk ve kayıpla plot_history() öğesini çağırmanız yeterlidir:

![plot_history()](https://files.realpython.com/media/loss-accuracy-baseline-model.ed95465579bd.png)

Eğitim seti %100 doğruluğa ulaştığından modelimizi çok uzun süredir eğittiğimizi görebilirsiniz. Modelin ne zaman takılmaya başladığını görmenin iyi bir yolu, doğrulama verilerinin kaybının tekrar artmaya başlamasıdır. Bu, modeli durdurmak için iyi bir nokta olma eğilimindedir. Bunu bu eğitimde yaklaşık 20-40 dönem görebilirsiniz.

<p class="notice--info"><strong>Not: </strong><a>Sinir ağlarını eğitirken ayrı bir test ve doğrulama seti kullanmalısınız. Genellikle yapacağınız en yüksek doğrulama doğruluğuna sahip modeli almak ve daha sonra modeli test setiyle test etmektir.

Bu, modele uymamanızı sağlar. En iyi modeli seçmek için doğrulama setini kullanmak, yüzlerce içinden en iyi test sonucunu üreten sonucu almak için bir veri sızıntısı (veya "hile") biçimidir. Veri kaçağı, eğitim veri seti dışındaki bilgiler modelde kullanıldığında meydana gelir.</a></p>

Bu durumda, daha küçük bir numune boyutuna sahip olduğumuz için test ve doğrulama setimiz aynıdır. Daha önce ele aldığımız gibi, (derin) sinir ağları çok sayıda örneğiniz olduğunda en iyi performansı verir. Bir sonraki bölümde, kelimeleri vektör olarak göstermenin farklı bir yolunu göreceksiniz. Bu, kelimeleri yoğun vektörler olarak nasıl temsil edeceğinizi göreceğiniz kelimelerle çalışmanın çok heyecan verici ve güçlü bir yoludur.

Sözcük Yerleştirme Nedir?
-
Metin, hava durumu verilerinde veya finansal verilerinizde bulunan zaman serisi verilerine benzer bir dizi veri türü olarak kabul edilir. Önceki BOW modelinde, bir kelime dizisinin tamamını tek bir özellik vektörü olarak nasıl temsil edeceğinizi gördünüz. Şimdi her bir kelimeyi vektör olarak nasıl temsil edeceğinizi göreceksiniz. Metni vektörleştirmenin çeşitli yolları vardır, örneğin:

- Her kelime ile bir vektör olarak temsil edilen kelimeler
- Her karakterle vektör olarak temsil edilen karakterler
- Bir vektör olarak temsil edilen N-gram kelime / karakter (N-gram, metinde birbirini izleyen birden fazla kelime / karakterden oluşan çakışan gruplardır)

Bu öğreticide, sinir ağlarında metni kullanmanın yaygın yolu olan kelimeleri vektör olarak göstermeyle nasıl başa çıkacağınızı göreceksiniz. Bir kelimeyi vektör olarak göstermenin iki olası yolu, tek etkin kodlama ve sözcük düğünleridir.

One-Hot Encoding
-
Bir kelimeyi bir vektör olarak temsil etmenin ilk yolu, basitçe kodlama adı verilen bir kodlama oluşturmaktır.

Bu şekilde, her kelime için, kelime haznesinde bir nokta olduğu için, bire ayarlanan sözcüğün karşılık gelen noktası hariç, her yerde sıfırlı bir vektör var. Tahmin edebileceğiniz gibi, bu her kelime için oldukça büyük bir vektör olabilir ve kelimeler arasındaki ilişki gibi herhangi bir ek bilgi vermez.

Diyelim ki aşağıdaki örnekteki gibi bir şehirler listeniz var:

```
cities = ['London', 'Berlin', 'Berlin', 'New York', 'London']
>>> cities
['London', 'Berlin', 'Berlin', 'New York', 'London']
Şehir listesini aşağıdaki gibi kategorik tamsayı değerlerine kodlamak için scikit-learn ve LabelEncoder kullanabilirsiniz:

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
city_labels = encoder.fit_transform(cities)
>>> city_labels
array([1, 0, 0, 2, 1])
```

Bu gösterimi kullanarak, scikit-learn tarafından sağlanan OneHotEncoder aracını kullanarak daha önce aldığımız kategorik değerleri bir sıcak kodlanmış sayısal diziye kodlayabiliriz. OneHotEncoder, her bir kategorik değerin ayrı bir satırda olmasını beklediğinden diziyi yeniden şekillendirmeniz gerekir, ardından kodlayıcıyı uygulayabilirsiniz:

```
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
city_labels = city_labels.reshape((5, 1))
encoder.fit_transform(city_labels)
array([[0., 1., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [0., 0., 1.],
       [0., 1., 0.]])
```

Kategorik tamsayı değerinin dizinin 1 ve geri kalanı 0 olduğunu temsil ettiğini görebilirsiniz. Bu genellikle sayısal bir değer olarak temsil edemediğiniz ancak yine de kullanmak istediğiniz kategorik bir özelliğiniz olduğunda kullanılır makine öğreniminde. Bu kodlama için bir kullanım örneği elbette bir metindeki kelimelerdir, ancak en belirgin şekilde kategoriler için kullanılır. Bu kategoriler örneğin şehir, departman veya diğer kategoriler olabilir.

Word Embeddings (Sözcük Gömme)
-
Bu yöntem, sözcükleri, sabit kodlanmış tek sıcak kodlamanın aksine eğitilen yoğun sözcük vektörleri (kelime düğünleri olarak da bilinir) olarak temsil eder. Bu, düğün kelimesinin daha az boyuta daha fazla bilgi topladığı anlamına gelir.

Düğün kelimesinin metni bir insan gibi anlayamadığını, bunun yerine kurumda kullanılan dilin istatistiksel yapısını eşleştirdiklerini unutmayın. Amaçları anlamsal anlamı geometrik bir mekana eşlemek. Bu geometrik boşluğa gömme boşluğu denir.

Bu, gömme alanındaki sayılar veya renkler gibi benzer anlamlara benzer kelimeleri eşler. Gömme sözcükler arasındaki ilişkiyi iyi yakalarsa, vektör aritmetiği gibi şeyler mümkün olmalıdır. Bu çalışma alanındaki ünlü bir örnek, Kral - Erkek + Kadın = Kraliçe'yi haritalama yeteneğidir.

Böyle bir kelimeyi nasıl yerleştirebilirsiniz? Bunun için iki seçeneğiniz var. Bir yol, sinir ağınızın eğitimi sırasında kelime düğünlerinizi eğitmektir. Diğer yol ise, doğrudan modelinizde kullanabileceğiniz önceden hazırlanmış kelime düğünlerini kullanmaktır. Orada, bu kelime düğünlerini eğitim sırasında değiştirmeden bırakma seçeneğiniz var veya bunları da eğitiyorsunuz.

Şimdi verileri, kelime düğünleri tarafından kullanılabilecek bir biçime dönüştürmeniz gerekiyor. Keras, metninizi hazırlamak için kullanabileceğiniz metin önişleme ve dizi önişleme için birkaç kolaylık yöntemi sunar.

Bir metin kümesini tamsayılar listesine ayırabilen Tokenizer yardımcı programı sınıfını kullanarak başlayabilirsiniz. Her tam sayı sözlükteki tüm değeri kodlayan bir değerle eşleşir ve sözlükteki anahtarlar kelime terimlerinin kendisidir. Kelime büyüklüğünü ayarlamaktan sorumlu olan num_words parametresini ekleyebilirsiniz. Daha sonra en yaygın num_words kelimeleri saklanır. Önceki örnekten hazırlanan test ve eğitim verileri var:

```
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(sentences_train[2])
print(X_train[2])
Of all the dishes, the salmon was the best, but all were great.
[11, 43, 1, 171, 1, 283, 3, 1, 47, 26, 43, 24, 22]
```

Dizinleme, metinde en çok kullanılan 1 dizinden sonra gelen sözcükten görebileceğiniz şekilde sıralanır. 0 dizininin ayrıldığını ve hiçbir sözcüğe atanmadığını belirtmek önemlidir. Bu sıfır endeksi, birazdan tanıtacağım dolgu için kullanılır.

Bilinmeyen kelimeler (kelime bilgisinde olmayan kelimeler) Keras'ta word_count + 1 ile belirtilir çünkü bazı bilgileri de tutabilirler. Tokenizer nesnesinin word_index sözlüğüne bakarak her kelimenin dizinini görebilirsiniz:

```
for word in ['the', 'all', 'happy', 'sad']:
     print('{}: {}'.format(word, tokenizer.word_index[word]))
the: 1
all: 43
happy: 320
sad: 450
```

<p class="notice--info"><strong>Not: </strong><a>Bu teknik ile scikit-learn’un CountVectorizer tarafından üretilen X_train arasındaki farka dikkat edin.

CountVectorizer ile, kelime sayımlarının üst üste yığınlanmış vektörleri vardı ve her vektör aynı uzunluktaydı (toplam korpus kelimesinin boyutu). Tokenizer ile, elde edilen vektörler her metnin uzunluğuna eşittir ve sayılar sayıları belirtmez, bunun yerine tokenizer.word_index sözlüğündeki kelime değerlerine karşılık gelir.</a></p>

Sahip olduğumuz bir sorun, her metin dizisinin çoğu durumda farklı kelime uzunluğuna sahip olmasıdır. Buna karşı koymak için, kelime sırasını sıfırlarla dolduran pad_sequence() yöntemini kullanabilirsiniz. Varsayılan olarak, sıfırlar ekler, ancak bunları eklemek istiyoruz. Genellikle sıfırların başına veya sonuna eklemeniz önemli değildir.

Ayrıca, dizilerin ne kadar süreceğini belirtmek için bir maxlen parametresi eklemek istersiniz. Bu, bu sayıyı aşan dizileri keser. Aşağıdaki kodda, dizilerin Keras ile nasıl doldurulacağını görebilirsiniz:

```
from keras.preprocessing.sequence import pad_sequences

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train[0, :])
[  1  10   3 282 739  25   8 208  30  64 459 230  13   1 124   5 231   8
  58   5  67   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0]
```

İlk değerler, önceki örneklerden öğrendiğiniz gibi kelime haznesindeki dizini temsil eder. Ayrıca, oldukça kısa bir cümleniz olduğundan, ortaya çıkan özellik vektörünün çoğunlukla sıfır içerdiğini görebilirsiniz. Bir sonraki bölümde Keras'ta kelime düğünleriyle nasıl çalışacağınızı göreceksiniz.

Keras Embedding Layer (Keras Gömme Katmanı)
-
Bu noktada, verilerimizin hala kodlanmış olduğuna dikkat edin. Keras'a birbirini takip eden görevler yoluyla yeni bir gömme alanı öğrenmesini söylemedik. Şimdi önceden hesaplanan tam sayıları alan ve bunları yoğun bir gömme vektörüne eşleyen Keras'ın Gömme Katmanını kullanabilirsiniz. Aşağıdaki parametrelere ihtiyacınız olacak:

- input_dim: kelime dağarcığının boyutu
- output_dim: yoğun vektörün boyutu
- input_length: dizinin uzunluğu
Gömme katmanıyla artık birkaç seçeneğimiz var. Bunun bir yolu gömme katmanının çıktısını almak ve Yoğun bir katmana takmak olabilir. Bunu yapmak için aralarına Yoğun katman için sıralı girdiyi hazırlayan bir Düzleştir katmanı eklemeniz gerekir:

```
from keras.models import Sequential
from keras import layers

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
```
Sonuçlar:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_8 (Embedding)      (None, 100, 50)           87350     
_________________________________________________________________
flatten_3 (Flatten)          (None, 5000)              0         
_________________________________________________________________
dense_13 (Dense)             (None, 10)                50010     
_________________________________________________________________
dense_14 (Dense)             (None, 1)                 11        
=================================================================
Total params: 137,371
Trainable params: 137,371
Non-trainable params: 0
_________________________________________________________________
```

Artık eğitilecek 87350 yeni parametremizin olduğunu görebilirsiniz. Bu sayı, embedding_dim öğesinin vocab_size kezinden gelir. Gömme tabakasının bu ağırlıkları rastgele ağırlıklar ile başlatılır ve daha sonra eğitim sırasında geri yayılım yoluyla ayarlanır. Bu model kelimeleri girdi vektörleri olarak cümle sırasına göre alır. Aşağıdakilerle eğitebilirsiniz:

```
history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

Sonuçlar: 
Training Accuracy: 0.5100
Testing Accuracy:  0.4600
```

![İlk model için doğruluk ve kayıp](https://files.realpython.com/media/loss-accuracy-first-model.95140204b674.png)
