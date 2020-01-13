---
title: "İşte Uber'in Trafik Sistemi: Kümeleme ve K-Means Algoritması"
categories:
  - Yazılım
header:
  teaser: https://images.unsplash.com/photo-1465447142348-e9952c393450?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1268&q=80
---
![Araba trafiği](https://images.unsplash.com/photo-1465447142348-e9952c393450?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1268&q=80)

Gartner'a göre, 2020 yılına kadar çeyrek milyar bağlantılı araç IoT'nin önemli bir unsurunu oluşturacak. Bağlı araçların gerçek zamanlı izleme ve uygulamalar sağlamak için analiz edilebilen ve yeni mobilite ve araç kullanımı kavramlarına yol açacak şekilde saatte 25 GB veri üretmesi öngörülmektedir. [Referans: Gartner](https://www.gartner.com/en/newsroom/press-releases/2015-01-26-gartner-says-by-2020-a-quarter-billion-connected-vehicles-will-enable-new-in-vehicle-services-and-automated-driving-capabilities)

Uber ve Makine Öğrenimi Bağlantısı
-
Uber, kârı en üst düzeye çıkarmak için fiyatlandırmayı hesaplamaktan otomobillerin en uygun şekilde konumlandırılmasına kadar makine öğrenimini kullanır. Araç GPS verilerinin analizi ve izlenmesi için genel uber yolculuk veri kümesi kullanıldı.

Uber tarafından New York'tan üretilen verileri içeren Uber gezi veri kümesi. [Veriler FiveThirtyEight üzerinde ücretsiz olarak mevcut](https://data.fivethirtyeight.com/).

Beş ilçesi olan New York City'den veriler: Brooklyn, Queens, Manhattan, Bronx ve Staten Island. Uygulama, Uber'e yapılan seyahatleri anlamak ve New York'taki farklı ilçeleri tanımlamak için bu veri kümesinde Kümeleme anlamına gelir.

Kümeleme, veri kümelerini benzer veri noktalarından oluşan gruplara bölme işlemidir. Kümeleme, etiketlenmemiş verileriniz olduğunda kullanılan bir tür denetimsiz makine öğrenimidir.

Burada, ana amacı benzer öğeleri veya veri noktalarını bir kümede gruplamak olan bir K-Means kümeleme algoritması uyguladık. K-ortalamalarındaki “K”, kümelerin sayısını temsil eder. İnternette, K-Means algoritmasının çalışma prensibini aratarak kontrol edebilirsiniz.

Gerekli kütüphaneleri içe aktarma
-
![Gerekli kütüphaneleri içe aktarma](https://miro.medium.com/max/876/1*3GJQEyicBzzfyv1spz-FcA.png)

CSV okuma
-
![CSV okuma](https://miro.medium.com/max/886/1*CnmbWbNhId3T77yRligvpQ.png)

Çıktı
-
![Çıktı](https://miro.medium.com/max/748/1*6HUeSw3qEfx-8cNA5t2Yag.png)

Veri kümesinde 829.275 gözlem ve dört sütun var. İşte bulunan dört özellik:

- Tarih / Saat: Uber toplayıcının tarihi ve saati.
- Lat (Enlem): Uber alıcısının enlemi
- Lon (Boylam): Uber alıcısının boylamı.
- Temel: Uber pikapına bağlı TLC temel şirket kodu.

Özellik seçmek
-
![Özellik seçmek](https://miro.medium.com/max/877/1*18Yu_OkgkwSILaortDbnzw.png)

Çıktı
-
![Çıktı](https://miro.medium.com/max/344/1*CTrA9nS5s-6PKGBfjG539w.png)

K-Means kümelenmesi uygulanır. İlk adım K için en uygun değeri bulmaktır. Bu, aşağıda gösterildiği gibi dirsek grafiğinden öğrenilebilir.

Çıktı
-
![Çıktı](https://miro.medium.com/max/742/1*Gwkt403ultMXY-sei6OwZA.png)
![Çıktı](https://miro.medium.com/max/880/1*Cc0F40hLApbO4RH8bVGHZg.png)
![Dirsek metodu](https://miro.medium.com/max/1079/1*qTUzSX4EGvjx-cvQE2JRBQ.png)

Yukarıdaki dirsek grafiğinden, en yakın küme ağırlık merkezinden gözlemlerin kare mesafesinin toplamının, kümelerin sayısındaki bir artışla azalmaya devam ettiğini görebiliriz. K = 6 sonrası önemli bir azalma olduğunu görebiliriz. 6 veya 7 kümeden birini seçebiliriz. Bu veri kümesi için 6 tane seçildi.

K-Means Kümeleme Yapma
-
K-Means algoritmasında birkaç küme atama

![K-Means Kümeleme Yapma](https://miro.medium.com/max/878/1*p6zYdAayZkJLyXNur35fNA.png)

Küme Ağırlık Merkezlerini Saklama
![Küme Ağırlık Merkezlerini Saklama](https://miro.medium.com/max/881/1*bF8a8E9Y1Q53y87N_JilUg.png)

Çıktı
-
![Çıktı](https://miro.medium.com/max/415/1*fAIzKHPyi15jPIwtOPUr0Q.png)

Ağırlık merkezlerini görselleştirmek
-
![Ağırlık merkezlerini görselleştirmek](https://miro.medium.com/max/881/1*NOn8tWoOr02pyz-jDptYtQ.png)

Enlem ve boylamları ağırlık merkezlerinden almak ve iki ayrı veri çerçevesine dönüştürmek. Hem veri çerçevesini birleştirdi hem de kolay görselleştirme için "clocation" olarak adlandırdı.

![Çıktı](https://miro.medium.com/max/243/1*9CwLx3QP2W3_J0z7n5doOw.png)
![Çıktı](https://miro.medium.com/max/879/1*1Po7fl9S3UZCjeBEI3KN1w.png)
![Çıktı](https://miro.medium.com/max/993/1*CdDctqcTFeBvMR3vFaPWNw.png)

Yukarıdaki dağılım grafiğinde her bir kümeyle ilgili tüm ağırlık merkezlerini görebiliriz. Ancak, bu anlamlı bir bilgi göstermez. Aynı şeyi Google haritasına (enlem ve boylam) çizelim ve görselleştirelim.

Ağırlık merkezlerini seçmek ve yeri haritalamak için bir folium kütüphanesi kullanıldı.

<script src="https://gist.github.com/sdhilip200/012d4bfed59841ae38b2bedf06b8daf2.js"></script>

![Çıktı](https://miro.medium.com/max/993/1*wXEL9mnZIgKEf2Gfq1EPqA.png)

Altı ağırlık merkezinin hepsinin harita üzerinde çizildiğini görebiliriz. Bu ağırlık merkezleri Uber'e nasıl yardımcı olur?
-
- Uber bu ağırlık merkezlerini merkez olarak kullanabilir. Uber yeni bir sürüş talebi aldığında, bu ağırlık merkezlerlerinin her birinin yakınlığını kontrol edebilirler. Hangi belirli ağırlık merkezi daha yakınsa, Uber aracı o belirli konumdan müşteri konumuna yönlendirebilir.
- Uber'in birçok sürücüsü var ve birçok yere hizmet veriyor. Uber hub'ı (belirli ağırlık merkezi) biliyorsa ve çok fazla sürüş isteği alıyorlarsa, stratejik olarak şoförlerini, sürüş isteği alma olasılığının büyük olduğu iyi bir yere yerleştirebilirler. Bu, araçlar konuma daha yakın yerleştirildiğinden Uber'in müşteriye daha hızlı hizmet etmesine yardımcı olacak ve aynı zamanda işlerini büyütmeye yardımcı olacaktır.
- Uber, bu ağırlık merkezlerini araçlarının en uygun şekilde yerleştirilmesi için kullanabilir. Günün hangi kısmına daha fazla sürüş talebi geldiğini bulabilirler. Örneğin, Uber 11: 00'da ağırlık merkezi 0'dan (küme 1) daha fazla istek alırsa, ancak ağırlık merkezi 3'ten (küme 4) çok daha az talep alırsa, araçları küme 4'ten küme 1'e yönlendirebilir (küme 4'te daha fazla araç varsa).
- Uber, hangi kümelerin maksimum istekler, yoğun zamanlar vb. İle ilgili olduğunu analiz ederek bu ağırlık merkezlerini en uygun fiyatlandırma için kullanabilir. Varsayalım ki, belirli bir konuma gönderilecek çok fazla araç yoksa (daha fazla talep), en uygun fiyatlandırmayı yapabilirler. çünkü talep yüksek ve arz daha az.

Kümeleri Depolama
-
![Kümeleri Depolama](https://miro.medium.com/max/882/1*u2v1P_6fC3UV6YVDEjBOog.png)

Hangi küme maksimum sürüş talebi alır?
-
![Hangi küme maksimum sürüş talebi alır?](https://miro.medium.com/max/878/1*pyFAd4mJMkcCKcGtXFWobQ.png)

![Çıktı](https://miro.medium.com/max/1223/1*fCIwD0FtxlFg6_3N-0xxew.png)
Küme 3, maksimum küme isteğini alır ve ardından küme 1 alır. Uber, daha yüksek talepleri karşılamak için Küme 3'e daha fazla araç yerleştirebilir.

Yeni konum kontrolü
-
![Yeni konum kontrolü](https://miro.medium.com/max/720/0*PUfafKs2Ho1q9EnA)

Uber yeni bir sürüş talebi alırsa (yeni konumlarını boylam ve enlem yoluyla alırken) enlem ve boylam değerini geçerse, o zaman araçtan hangi kümenin gitmesi gerektiğini tahmin eder?
![Yeni lokasyon tahmini](https://miro.medium.com/max/879/1*gUy-BCTBLHHQJ2yZEh7M4Q.png)

```
array([2])
```

Bu durumda, araç küme 2'den gelecektir.

Kaynak: https://towardsdatascience.com/how-does-uber-use-clustering-43b21e3e6b7d
