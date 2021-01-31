## Pilot Yüz. Murat Eren Projesi

![output2](https://user-images.githubusercontent.com/2303854/106377768-81eb2f80-63b0-11eb-94bb-863815496cf0.gif)



2018-2019 yılları arasında PKK ve uzantılarına ait "obje" ve "simgeleri" tespit etmeye çalışıyordum. O dönem YOLOv3 ile 'object detection' ile yaptığım araştırmalarda açık kaynak projelerde 'suç unsuru' tespitine yönelik olarak eğitilmiş modellerin hiç birinde PKK'ya ait çalışma bulamamıştım. Bu sebeple kendim model geliştirmiştim. O dönem sadece sınıflandırma için kullanmıştım farklı alanda ihtiyacı olanlar fikir edinebilir, kullanabilir veya kendilerine göre geliştirebilirler.

# YOLOV3 
YOLOv3 içeriğini 2018-2019 yıllarında oluşturmuştum. Zamanında sınıflandırma için kullanmıştım çok fazla false positive veriyor, training içeriğinide verdim isteyenler kendine göre düzenleyebilir ve tekrar eğitebilir.

# Tensorflow (v1.15.0)
Tensorflow içeriğini 2021'in başında oluşturdum. YOLOv3 göre çok daha başarılı (bu oluşturduğum dataset ve konfigürasyon ile ilgili) daha stable çalışıyor. training içeriğinide verdim isteyenler tensorflow içinde modelleri tekrardan eğitebilir.

# Hatalar :
Zafer İşareti, MLKP ve PJAK için eğitim seti içindeki veri çok az olduğu için dataseti biraz daha düzenlemek gerekiyor. v4 te düzeltebileceğime inanıyorum.
Apo için yüz imzası kullanmadığımız için benzer içerikleri (insan yüzü gibi) false positive verebiliyor.
Silah için kafasına silah dayanmış kişileri örnekledim ve burada azda olsa bazen insan kafasına arkadan silah diyebiliyor. (v4 model için silah'ı sadece şekil olarak işaretleyeceğim )

# Tensorflow Class
APO,HDP,PKK,KCK,YPG,YPJ,PYD,PJAK,MLKP,DHKPC,IBKY,ZI,AK47,SILAH

# YOLOv3 Class
APO,PKK,YPG,PYD,YPJ,YAT,YPS,YDG-H,I-BKY,HDP,ELBOMBASI,SNIPER,RPG-7,RPG-7-MERMI,AK47,M4,TABANCA,ZISARETI,PUSI,KARMASKESI,BIKSI,KUSAK,HAT

# exit(0);
İsteyen istediği gibi kullanabilir... Eğer kullandığınızda Murat Eren'in(fetö mağduru) adını geçirseniz eyvAllah derim (: 