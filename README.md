# Beş sebze türünün sınıflandırılması

Bu bilgisayarlı görü projesi fotoğrafları **Patlıcan, Brokoli, Havuç, Patates** ve **Domates** sınıflarına ayırır. Karşılaştırma için küçük bir CNN sıfırdan eğitildi. Son model, ImageNet ağırlıklarıyla başlatılan ve düşük doğrulama kaybına göre seçilen ResNet18’dir. Uygulama Gradio ile yerel olarak çalışır. [English README](README_en.md)

## Sonuçlar

| Ölçüt | Küçük CNN | ResNet18 |
| --- | ---: | ---: |
| Veri seti test doğruluğu | %97,70 | %100,00 |
| Test makro F1 | 0,9770 | 1,0000 |
| Eğitim süresi (s) | 115,56 | 326,44 |
| Ortalama GPU ileri geçişi (ms) | 0,547 | 4,264 |
| Ortalama JPEG’den sınıf skorlarına (ms) | 2,507 | 6,349 |

Çıkarım süreleri aynı GPU ve JPEG ile tek görüntülük yığında ölçüldü. ResNet18 ilk beş kişisel fotoğrafta **3/5**, genişletilmiş koleksiyonda **11/14** doğru tahmin yaptı. Veri seti bölümleri arasında görsel olarak çok benzer görüntüler bulundu. Bu nedenle %100 test doğruluğundan bağımsız fotoğraflardaki başarı çıkarılamaz. [Karşılaştırma verileri](outputs/model_comparison.json) · [Genişletilmiş koleksiyon sonuçları](outputs/personal_new_evaluation/predictions.csv)

## Windows kurulumu

Kurulum Windows 11 ve **64 bit Python 3.14.0** ile doğrulandı. Proje kökünde PowerShell kullanın. CPU kurulumu uygulamayı çalıştırmak için yeterlidir.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

### NVIDIA GPU kurulumu

GPU kullanacaksanız yeni sanal ortamda CPU PyTorch satırı yerine aşağıdaki satırı çalıştırın, ardından `requirements.txt` kurulumuna devam edin. CUDA 13.0 paketi RTX 5070 Ti Laptop GPU ve NVIDIA 617.42 sürücüsüyle doğrulandı. [Resmî PyTorch kurulum seçenekleri](https://pytorch.org/get-started/previous-versions/)

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

## Model ve uygulama

Önce ResNet18 ağırlığını `checkpoints/resnet18/best.pt` konumuna yerleştirin. Ağırlıklar bir GitHub sürüm eki olarak dağıtıma hazırlanmıştır. Dosya adları, doğrulama değerleri ve isteğe bağlı küçük CNN ağırlığı için [model ağırlıkları kılavuzuna](MODEL_WEIGHTS.md) bakın.

```powershell
.\.venv\Scripts\python.exe app.py
```

Tarayıcıda `http://127.0.0.1:7860` adresini açın. Bir fotoğraf yükleyip **Classify photograph** düğmesine basın. Arayüz tahmini ve beş softmax skorunu gösterir. CPU’yu açıkça seçmek için `app.py --device cpu` kullanabilirsiniz.

## Veri seti ve rapor

Deneyleri yeniden üretmek için [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) arşivini indirin. Özgün bölümleri `data/vegetables/` altına çıkarın ve veri setindeki İngilizce sınıf klasörlerini koruyun.

```text
data/vegetables/
  train/{Brinjal,Broccoli,Carrot,Potato,Tomato}/
  validation/{Brinjal,Broccoli,Carrot,Potato,Tomato}/
  test/{Brinjal,Broccoli,Carrot,Potato,Tomato}/
data/personal/                 İlk beş fotoğraf ve labels.csv
data/personal_new/             Genişletilmiş koleksiyon, sınıf klasörleriyle
```

[Türkçe rapor (PDF)](report/rapor_tr.pdf) · [Türkçe rapor (Markdown)](report/rapor_tr.md) · [Eğitim eğrileri](report/figures_tr/egitim_egrileri.png) · [Karışıklık matrisi](report/figures_tr/karisiklik_matrisi.png) · [Kişisel fotoğraflar](report/figures_tr/kisisel_fotograflar.png) · [Uygulama ekran görüntüsü](outputs/app_screenshot_followup.png)
