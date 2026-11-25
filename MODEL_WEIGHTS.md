# Model ağırlıklarının dağıtımı

Aşağıdaki dosyalar gelecekteki GitHub Release için hazırlanmıştır. Sürüm yayımlandığında ilgili eki edinin, SHA-256 değerini kontrol edin ve dosyayı tablodaki hedefe yerleştirin. Varsayılan uygulama için ResNet18 ağırlığı gerekir. Küçük CNN ağırlığı, kaydedilmiş karşılaştırma modelini çalıştırmak ve eş koşullarda ölçümleri tekrarlamak içindir.

| Model | Gelecekteki sürüm eki | Yerel kaynak ve hedef | Bayt | SHA-256 |
| --- | --- | --- | ---: | --- |
| ResNet18, doğrulamayla seçilen 8. dönem | `vegetable-resnet18.pt` | `checkpoints/resnet18/best.pt` | 44.797.451 | `4d312113394e92367297adc9e16aef2f0442e97b44b8316647c331ec630c38a4` |
| Küçük CNN, temel model | `vegetable-small-cnn.pt` | `baseline/checkpoints/best.pt` | 386.855 | `7e63ad147ac022e372febd6ed688d1bf7c25aea573223d99a3faf9ed542aedb6` |

İndirilen dosyaları proje köküne koyduktan sonra PowerShell ile ResNet18’i yerleştirin.

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\vegetable-resnet18.pt
New-Item -ItemType Directory -Force .\checkpoints\resnet18 | Out-Null
Copy-Item -LiteralPath .\vegetable-resnet18.pt -Destination .\checkpoints\resnet18\best.pt
```

Küçük CNN’i de kullanacaksanız şu adımları uygulayın.

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\vegetable-small-cnn.pt
New-Item -ItemType Directory -Force .\baseline\checkpoints | Out-Null
Copy-Item -LiteralPath .\vegetable-small-cnn.pt -Destination .\baseline\checkpoints\best.pt
.\.venv\Scripts\python.exe app.py --checkpoint baseline/checkpoints/best.pt --port 7861
```

Varsayılan model yolu ve beklenen dosya özeti [`checkpoints/deployment.json`](checkpoints/deployment.json) içinde kayıtlıdır. Ağırlık dosyaları sınıf eşlemesini, mimariyi ve ön işleme ayarlarını içerir.
