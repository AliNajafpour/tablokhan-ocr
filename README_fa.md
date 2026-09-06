# تابلوخوان ([EN](https://github.com/The-nd/tablokhan-archive/blob/main/README.md) / FA)

*تابلوخوان* یک مدل تشخیص متن فارسی بر اساس [PaddleOCR](https://github.com/PADDLEPADDLE/PADDLEOCR) است که بر روی داده های ساختگی آموزش داده شده و قابلیت تشخیص و خواندن متن در تصاویر نویزدار، محیطی و صحنه‌های واقعی را دارد.

### ویژگی های کلیدی
- *تابلوخوان* از [FastAPI](https://github.com/fastapi/fastapi) استفاده میکند که به کاربر تجربه ای آسان و بدون پیچیدگی برای استفاده از مدل ارائه میدهد.
- *تابلوخوان* از نسخه بهبودیافته (برای نوشته های پارسی) مدل های شناخته شده `PP-OCRv6(Medium)` و `Paddle Arabic Fine-tuned V1` کمک میگیرد. 


## نصب و آماده سازی تابلوخوان
#### پیش نیازهای اولیه:
- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/install/) (برای کلون کردن ریپازیتوری. برای استفاده از نصب آسان نیاز است.)
- پیشنهادی: کارت گرافیکی انویدیا با پشتیبانی از CUDA 12.6+, Driver ver. >= 560.94 و cuDNN 9. اغلب کارت های گرافیکی انویدیا از این نسخه پشتیبانی میکنند.
### 🚀نصب آسان
میتوانید به سادگی فایل `StartWindows.bat` یا `StartMacLin.sh` را (با توجه به سیستم عامل خود) از قسمت [RELEASES](https://github.com/The-nd/tablokhan-archive/releases) دانلود و اجرا کرده، و با دنبال کردن مراحل، منتظر اجرای تابلوخوان باشید و  پس از اتمام کار مستقیما به قسمت ***استفاده از تابلوخوان*** بروید.

اگر هنگام استفاده از این برنامه دچار مشکلی شدید، میتوانید از نصب پیشرفته استفاده کنید.
### نصب پیشرفته
### قدم اول: کلون کردن ریپازیتوری
برای دسترسی به تابلوخوان، با استفاده از دستور های زیر، ریپازیتوری را روی سیستم شخصی خود کلون کنید:

```
git clone https://github.com/The-nd/tablokhan-archive.git
cd tablokhan-archive
```
### قدم دوم: آماده سازی و اجرای سرور
Windows:

```cmd
py -m venv TabloKhanOCR
.\TabloKhanOCR\scripts\activate.bat
pip install -r requirements.txt
py -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Linux/macOS:

```sh
python3 -m venv TabloKhanOCR
source ./TabloKhanOCR/bin/activate
pip install -r requirements.txt
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
```
این دستورات به شما کمک میکنند تا سرور را راه اندازی کنید و آماده اجرای تابلوخوان باشید

## استفاده از تابلوخوان
مرورگر اینترنتی خود را باز کرده و وارد [127.0.0.1:8000](http://127.0.0.1:8000) شوید. اینجا تابلوخوان است!

با استفاده از دکمه *Browse..* که در صفحه لوکال هاست مشهود است، فایل(های) خود که میخواهید *تابلوخوان* را روی آن استفاده کنید انتخاب کنید، و سپس دکمه **تابلو را بخوان** را کلیک کنید. تا پایان مراحل منتظر بمانید.


تصاویر دارای کادرهای تشخیص اضافه‌شده در مسیر `results/images` قرار خواهند گرفت. همچنین، اگر حالت "Full OCR" را انتخاب کنید، نتایج داخل یک فایل JSON ذخیره می‌شوند که می‌توانید آن را در پوشه `results` پیدا کنید.


##

### یادداشتی درباره مخزن
برخی پوشه‌ها (`data`، `notebooks`، `scripts` و غیره) به دلایل تاریخی در مخزن نگه داشته شده‌اند و دیگر بخشی از کد فعال برنامه محسوب نمی‌شوند. این پوشه‌ها برای اجرای برنامه موردنیاز نیستند.
### تماس با ما
- Agha Sia: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/itisAGHA_SIA) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:siyamardaarsalan@gmail.com)

- AliNajafpour: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/Ali_NJ07) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:ali.najafpour07@gmail.com)

- MahiZab: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/MahiZab) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:MahiiZabb@gmail.com)

- The nd: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/The_nd_Org) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:taha.naderi2008@gmail.com)

### لایسنس
GPL-3.0
