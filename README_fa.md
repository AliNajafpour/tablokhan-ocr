# تابلوخوان ([EN](https://github.com/AliNajafpour/tablokhan-ocr/blob/main/README.md) / FA)

*تابلوخوان* یک مدل تشخیص متن فارسی بر اساس [PaddleOCR](https://github.com/PADDLEPADDLE/PADDLEOCR) و [Hezar](https://github.com/hezarai/hezar) است که بر روی داده های ساختگی آموزش داده شده و قابلیت تشخیص و خواندن متن در تصاویر نویزدار، محیطی و صحنه‌های واقعی را دارد.

### ویژگی های کلیدی
- *تابلوخوان* از [FastAPI](https://github.com/fastapi/fastapi) استفاده میکند که به کاربر تجربه ای آسان و بدون پیچیدگی برای استفاده از مدل ارائه میدهد.
- *تابلوخوان* در مرحله تشخیص متن (Detection) از نسخه بهبودیافته (برای نوشته های پارسی) مدل شناخته شده `PP-OCRv6(Medium)` کمک میگیرد.
- *تابلوخوان* در مرحله بازشناسی متن (Recognition) از نسخه تنظیم شده مدل "هزار" که یک مدل متن باز تشخیص متن پارسی است بهره میگیرد.


## نصب و آماده سازی تابلوخوان
#### پیش نیازهای اولیه:
- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/install/) (برای کلون کردن ریپازیتوری. برای استفاده از نصب آسان نیاز است.)
- پیشنهادی: کارت گرافیکی انویدیا با پشتیبانی از CUDA 12.6+, Driver ver. >= 560.98 و cuDNN 9. اغلب کارت های گرافیکی انویدیا از این نسخه پشتیبانی میکنند.
### 🚀نصب آسان
میتوانید به سادگی فایل `StartWindows.bat` یا `StartMacLin.sh` را (با توجه به سیستم عامل خود) اجرا کرده، و با دنبال کردن مراحل، منتظر اجرای تابلوخوان باشید و  پس از اتمام کار مستقیما به قسمت ***استفاده از تابلوخوان*** بروید.

اگر هنگام استفاده از این برنامه دچار مشکلی شدید، میتوانید از نصب پیشرفته استفاده کنید.
### نصب پیشرفته
### قدم اول: کلون کردن ریپازیتوری
برای دسترسی به تابلوخوان، با استفاده از دستور های زیر، ریپازیتوری را روی سیستم شخصی خود کلون کنید:

```
git clone https://github.com/AliNajafpour/tablokhan-ocr.git
cd tablokhan-ocr
```
### قدم دوم: آماده سازی و اجرای سرور
Windows:

```cmd
py -m venv TabloKhanOCR
.\TabloKhanOCR\scripts\activate.bat
pip install -r requirements.txt
fastapi run main.py
```

Linux/macOS:

```sh
python3 -m venv TabloKhanOCR
source ./TabloKhanOCR/bin/activate
pip install -r requirements.txt
fastapi run main.py
```
این دستورات به شما کمک میکنند تا سرور را راه اندازی کنید و آماده اجرای تابلوخوان باشید

## استفاده از تابلوخوان
مرورگر اینترنتی خود را باز کرده و وارد [127.0.0.1:8000](http://127.0.0.1:8000) شوید. اینجا تابلوخوان است!

با استفاده از دکمه *Browse..* که در صفحه لوکال هاست مشهود است، فایل(های) خود که میخواهید *تابلوخوان* را روی آن استفاده کنید انتخاب کنید، و سپس دکمه **تابلو را بخوان** را کلیک کنید. تا پر شدن نوار پیشرفت منتظر بمانید.

نتایج مرتبط با فایل ها در مسیر `results` در یک فایل JSON ذخیره سازی خواهند شد.

##


### تماس با ما
- Agha Sia: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/itisAGHA_SIA) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:siyamardaarsalan@gmail.com)

- AliNajafpour: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/Ali_NJ07) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:ali.najafpour07@gmail.com)

- MahiZab: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/MahiZab) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:MahiiZabb@gmail.com)

- The nd: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/The_nd_Org) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:the.nd.orgh@gmail.com)

### لایسنس
MIT