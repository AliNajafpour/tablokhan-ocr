# Tablo Khan (EN / [FA](https://github.com/The-nd/tablokhan-ocr/blob/main/README_fa.md))

**Tablo Khan** (*Lit. Board Reader*) is a persian OCR based on [PaddleOCR](https://github.com/PADDLEPADDLE/PADDLEOCR) trained on synthetic datasets, capable of detecting and recognizing Persian text in noisy, scenic, and real-world environmental images.

### Key Features
- *Tablo Khan* uses [FastAPI](https://github.com/fastapi/fastapi) which gives user a hassle-free experience and ease of use.
- *Tablo Khan* is based on fine-tuned (for Farsi) variations of `PP-OCRv6 (Medium)` and `Paddle Arabic Fine-tuned V1`, two of the well-known text detection & recognition models.


## Installing & Preparing TabloKhanOCR
#### Pre-requirements:
- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/install/) (to clone the repository. Needed for Easy start method.)
- Recommended: NVIDIA GPU with CUDA 12.6+, driver ≥ 560.94 recommended; CPU works too. Note: Most modern NVidia GPUs are compatible, you can check your GPU details using `nvidia-smi` command.

### Easy start 🚀
You can simply download and run `StartWindows.bat` or `StartMacLin.sh` (depending on your operating system) file from [RELEASES](https://github.com/The-nd/tablokhan-ocr/releases) and jump to **Usage**.

If you have a problem with this method, use the Advanced mode written below.

### Advanced mode 🔧
### Step 1: Clone repository
To have access to this repository on your local machine:

```
git clone https://github.com/AliNajafpour/tablokhan-ocr.git
cd tablokhan-ocr
```
### Step 2: Preparing and starting the server
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
It would let you to start TabloKhan server to be used as intended.


## Using *Tablo Khan*

Open your internet browser of choice and go to [127.0.0.1:8000](http://127.0.0.1:8000)

Using the *Browse...* button on webpage, choose your file(s) which you want to use *Tablo khan* on, and press on **تابلو را بخوان** button. Wait until the process completes.

Images with added detection boxes will be under `results/images` path. And if you choose "Full OCR" mode, the results will be stored inside a JSON file which can be found inside `results` folder.
##
### Repository Note
Some folders (`data`, `notebooks`, `scripts`, etc.) are kept for historical reasons and are no longer part of the application's active codebase. They are not required for the application to run.

### Contact Us
- Agha Sia: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/itisAGHA_SIA) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:siyamardaarsalan@gmail.com)

- AliNajafpour: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/Ali_NJ07) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:ali.najafpour07@gmail.com)

- MahiZab: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/MahiZab) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:MahiiZabb@gmail.com)

- The nd: [![Telegram](https://img.shields.io/badge/Telegram-blue?style=flat-square&logo=Telegram&logoColor=white)](https://t.me/The_nd_Org) OR [![E-Mail](https://img.shields.io/badge/E--Mail-red?style=flat-square&logo=Gmail&logoColor=white)](mailto:taha.naderi2008@gmail.com)

### License
GPL-3.0
