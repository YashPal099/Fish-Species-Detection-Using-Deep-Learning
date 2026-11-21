# AquaID — Fish Species Detection Website

Ek Flask-based website jisme fish ki photo upload karne par species ka naam,
confidence score, aur species ki details (habitat, diet, size, conservation
status, fun fact) dikhti hai.

## Kya-kya hai project me

```
fish-species-detector/
├── app.py                  # Flask server
├── requirements.txt
├── model/
│   ├── predict.py          # Prediction logic (real model + demo fallback)
│   ├── train_model.py      # Script to train the REAL ML model on your dataset
│   └── fish_info.json      # Species database (15 common species)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    ├── js/script.js
    └── uploads/            # Uploaded images saved here temporarily
```

## Quick start (Demo Mode)

Bina kisi dataset/training ke bhi website turant chal jaayegi — demo mode
me clearly labelled rahega ki yeh illustrative result hai, real detection
nahi.

```bash
cd fish-species-detector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Ab browser me kholo: **http://localhost:5000**

## Real ML detection kaise enable karein (important — please read)

Is project ke andar koi bhi **pre-trained fish model include nahi hai**,
kyunki real species detection ke liye ek labelled fish-image dataset chahiye
hota hai (jaise Kaggle par available "A Large Scale Fish Dataset"), aur
training ke liye compute/time chahiye jo yeh chat environment provide nahi
kar sakta. Isiliye maine ek **honest demo mode** banaya hai jo out-of-the-box
kaam karta hai, aur ek **training script** diya hai jisse aap apna real
model bana sakte ho.

### Steps:

1. Kisi bhi fish dataset ko download karo, is structure me arrange karo:
   ```
   dataset/
     Rohu/
       img1.jpg
       img2.jpg
     Catla/
       img1.jpg
       ...
     Tilapia/
       ...
   ```
   (Folder ka naam hi species ka label ban jaayega — `model/fish_info.json`
   me jo species names hain unhi naamon se folders banao taaki details bhi
   match ho jaayein. Naye species add karne ho to `fish_info.json` me bhi
   entry add kar dena.)

2. TensorFlow install karo (requirements.txt me commented hai, uncomment
   kar do ya directly install karo):
   ```bash
   pip install tensorflow
   ```

3. Training run karo:
   ```bash
   cd model
   python train_model.py --data_dir /path/to/dataset --epochs 15
   ```
   Yeh MobileNetV2 (ImageNet pretrained) par transfer learning karke
   `fish_model.h5` aur `class_indices.json` bana dega.

4. Flask app ko restart karo — `predict.py` automatically real model
   detect kar lega aur demo mode band ho jaayega.

## Notes

- Max upload size: 8MB. Supported formats: JPG, PNG, WEBP.
- Agar aap production me deploy kar rahe ho, `app.py` me `debug=True` ko
  `False` kar dena aur ek proper WSGI server (gunicorn) use karna.
- `model/fish_info.json` me species details edit/add kar sakte ho — bas
  JSON structure follow karna.
