# FER2013 7 alapérzelem tanítása, TODO: tesztelés
project_root/
│
├── data/
│   ├── fer2013/
│   │   ├── train/       # Tanító adathalmaz (FER2013 train rész)
│   │   ├── test/        # Teszt adathalmaz (FER2013 test rész)
│
├── models/
│   └── emotion_detection_model.h5  # Betanított modell fájl
│
├── src/
│   ├── __init__.py      
│   ├── train_model.py   # Modell építése és tanítása
│   ├── data_preprocessing.py  # Adatok betöltése és előfeldolgozása
│
└── fer_train_test.py    # Futtató fájl tanításhoz (tesztelést még nem tartalmaz)


# FER2013:
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   ├── neutral/
├── test/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   ├── neutral/

