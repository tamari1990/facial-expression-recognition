# Facial Expression Recognition (FER2013)

Kaggle Competition: [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)

WandB Project: [facial-expression-recognition](https://wandb.ai/tgela23-free-university-of-tbilisi-/facial-expression-recognition)

---

## Dataset

FER2013 — 28,709 სურათი, 48x48 grayscale, 7 კლასი:

| Label | Emotion  | Count |
|-------|----------|-------|
| 0     | Angry    | 3,995 |
| 1     | Disgust  | 436   |
| 2     | Fear     | 4,097 |
| 3     | Happy    | 7,215 |
| 4     | Sad      | 4,830 |
| 5     | Surprise | 3,171 |
| 6     | Neutral  | 4,965 |

კლასების დისბალანსი შესამჩნევია — Disgust კლასი 16x ნაკლებია Happy-სთან შედარებით.

---

## Repository Structure

```
facial-expression-recognition/
├── src/
│   ├── dataset.py       # FERDataset, get_dataloaders
│   ├── models.py        # SimpleCNN, MediumCNN, DeepCNN
│   └── train.py         # train loop, WandB logging
├── notebooks/
│   └── fer_training.ipynb
└── README.md
```

---

## Preprocessing

- pixels სტრიქონი → numpy array → 48x48
- Normalize: 0-255 → 0-1
- Shape: HxW → 1xHxW (grayscale channel)
- Train/Val split: 80/20

---

## Model Architecture Decisions

არქიტექტურები შეირჩა იტერაციულად — პატარიდან დიდამდე, თითოეული წინაზე დაყრდნობით.

### Model 1: SimpleCNN

ყველაზე მინიმალური არქიტექტურა — baseline-ის დასამყარებლად. მიზანი იყო გაგვეგო რამდენს შეძლებს ქსელი ყოველგვარი regularization-ის გარეშე.

```
Conv(1→16) → ReLU → MaxPool
Conv(16→32) → ReLU → MaxPool
Flatten → Linear(4608→128) → ReLU → Linear(128→7)
```

პარამეტრები: **595,655**

**გადაწყვეტილება:** BatchNorm და Dropout არ დაემატა განზრახ — გვინდოდა გვენახა სუფთა overfitting-ის ქცევა.

---

### Model 2: MediumCNN

SimpleCNN-ის overfitting-ის საპასუხოდ დაემატა: 4 Conv layer, BatchNorm და Dropout. BatchNorm სტაბილიზაციისთვის, Dropout regularization-ისთვის.

```
Conv(1→32) → BN → ReLU → MaxPool
Conv(32→64) → BN → ReLU → MaxPool
Conv(64→128) → BN → ReLU → MaxPool
Conv(128→256) → BN → ReLU → MaxPool
Flatten → Linear(2304→512) → ReLU → Dropout(0.5) → Linear(512→7)
```

პარამეტრები: **1,572,551**

**გადაწყვეტილება:** Dropout(0.5) შეირჩა aggressive regularization-ისთვის. Single batch ტესტმა აჩვენა რომ 0.5 ძალიან მაღალია — 1000 iteration-ზეც კი 78%-ს ვერ სცდება. Dropout გამორთვისას 500 iteration-ში 100%-ს აღწევს, რაც ნიშნავს რომ capacity საკმარისია.

---

### Model 3: DeepCNN (VGG-style)

MediumCNN-ის გამოცდილებით — deeper ქსელი paired conv layers-ით (VGG სტილი), Dropout2d feature map დონეზე.

```
Block 1: Conv(1→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout2d
Block 2: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → Dropout2d
Block 3: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool → Dropout2d
Flatten → Linear(9216→1024) → ReLU → Dropout → Linear(1024→512) → ReLU → Dropout → Linear(512→7)
```

პარამეტრები: **11,112,647**

**გადაწყვეტილება:** Dropout2d შეირჩა რადგან ის მთელ feature map-ს თიშავს (არა ცალკეულ neurons-ს), რაც უფრო ეფექტური regularization-ია კონვოლუციური ქსელებისთვის. dropout პარამეტრად გადავიტანეთ hyperparameter tuning-ისთვის.

---

## Forward & Backward Pass Verification

სანამ სრულ training-ზე გადავიდოდით, შევამოწმეთ ქსელების სისწორე:

### Forward Pass
```
SimpleCNN:  input=[4,1,48,48] → output=[4,7] 
MediumCNN:  input=[4,1,48,48] → output=[4,7] 
DeepCNN:    input=[4,1,48,48] → output=[4,7] 
```

### Backward Pass (Gradient Flow)
```
SimpleCNN:  gradients flowing = True 
MediumCNN:  gradients flowing = True 
DeepCNN:    gradients flowing = True 
```

### Overfit on Single Batch (100 iterations)
ეს ტესტი გვიჩვენებს აქვს თუ არა მოდელს საკმარისი capacity:

```
SimpleCNN  dropout=default → 100%   სწრაფად overfit-დება
MediumCNN  dropout=0.5     →  57%   Dropout ძალიან აგრესიულია
DeepCNN    dropout=0.25    →  25%   Dropout2d channels-ს თიშავს
```

**ანალიზი:** MediumCNN და DeepCNN-ი ვერ overfit-დებოდნენ არა capacity-ის ნაკლებობის გამო, არამედ Dropout-ის სიძლიერის გამო. ეს დავადასტურეთ:

```
MediumCNN  dropout=0.0  500 iter  → 100% 
DeepCNN    dropout=0.0  500 iter  → 100% 
DeepCNN    dropout=0.05 100 iter  →  81%
DeepCNN    dropout=0.01 300 iter  →  95%
```

---

## Experiments & Results

### SimpleCNN Experiments

| Run | LR | Batch | Train Acc | Val Acc | Gap | დასკვნა |
|-----|----|-------|-----------|---------|-----|---------|
| simplecnn-baseline | 0.001 | 64 | 94% | 49% | 45% | ძლიერი overfitting |
| simplecnn-lr0001 | 0.0001 | 64 | 47% | 45% | 2% | underfitting — ძალიან ნელი სწავლა |

**ანალიზი:** LR შემცირებამ overfitting გადაჭრა, მაგრამ 20 epoch საკმარისი არ აღმოჩნდა სწავლისთვის. SimpleCNN-ს regularization არ აქვს, ამიტომ lr=0.001 სწრაფად ზეპირად სწავლობს train data-ს.

---

### MediumCNN Experiments

| Run | LR | Train Acc | Val Acc | Gap | დასკვნა |
|-----|----|-----------|---------|-----|---------|
| mediumcnn-baseline | 0.001 | 87% | 57% | 30% | overfitting, მაგრამ SimpleCNN-ზე უკეთესი |
| mediumcnn-lr0001 | 0.0001 | 97% | 54% | 43% | კიდევ უარესი overfitting |

**ანალიზი:** დაბალმა LR-მა მეტი ეპოქა მისცა მოდელს train data-ს დასამახსოვრებლად. BatchNorm დაეხმარა სტაბილიზაციაში, მაგრამ Dropout(0.5) classifier-ზე საკმარისი არ აღმოჩნდა feature extractor-ის regularization-ისთვის.

---

### DeepCNN Experiments

| Run | Dropout | Optimizer | Epochs | Train Acc | Val Acc | Gap | დასკვნა |
|-----|---------|-----------|--------|-----------|---------|-----|---------|
| deepcnn-baseline | 0.25 | Adam | 20 | 57% | 56% | 1% | underfitting — Dropout ძალიან ძლიერი |
| deepcnn-dropout015 | 0.15 | Adam | 20 | 69% | 59% | 10% | გაუმჯობესდა |
| deepcnn-low-dropout | 0.10 | Adam | 20 | 81% | 60% | 21% | საუკეთესო Adam-ით |
| deepcnn-adamw-dropout010 | 0.10 | AdamW | 30 | 91% | **63%** | 28% | საუკეთესო საერთოდ  |
| deepcnn-adamw-dropout012 | 0.12 | AdamW | 30 | 89% | 62% | 27% | თითქმის იგივე |

**ანალიზი:** 
- dropout=0.25 → underfitting: Dropout2d ძალიან ბევრ channels-ს თიშავდა
- dropout შემცირებამ val accuracy გაზარდა, მაგრამ gap-იც გაიზარდა
- AdamW-მ weight decay-ით დამატებითი regularization მოახდინა და val accuracy 60% → 63%-მდე აიყვანა

---

## Final Comparison

| Model | Best Val Acc | Train Acc | Gap | მდგომარეობა |
|-------|-------------|-----------|-----|-------------|
| SimpleCNN (lr=0.001) | 49% | 94% | 45% | Overfitting |
| SimpleCNN (lr=0.0001) | 45% | 47% | 2% | Underfitting |
| MediumCNN (lr=0.001) | 57% | 87% | 30% | Overfitting |
| MediumCNN (lr=0.0001) | 54% | 97% | 43% | Overfitting |
| DeepCNN (dropout=0.25) | 56% | 57% | 1% | Underfitting |
| **DeepCNN (AdamW, dropout=0.10)** | **63%** | 91% | 28% | **საუკეთესო**  |

---

## Key Findings

1. **SimpleCNN** — regularization-ის გარეშე სწრაფად overfit-დება. LR შემცირება underfitting-ს იწვევს 20 epoch-ში.

2. **MediumCNN** — BatchNorm სტაბილიზაციას უწყობს ხელს, მაგრამ Dropout(0.5) მხოლოდ classifier-ზე საკმარისი regularization არ არის.

3. **DeepCNN** — Dropout2d-ის სიდიდე კრიტიკულია: 0.25 → underfitting, 0.10 → ბალანსი. AdamW weight decay-ით საუკეთესო შედეგი მოგვცა.

4. **Forward/Backward ტესტი** — single batch overfit ტესტმა აჩვენა რომ capacity პრობლემა არ იყო — regularization იყო ძალიან აგრესიული.

---

## Tech Stack

- PyTorch
- WandB — experiment tracking
- Google Colab (T4 GPU)
- Kaggle API — dataset
- GitHub — version control
