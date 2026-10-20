# Automated Attendance System using Face Recognition

A camera takes a group photo, and the system marks everyone it recognises as present.
Built as a diploma major project at the Centre for Computers and Communication
Technology, Sikkim, 2021, by a team of four.

The first version scored 100% on its test set and marked a stranger as present in a
real classroom photo. This repository is the fixed version, and the README explains
what went wrong, because that is the interesting part.

---

## How it works

```
photo (phone / IP camera)
      │
      ▼
MTCNN face detection  ──►  crop, resize 160×160
      │
      ▼
FaceNet (pre-trained)  ──►  128-dimensional embedding
      │
      ├──► softmax classifier (Dense 512 → Dropout → Dense 512 → Dropout → num_classes)
      │           │
      │           ▼
      │     predicted class + confidence
      │
      └──► cosine distance to 5 stored embeddings of that class
                  │
                  ▼
        accept  ⟺  confidence > 0.90  AND  mean distance < 0.50
                  │
                  ▼
            attendance.csv
```

Two independent gates. Either one can reject. That is the whole design.

---

## The failure that shaped it

The first version used **SVM and KNN** on the FaceNet embeddings. On three test sets
they scored **100%, 100% and 91.66%**.

Deployed, they marked a person who had never been enrolled as present.

The reason: the models were answering *"which of these four people is this?"* — a
closed question over four classes. The classroom asks *"is this anyone we know?"* — an
open question. A softmax over four people cannot say *none of them*. Given a stranger,
it must pick somebody, and it picked confidently.

Accuracy could never have caught this, because accuracy was computed on the same four
people. A closed set cannot produce evidence about a fifth person who is not in it.

The fix was not a bigger model. It was a change of decision rule: keep the classifier,
add a way to refuse.

## The measurement that says it works

| | Mean cosine distance | Class probability |
|---|---|---|
| Enrolled face | **0.19** | **99.99%** |
| Unknown face | **0.96** | **70.16%** |

A 0.77 distance gap and a 29.8-point confidence gap. The gate has something real to cut
on. Tested on two enrolment strategies — one photo expanded into ~36 augmented
variants, and nine photos at different angles — with the augmentation approach scoring
slightly better.

---

## Running it

**Requirements:** Python with TensorFlow/Keras, OpenCV, MTCNN, Pillow, pandas,
scikit-learn. A 4 GB machine with integrated graphics is enough — this runs on CPU.

```bash
pip install -r requirements.txt     # see the note below
```

1. Install **Sharik** on an Android phone
   ([Play Store](https://play.google.com/store/apps/details?id=dev.monora.sharik)) — it
   is free, open source, and sends photos to the PC over local FTP.
2. Put the phone and the PC on the same Wi-Fi network and note the URL Sharik shows.
3. Run `RecogAtten.py` once and enter that URL when prompted (it is written to
   `IPinfo.ini`).
4. Capture training samples for each person and place them under `images/`.
5. Run `DetectTrain.py` to train. This writes `Face_recogniser.h5` and
   `embed_train.npz`.
6. Take a group photo and send it from the phone via Sharik.
7. Run `RecogAtten.py` again. Attendance lands in `attendance.csv`.

> **Note on `requirements.txt`:** the file is not in this repository. Install
> `tensorflow`, `keras`, `mtcnn`, `opencv-python`, `pillow`, `pandas`, `numpy`,
> `scikit-learn` and `requests` directly, pinned to versions compatible with the
> TensorFlow release you have.

## Repository contents

| Path | What it is |
|---|---|
| `DetectTrain.py` | Enrolment: detection, embedding, training the classifier, saving the model |
| `RecogAtten.py` | Recognition: fetch photo, embed each face, classify, gate, write attendance |
| `Face_recogniser.h5` | Trained softmax head over the enrolled classes |
| `embed_train.npz` | Stored 128-d embeddings used for the distance gate |
| `face_dataset_numpy.npz` | Pre-processed face crops |
| `facenet_keras.h5` | Pre-trained FaceNet weights (92 MB) |
| `images/` | Enrolment photos |
| `attendance.csv` | Output |

## Limitations

- **Small enrolment.** Designed for a classroom-sized set of known people, not a
  general face database.
- **The distance threshold is hand-tuned** (0.50 in the shipped code, 0.4 in the
  written report). It works; it was not chosen from a precision-recall curve, and that
  is what I would do next.
- **No liveness check.** A printed photo would pass.
- **Single machine, single camera.** No concurrency, no server.
- **CPU only.** Fast enough for a class photo, not for video.

## What I would do differently

1. Split by **person**, never by row, and hold out an entire identity so there is a
   genuine unknown to test rejection against.
2. Choose the threshold from a precision-recall curve, not by eye.
3. Report false-accept and false-reject rates, not accuracy.

---

Team: Sumiran Bhattarai, Roman Pradhan, Pragesh Pradhan, Sudip Chettri.
Guide: Mr. Bijoy Chettri. Submitted July 2021.
