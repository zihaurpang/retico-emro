# Retico EMRO 🤖🎭

Retico modules for **EMRO** (_Emotion Recognition Model from Robot Behaviors_) that can infer emotions from sequences of robot-generated behaviors.

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/zihaurpang/retico-emro.git
cd retico-emro
````

---

## 📦 Model Download

The EMRO model is hosted on Hugging Face. Download it from:

[bsu‑slim/emro‑misty](https://huggingface.co/bsu-slim/emro-misty)


You can clone it with:

```bash
git lfs install
git clone https://huggingface.co/bsu-slim/emro-misty
```

---

## 🗂️ Input Format

The module expects inputs in the form of `GREDTextIU`, which you can obtain via the [**retico‑gred**](https://github.com/zihaurpang/retico-gred) repository.

Each input consists of a robot behavior sequence like:

```
drive_track_0_0_1 move_arm_both_51_80 display_face_resources_misty_faces_black_7_1
say_text_wow! drive_track_24_24_1 display_face_resources_misty_faces_black_8_1
move_head_0_-20_0_80 say_text_ah! display_face_resources_misty_faces_black_8_1
```

---

## 🧠 Output Emotion Categories

The model provides probabilities across these emotional groups:

* **anger\_frustration**
* **confusion\_sorrow\_boredom**
* **disgust\_surprise\_alarm\_fear**
* **interest\_desire**
* **joy\_hope**
* **understanding\_gratitude\_relief**

---

## 🔗 Dependencies

* **retico-gred** – for `GREDTextIU` class and behavior parsing
* **Hugging Face 🤗 transformers** – for loading and running `emro-misty`
