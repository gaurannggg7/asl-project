# 🧏‍♂️ Speech-to-ASL Translator
### Dataset

This project uses video clips from **ASL-LEX v2**  
*Caselli et al. (2022), “ASL-LEX 2.0: A lexical database...”*  
See the original OSF repository for license & citation details.

This project converts real-time **spoken English** into **American Sign Language (ASL)** gloss and visual signs — completely offline. It's designed to support the Deaf and Hard-of-Hearing community in live conversations, workshops, or classrooms.

---

## 🎥 Demo

> Speak into the mic (e.g., “Hello borrow book”)  
> → ASL gloss generated  
> → Sign videos stitched and previewed  
> → All offline!

---

## 🧠 How It Works

**🎤 Voice Input**  
➡ **📝 Offline Speech Recognition** with [Vosk](https://alphacephei.com/vosk/)  
➡ **🔀 English-to-ASL Gloss Conversion**  
➡ **🤟 Video Playback for ASL Signs**  

---
## ✅ Features

- 🎙️ Real-time microphone transcription (offline)
- 🔠 Converts English to basic ASL grammar (Time–Topic–Comment)
- 🎬 Plays stitched sign videos for each word
- 💻 Runs completely offline — no internet required
- 🧪 Testable with `.wav` or microphone input

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd speech-to-asl
```
