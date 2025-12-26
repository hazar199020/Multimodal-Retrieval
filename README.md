# Multimodal Retrieval using CLIP + FAISS  
A simple Python pipeline that reads **texts and images**, embeds them using **OpenAI CLIP**, stores them in a **FAISS vector database**, and allows **text or image queries** to retrieve the most similar items.

This project demonstrates a minimal, end‑to‑end **multimodal search system** using:
- CLIP (Contrastive Language–Image Pretraining)
- FAISS (Facebook AI Similarity Search)
- PyTorch + Transformers

---

## 🚀 Features
- Embed **text documents** using CLIP text encoder  
- Embed **images** using CLIP image encoder  
- Store all embeddings in a FAISS index  
- Query using:
  - **Text → retrieve similar texts/images**
  - **Image → retrieve similar texts/images**
- Cosine‑similarity based ranking  
- Fully GPU‑accelerated if CUDA is available  

---
ip install faiss-cpu
pip install transformers pillow numpy
