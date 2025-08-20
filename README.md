# 📚 Research Paper Summarizer & Chatbot

A **FastAPI + Streamlit** application that allows you to:
- Upload or fetch **research papers** (PDFs or web links)
- Generate structured **summaries** (Objective, Methods, Results, Conclusion)
- **Translate summaries** into multiple languages
- **Chat with your papers** using Cohere LLM
- **Download** summaries in TXT, DOCX, or PDF format
- Convert summaries into **speech (TTS)** for easy listening

---

## 🚀 Features

- **Upload PDFs**: Upload local research papers and process them into a vector database.
- **Fetch from Links**: Fetch and index papers directly from PDF URLs or web pages.
- **Summarization**: Generate concise, structured summaries of papers.
- **Translation**: Translate summaries into languages like Arabic, French, Spanish, and German.
- **Chatbot**: Ask questions about your uploaded papers and get contextual answers with references.
- **Export & TTS**: Download summaries in multiple formats or listen to them as audio.

---

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Vector DB**: [Chroma](https://docs.trychroma.com/)
- **Embeddings**: Google Generative AI (`text-embedding-004`)
- **LLM**: [Cohere](https://cohere.com/) (`command-a-03-2025`)
- **Text-to-Speech**: Microsoft Edge TTS (`edge-tts`)
- **PDF/Text Handling**: LangChain loaders (PyPDFLoader, WebBaseLoader), Regex, ReportLab, python-docx

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/research-paper-summarizer.git
cd research-paper-summarizer
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file in the root folder with your API keys:

```ini
GOOGLE_API_KEY=your_google_api_key
CO_API_KEY=your_cohere_api_key
```

---

## ▶️ Usage

### 1. Start the FastAPI backend

```bash
uvicorn main:app --reload
```

This runs the backend at `http://127.0.0.1:8000`.

### 2. Start the Streamlit frontend

```bash
streamlit run app.py
```

The UI will open in your browser (default: `http://localhost:8501`).

---

## 📌 API Endpoints

### **Upload Paper**

```
POST /upload
```

Upload a PDF research paper for indexing.

### **Fetch Paper from Link**

```
POST /fetch_link
```

Fetch and index a paper from a given URL (PDF or webpage).

### **Ask Question**

```
POST /ask
```

Ask a question about uploaded papers.

### **Summarize Papers**

```
GET /summarize
```

Generate a structured summary of indexed papers.

### **Translate Text**

```
POST /translate
```

Translate given text into a target language.

---

## 🖥️ Frontend Features (Streamlit)

* **Upload or fetch** research papers
* Generate **summaries**
* **Translate** summaries to other languages
* **Download** as TXT, DOCX, or PDF
* Convert summary into **speech (TTS)**
* **Chat interface** to query papers interactively

---

## 📂 Project Structure

```
.
├── main.py          # FastAPI backend
├── app.py           # Streamlit frontend
├── requirements.txt # Dependencies
├── .env             # API keys
├── uploaded_papers/ # Uploaded PDFs
├── downloaded_papers/ # Papers fetched from links
```

---

## ✅ Example Workflow

1. Upload a research paper PDF or fetch from a link
2. Generate a structured summary
3. Translate it into another language
4. Download the summary as TXT/DOCX/PDF
5. Play or download the audio version
6. Ask detailed questions via chatbot

---

## 🔮 Future Improvements

* Support for **multiple LLM providers** (OpenAI, Anthropic)
* Enhanced **multi-document summaries**
* Advanced **citation tracking** in answers
* **Fine-tuned academic style summaries**

---

## 📜 License

This project is licensed under the **MIT License**.

---

👨‍💻 Developed by **Mohamed Mahmoud Emam**
