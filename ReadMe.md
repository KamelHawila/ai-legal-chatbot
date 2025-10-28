# ⚖️ AI Legal Chatbot – Lebanese Penal Code

A multilingual AI chatbot that answers legal questions based on the **Lebanese Penal Code (قانون العقوبات اللبناني)** using NLP, semantic embeddings, and OpenAI models.

---

## 🚀 Features
- 🧠 **Semantic search** powered by `SentenceTransformer (intfloat/multilingual-e5-large)`
- 💬 **Chat interface** built with `Streamlit`
- 🔍 Retrieves relevant **law articles** from the Penal Code
- 🌍 **Multilingual support** – Arabic, English, and French
- 🗳️ Feedback system with persistent logging
- 🧾 Chat & feedback logs stored in JSONL format for further analysis

---

## 🧩 Project Structure

- **app/** – Streamlit application  
  - `streamlit_app.py` → Main chatbot interface  
  - `streamlit_app_old.py` → Older version (before redesign)

- **data/** – Data and vector storage  
  - `processed/` → Cleaned & split Lebanese Penal Code articles  
  - `embeddings/` → Generated embedding arrays *(ignored in Git)*  
  - `chroma_db_articles/` → Chroma vector database *(ignored)*

- **logs/** – Chat & feedback logs *(ignored)*  
  - `chat_logs.jsonl` → Logs of questions, answers, and retrieved articles  
  - `feedback.jsonl` → User feedback (helpful / not helpful)

- `LBN_PenalCode1943_AR.pdf` → Original Lebanese Penal Code (Arabic, 1943)  
- `.env` → Contains the OpenAI API key *(ignored)*  
- `.gitignore` → Files and folders to exclude from Git  
- `requirements.txt` → Python dependencies for the app  
- `README.md` → Full project documentation  

- **Notebooks (Data Preparation & Embedding)**
  - `scraping.ipynb` → Extracts and cleans text from the legal PDF  
  - `Embeddings.ipynb` → Generates embeddings for articles  
  - `upgrade_embeddings.ipynb` → Upgrades to multilingual embedding model  
  - `Meta_data.ipynb` → Creates metadata for each article  
  - `chunks_upgrade.ipynb` → Splits text by article and cleans formatting

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/KamelHawila/ai-legal-chatbot.git
cd ai-legal-chatbot


2. Create a virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Mac/Linux

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables

Create a .env file in the project root:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

5. Run the chatbot
streamlit run app/streamlit_app.py


---

### 2️⃣ **💡 Usage Example**

Show how a user interacts with your chatbot:

```markdown
---

## 💡 Usage Example

Once launched, you can ask legal questions such as:

**Arabic:**  
> ما هي العقوبة على السرقة؟

**English:**  
> What is the punishment for theft?

**French:**  
> Quelle est la peine pour un vol commis la nuit dans une maison habitée ?

The chatbot retrieves relevant articles from the **Lebanese Penal Code**, then summarizes them with references like:
> تعاقب السرقة بالأشغال الشاقة من ثلاث سنوات إلى سبع سنوات في حالات معينة، مثل الكسر أو استعمال العنف... [المادة 639]


---

## 📊 Logging & Feedback System

All interactions are stored for analysis and improvement:

| File | Description |
|------|--------------|
| `logs/chat_logs.jsonl` | Stores each Q&A interaction, including retrieved article IDs |
| `logs/feedback.jsonl` | Stores user feedback (helpful / unhelpful) |
| `data/chroma_db_articles/` | Vector database for semantic search (ignored in Git) |

These logs help analyze accuracy, track common legal queries, and improve future model versions.

---

## 🧠 Tech Stack

- **Language:** Python  
- **Frontend:** Streamlit  
- **AI Models:** OpenAI GPT-4o / GPT-4o-mini  
- **Embeddings:** SentenceTransformer (intfloat/multilingual-e5-large)  
- **Database:** ChromaDB  
- **NLP Tools:** Langdetect, Deep-Translator  
- **Utilities:** NumPy, dotenv


---

## 📜 License

This project is released under the **MIT License**.  
You’re free to use, modify, and distribute it with attribution.

---

## 👨‍💻 Author

**Kamel Hawila**  
🎓 Computer Science Graduate | Data Science & AI Enthusiast  
🌐 [GitHub](https://github.com/KamelHawila) • [LinkedIn](https://www.linkedin.com/in/kamel-hawila-70052b355/)


---

## 🌐 Live Demo

You can try the chatbot online here:  
👉 [https://ai-legal-chatbot.streamlit.app](https://ai-legal-chatbot.streamlit.app)


