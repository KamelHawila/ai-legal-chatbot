# app/streamlit_app.py
# -*- coding: utf-8 -*-
import os, json, datetime, traceback
import numpy as np
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from langdetect import detect
from deep_translator import GoogleTranslator

# ==============================
# 0) BOOTSTRAP
# ==============================
try:
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
except Exception:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# If key found, set it for OpenAI client
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    st.error("❌ لم يتم العثور على مفتاح OpenAI. تأكد من إضافته في Secrets أو ملف .env.")

OPENAI_MODEL_DEFAULT = "gpt-4o-mini"  # change to "gpt-4o" if you have access

# Connect OpenAI
def get_openai_client():
    try:
        return OpenAI()
    except Exception as e:
        st.error("❌ لم يتم العثور على مفتاح OpenAI. الرجاء التأكد من ضبط المتغير OPENAI_API_KEY.")
        raise e

client = get_openai_client()

# Connect Chroma (articles DB)
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)

CHROMA_PATH = "data\chroma_db_articles"
COLLECTION_NAME = "penal_code_articles"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embed_func)

# for sidebar stats
ARTICLES_PATH = "data\processed\penal_code_articles.json"
try:
    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        _articles_preview = json.load(f)
    TOTAL_ARTICLES = len(_articles_preview)
except Exception:
    TOTAL_ARTICLES = None

# ==============================
# 1) PAGE CONFIG + STYLE
# ==============================
st.set_page_config(page_title="⚖️ AI Legal Chatbot – Lebanese Law", page_icon="⚖️", layout="wide")

# Custom CSS: Arabic RTL chat bubbles, nicer cards, code blocks
st.markdown("""
<style>
/* Global Light Mode */
html, body, [class*="css"] {
  direction: rtl;
  font-family: "Noto Kufi Arabic", "Segoe UI", Arial, sans-serif;
  background-color: #F9FAFB !important;
  color: #111827 !important;
}

/* Chat container */
.block-container {
  padding-top: 1.5rem !important;
  background-color: #F9FAFB !important;
}

/* Chat bubbles */
.msg {
  padding: 0.9rem 1.1rem;
  border-radius: 14px;
  margin: 0.4rem 0;
  max-width: 92%;
  line-height: 1.7;
  word-wrap: break-word;
  white-space: pre-wrap;
  font-size: 0.95rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.user {
  background: #E8EDFF;
  color: #111827;
  margin-left: auto;
  border-top-left-radius: 6px;
  border: 1px solid #D0DAFF;
}
.bot {
  background: #F3F4F6;
  color: #111827;
  margin-right: auto;
  border-top-right-radius: 6px;
  border: 1px solid #E5E7EB;
}

/* Source cards */
.src-card {
  border: 1px solid #E5E7EB;
  background: #FFFFFF;
  border-radius: 12px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.6rem;
  color: #111827;
}
.src-title {
  font-weight: 700;
  color: #2563EB;
}
.src-meta {
  font-size: 0.85rem;
  color: #6B7280;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background-color: #F3F4F6 !important;
  border-left: 1px solid #E5E7EB !important;
  transition: all 0.3s ease-in-out;
}
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] div, 
[data-testid="stSidebar"] p {
  color: #111827 !important;
}
[data-testid="stSidebar"] .stSlider [role='slider'] {
  background-color: #2563EB !important;
  border: 1px solid #1E40AF !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb='slider'] {
  color: #111827 !important;
}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] select {
  background-color: #FFFFFF !important;
  color: #111827 !important;
  border: 1px solid #D1D5DB !important;
}

/* Widgets */
.stTextInput>div>div>input, textarea, .stTextArea textarea {
  background-color: #FFFFFF !important;
  color: #111827 !important;
  border: 1px solid #D1D5DB !important;
}
.stButton>button {
  background-color: #2563EB !important;
  color: #F9FAFB !important;
  border-radius: 8px;
  border: none;
  font-weight: 600;
  transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
  background-color: #1D4ED8 !important;
}

/* Divider + badge + footer */
hr, .stDivider {
  border-color: #E5E7EB !important;
}
.badge {
  background: #2563EB;
  color: #FFF;
  border-radius: 8px;
  padding: 2px 8px;
  font-size: 0.75rem;
}
.footer {
  font-size: 0.85rem;
  color: #6B7280;
  text-align: center;
  margin-top: 20px;
  border-top: 1px solid #E5E7EB;
  padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# 2) HELPERS
# ==============================
def preprocess_query(user_query: str):
    """
    Detects language, translates to Arabic for retrieval,
    and returns (translated_query, detected_lang).
    """
    try:
        lang = detect(user_query)
    except:
        lang = "ar"  # fallback if detection fails
    
    # Only translate if not Arabic
    if lang != "ar":
        try:
            translated = GoogleTranslator(source=lang, target="ar").translate(user_query)
        except Exception as e:
            st.warning("⚠️ Translation failed, using original text.")
            translated = user_query
    else:
        translated = user_query

    return translated, lang

def retrieve_context(query: str, top_k: int = 3):
    """Return top_k (doc, meta, sim) tuples."""
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    sims  = [1 - d for d in dists]
    return list(zip(docs, metas, sims))

def build_context_block(results):
    """Turn retrieval results into a compact context for the LLM, with article labels."""
    lines = []
    for doc, meta, _ in results:
        art = meta.get("article")
        tag = f"[المادة {art}]" if art else "[مادة]"
        # keep context not too long per item
        snippet = doc.strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "…"
        lines.append(f"{tag}\n{snippet}")
    return "\n\n".join(lines)

def build_prompt_multilingual(user_q, context_block, lang):
    """
    Builds a context-aware, multilingual prompt.
    If the user refers to 'that' or 'previous answer', include prior exchange.
    """
    last_turn = ""
    if st.session_state["conversation_memory"]:
      prev = st.session_state["conversation_memory"][-1]
      user_lower = user_q.lower()

    # Check if user wants to translate previous answer
      if "translate" in user_lower or "ترجم" in user_lower or "traduire" in user_lower:
        last_turn = (
            f"\nPrevious answer (to be translated): {prev['answer']}\n\n"
            "If the user requested a translation, translate the previous answer only — "
            "do NOT repeat explanations. Use the target language of the new query."
        )
      else:
        last_turn = (
            f"\nPrevious question: {prev['question']}\nPrevious answer: {prev['answer']}\n"
            "Use this context only if it helps clarify the new question."
        )


    if lang == "ar":
        return (
            "أنت مساعد قانوني لبناني. استخدم المحادثة السابقة إذا كانت الجملة تشير إلى إجابة سابقة "
            "مثل 'اشرح أكثر' أو 'ترجم ذلك'. أجب بالعربية الفصحى بدقة، واستند فقط على النص القانوني المقدّم.\n"
            + last_turn +
            f"السؤال الحالي:\n{user_q}\n\n"
            f"السياق:\n{context_block}\n\nالجواب:"
        )
    elif lang == "fr":
        return (
            "Vous êtes un assistant juridique libanais. Utilisez la question précédente si le nouvel "
            "énoncé fait référence à 'cela' ou 'la réponse précédente'. Répondez en français clairement "
            "en vous basant uniquement sur le contexte juridique fourni.\n"
            + last_turn +
            f"Question actuelle :\n{user_q}\n\nContexte :\n{context_block}\n\nRéponse :"
        )
    else:
        return (
            "You are a Lebanese legal assistant. Use the previous exchange if the user refers to "
            "'that' or 'the previous answer'. Reply in English only, based strictly on the legal text.\n"
            + last_turn +
            f"Current question:\n{user_q}\n\nLegal context:\n{context_block}\n\nAnswer:"
        )



def generate_answer(model_name, prompt, temperature=0.2):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

def render_message(text: str, sender: str):
    css_class = "user" if sender == "user" else "bot"
    st.markdown(f'<div class="msg {css_class}">{text}</div>', unsafe_allow_html=True)

def render_sources(results):
    with st.expander("📚 المواد القانونية المستخدمة", expanded=False):
        for doc, meta, sim in results:
            art = meta.get("article", "غير معروفة")
            st.markdown(
                f"""
                <div class="src-card">
                  <div class="src-title">المادة {art} <span class="badge">تشابه ≈ {sim:.2f}</span></div>
                  <div class="src-meta">قانون العقوبات اللبناني</div>
                  <div style="margin-top: 6px;">{(doc[:1200] + "…") if len(doc) > 1200 else doc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def log_interaction(question, answer, results):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "chat_logs.jsonl")

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "retrieved_articles": [meta.get("article") for _, meta, _ in results],
        "feedback": st.session_state.get("last_feedback", None),  # ✅ link feedback
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # Reset feedback state after logging
    st.session_state["last_feedback"] = None


def append_feedback(helpful: bool, question: str, answer: str):
    os.makedirs("logs", exist_ok=True)
    fb = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "helpful": helpful
    }
    with open("logs/feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(fb, ensure_ascii=False) + "\n")

# ==============================
# 3) SIDEBAR (settings + info)
# ==============================
with st.sidebar:
    st.header("⚙️ الإعدادات")
    model_name = st.selectbox("نموذج الإجابة", [OPENAI_MODEL_DEFAULT, "gpt-4o"], index=0)
    temperature = st.slider("Temperature (الدقة مقابل الإبداع)", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("عدد المواد المسترجعة", 1, 6, 3, 1)

    st.markdown("---")
    st.header("ℹ️ معلومات")
    st.write(f"نمذجة الاسترجاع: **{EMBED_MODEL_NAME}**")
    st.write(f"قاعدة المتجهات: **{COLLECTION_NAME}**")
    if TOTAL_ARTICLES:
        st.write(f"عدد المواد المفهرسة: **{TOTAL_ARTICLES}**")

    st.markdown("---")
    if st.button("🧹 مسح المحادثة"):
        st.session_state["history"] = []
        st.rerun()

# ==============================
# 4) HEADER
# ==============================
colA, colB = st.columns([0.1, 0.9])
with colA:
    st.write(" ")
with colB:
    st.title("⚖️ المساعد القانوني – قانون العقوبات اللبناني")
    st.caption("إجابات قانونية عربية دقيقة مع استشهاد مباشر بالمواد ذات الصلة.")

# ==============================
# 5) SESSION HISTORY
# ==============================
if "history" not in st.session_state:
    st.session_state["history"] = []  # each item: {"role": "user"/"assistant", "content": str}
if "conversation_memory" not in st.session_state:
    st.session_state["conversation_memory"] = []
if "feedback_given" not in st.session_state:
    st.session_state["feedback_given"] = False
if "last_feedback" not in st.session_state:
    st.session_state["last_feedback"] = None

# render previous messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state["history"]:
    render_message(msg["content"], msg["role"])
st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# 6) CHAT INPUT + PIPELINE
# ==============================
user_q = st.chat_input("اكتب سؤالك القانوني… (اضغط Enter للإرسال)")
if user_q:
    # 6.1 Show user bubble + add to history
    st.session_state["history"].append({"role": "user", "content": user_q})
    render_message(user_q, "user")

    # 6.2 Retrieve + answer
    try:
        with st.spinner("جارٍ تحليل المواد القانونية واستدعاؤها…"):
            # 🌍 Detect language + translate for retrieval
            translated_query, detected_lang = preprocess_query(user_q)

            # Retrieve using Arabic query
            results = retrieve_context(translated_query, top_k=3)

            #  Build a multilingual-aware prompt
            ctx = "\n\n".join([f"[المادة {meta.get('article', '?')}] {doc}" for doc, meta, _ in results])
            prompt = build_prompt_multilingual(user_q, ctx, detected_lang)

            answer = generate_answer(model_name, prompt, temperature=temperature)

        # 6.3 Show bot bubble + store
        render_message(answer, "assistant")
        st.session_state["history"].append({"role": "assistant", "content": answer})
        st.session_state["conversation_memory"].append({
          "question": user_q,
          "answer": answer,
          "lang": detected_lang
        })

        # 6.4 Sources
        render_sources(results)
        # Save last interaction for feedback use later
        st.session_state["last_question"] = user_q
        st.session_state["last_answer"] = answer
        st.session_state["last_results"] = results
        st.session_state["feedback_given"] = False
       
    except Exception as e:
        st.error("❌ حدث خطأ أثناء توليد الإجابة.")
        st.code("".join(traceback.format_exc()), language="python")
# 6.5 Feedback buttons (always visible for the last answer)
if "last_answer" in st.session_state and st.session_state["last_answer"]:
    st.subheader("🗳️ هل كانت هذه الإجابة مفيدة؟")
    col1, col2 = st.columns(2)

    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)
    feedback_path = os.path.join("logs", "feedback.jsonl")

    # Initialize feedback state
    if "feedback_given" not in st.session_state:
        st.session_state["feedback_given"] = False
    if "last_feedback" not in st.session_state:
        st.session_state["last_feedback"] = None

    # Handle buttons
    clicked_yes = col1.button("👍 نعم", disabled=st.session_state["feedback_given"])
    clicked_no = col2.button("👎 لا", disabled=st.session_state["feedback_given"])

    # 👍 Positive feedback
    if clicked_yes and not st.session_state["feedback_given"]:
        st.session_state["feedback_given"] = True
        st.session_state["last_feedback"] = True

        feedback = {
            "question": st.session_state.get("last_question"),
            "answer": st.session_state.get("last_answer"),
            "helpful": True,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
        st.success("✅ شكراً! تم تسجيل ملاحظتك.")

    # 👎 Negative feedback
    elif clicked_no and not st.session_state["feedback_given"]:
        st.session_state["feedback_given"] = True
        st.session_state["last_feedback"] = False

        feedback = {
            "question": st.session_state.get("last_question"),
            "answer": st.session_state.get("last_answer"),
            "helpful": False,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
        st.warning("📩 تم تسجيل ملاحظتك. سنعمل على تحسين الإجابة القادمة.")

        # 6.6 Always log the interaction (even without feedback)
log_interaction(
    st.session_state.get("last_question"),
    st.session_state.get("last_answer"),
    st.session_state.get("last_results", [])
    )

# ==============================
# 7) FOOTER
# ==============================
st.markdown(
    """
    <div class="footer">
      تنبيه: هذا المساعد يعتمد على نصوص قانون العقوبات اللبناني المفهرسة. 
      إذا لم يظهر نص المادة في "📚 المواد القانونية المستخدمة"، فالإجابة لا تُعدّ مرجعاً قانونياً نهائياً.
    </div>
    """,
    unsafe_allow_html=True
)
