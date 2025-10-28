import streamlit as st
from openai import OpenAI
import json
import numpy as np
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import datetime
# Load environment variables
load_dotenv()
client = OpenAI()

# Connect to upgraded article-based Chroma DB
embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large"
)

chroma_client = chromadb.PersistentClient(
    path=r"C:\Users\PC\OneDrive\Desktop\chatbot\data\chroma_db_articles"
)

collection = chroma_client.get_collection(
    name="penal_code_articles",
    embedding_function=embed_func
)


# Load chunks (optional for display)
with open(r"C:\Users\PC\OneDrive\Desktop\chatbot\data\processed\penal_code_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- Helper: retrieve context ---
def retrieve_context(query: str, top_k: int = 3):
    """Retrieve top_k articles + metadata for a user query from Chroma."""
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    sims = [1 - d for d in dists]
    return list(zip(docs, metas, sims))

# --- Helper: build prompt ---
def build_prompt_ar(question, context_block):
    return (
        f"أجب على السؤال التالي اعتماداً فقط على النصوص القانونية المقدمة.\n\n"
        f"السؤال: {question}\n\n"
        f"النصوص:\n{context_block}\n\n"
        "أعطني الجواب بالعربية الفصحى باختصار، مع ذكر رقم المادة القانونية المناسبة داخل أقواس مربعة."
    )

# --- Streamlit layout ---
st.set_page_config(page_title="AI Legal Chatbot – Lebanese Law", layout="wide")
st.title("⚖️ AI Legal Chatbot – قانون العقوبات اللبناني")

question = st.text_area("✍️ اكتب سؤالك القانوني هنا:", placeholder="مثال: ما هي العقوبة على السرقة؟")

if st.button("🔍 احصل على الإجابة"):
    if not question.strip():
        st.warning("الرجاء كتابة سؤال.")
    else:
        # Retrieve relevant context
        results = retrieve_context(question, top_k=3)
        ctx = "\n\n".join([f"[{cid}] {doc}" for cid, _, doc in results])
        prompt = build_prompt_ar(question, ctx)

        # Generate answer
        with st.spinner("جارٍ تحليل النص القانوني..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # or gpt-4o
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
        answer = response.choices[0].message.content

        # --- Display answer ---
        st.subheader("🧾 الجواب:")
        st.write(answer)

        with st.expander("📚 المواد القانونية المستخدمة"):
            for doc, meta, sim in results:
                art = meta.get("article", "غير معروفة")
                st.markdown(f"**المادة {art}**  |  تشابه: {sim:.2f}")
                st.write(doc[:1200] + ("…" if len(doc) > 1200 else ""))
                st.divider()
        # --- Feedback Section ---
        st.subheader("🗳️ هل كانت هذه الإجابة مفيدة؟")
        col1, col2 = st.columns(2)

        if "feedback_given" not in st.session_state:
            st.session_state["feedback_given"] = False

        if col1.button("👍 نعم", disabled=st.session_state["feedback_given"]):
            st.session_state["feedback_given"] = True
            feedback = {"question": question, "answer": answer, "helpful": True, "timestamp": datetime.datetime.now().isoformat()}
            with open("feedback.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
                st.success("✅ شكراً! تم تسجيل ملاحظتك.")

        if col2.button("👎 لا", disabled=st.session_state["feedback_given"]):
            st.session_state["feedback_given"] = True
            feedback = {"question": question, "answer": answer, "helpful": False, "timestamp": datetime.datetime.now().isoformat()}
            with open("feedback.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
            st.warning("📩 تم تسجيل ملاحظتك. سنعمل على تحسين الإجابة القادمة.")
        # --- Log each interaction automatically ---
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "retrieved_articles": [meta.get("article") for _, meta, _ in results],
}   

        os.makedirs("logs", exist_ok=True)
        with open("logs/chat_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

