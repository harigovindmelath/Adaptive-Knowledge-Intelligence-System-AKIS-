import streamlit as st
import requests
from typing import List, Dict, Any

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="AKIS – Adaptive Knowledge Intelligence System", layout="centered")

# --- Helper Functions ---
def call_api(query: str) -> Dict[str, Any]:
    try:
        resp = requests.post(API_URL, json={"query": query}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def render_confidence(confidence: float):
    color = "green" if confidence >= 80 else "orange" if confidence >= 50 else "red"
    st.markdown(f"**Confidence:** <span style='color:{color};font-size:1.2em'>{confidence:.0f}%</span>", unsafe_allow_html=True)

def render_claims(claims: List[Dict]):
    st.subheader("Claim Analysis")
    if not claims:
        st.info("No claims extracted.")
        return
    for c in claims:
        status = c.get("supported", False)
        icon = "✔️" if status else "❌"
        color = "green" if status else "red"
        conf = c.get("confidence", 0.0)
        chunk_id = c.get("source_chunk_id") or "-"
        st.markdown(f"<div style='margin-bottom:0.5em'>"
                    f"<span style='color:{color};font-size:1.2em'>{icon}</span> "
                    f"<b>{c.get('claim','')}</b> "
                    f"<span style='color:gray'>(Confidence: {conf:.2f}, Source: {chunk_id})</span>"
                    f"</div>", unsafe_allow_html=True)

def render_sources(sources: List[Dict]):
    st.subheader("Retrieved Context")
    if not sources:
        st.info("No sources retrieved.")
        return
    for s in sources:
        with st.expander(f"Chunk {s.get('chunk_id','-')} | Source: {s.get('source','-')}"):
            st.write(s.get("text", ""))

# --- UI Layout ---
st.title("AKIS – Adaptive Knowledge Intelligence System")
st.caption("Reliable AI with Explainable Answers")

query = st.text_input("Enter your question:", "")
submit = st.button("Submit")

if submit:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            result = call_api(query)
        if "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            st.markdown("---")
            st.subheader("Answer")
            answer = result.get("answer", "")
            st.markdown(f"<div style='background:#f5f5f5;padding:1em;border-radius:8px;font-size:1.3em'>{answer}</div>", unsafe_allow_html=True)
            confidence = result.get("confidence", 0.0)
            render_confidence(confidence)
            model = result.get("model_used", "-")
            model_str = "Ollama (Local)" if model == "ollama" else "Gemini (Fallback)" if model == "gemini" else model
            st.markdown(f"**Model Used:** {model_str}")
            status = result.get("status", "-")
            status_color = {"SUCCESS": "green", "FALLBACK_USED": "orange", "FAILED": "red"}.get(status, "gray")
            st.markdown(f"**Status:** <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
            render_claims(result.get("claims", []))
            render_sources(result.get("sources", []))
