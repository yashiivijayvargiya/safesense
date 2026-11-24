import os, pathlib
import streamlit as st
from safesense import llm, pipeline, rag

env = pathlib.Path(".env")
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if v.strip(): os.environ.setdefault(k.strip(), v.strip())

st.set_page_config(page_title="SafeSense", page_icon="🛡️", layout="centered")

@st.cache_resource
def get_retriever(): return rag.Retriever()
retriever = get_retriever()

with st.sidebar:
    st.link_button("⚡ Quick exit", "https://www.google.com", use_container_width=True)
    st.caption("Closes this page. Also clear your browser history if the other person can see your device.")
    use_llm = st.toggle("LLM mode (Gemini)", value=llm.available(), disabled=not llm.available())
    if not llm.available():
        st.caption("Add GEMINI_API_KEY to .env for LLM mode. Demo mode uses keyword rules + retrieval only.")
    st.caption(f"Retrieval: {retriever.mode}")
    st.divider()
    st.caption("🔒 Nothing is stored. Closing the tab clears everything. Use only synthetic scenarios in this prototype.")

st.title("SafeSense 🛡️")
st.caption("GenAI-powered safety planning & decision-support assistant")
st.info("If you are in immediate danger, call **112** now. SafeSense gives information and options — it is not legal, medical or emergency advice, and the decisions are always yours.")

story = st.text_area("Describe the situation in your own words", height=150,
    placeholder="Example: My former partner keeps threatening to share my private photographs if I don't meet him.")
audio = st.audio_input("…or record it") if hasattr(st, "audio_input") else None
files = st.file_uploader("Screenshots / chat logs (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
context = st.text_input("Optional context", placeholder="e.g. ongoing for 2 weeks, former partner, Uttarakhand")
constraints = st.multiselect("Anything you want me to respect?", [
    "I can't tell my family", "I don't want to involve the police right now",
    "I share a home with this person", "My phone/accounts may be monitored", "I have limited money"])
extra = st.text_input("Other constraints (optional)")

if st.button("Analyze situation", type="primary", use_container_width=True):
    media = [(f.getvalue(), f.type) for f in files] if (files and use_llm) else None
    if audio is not None and use_llm:
        with st.spinner("Transcribing…"):
            story = (story + "\n" + llm.transcribe(audio.getvalue(), "audio/wav")).strip()
    if not story.strip():
        st.warning("Please describe what happened first."); st.stop()
    with st.spinner("Thinking carefully…"):
        st.session_state.res = pipeline.analyze(
            story, context, "; ".join(constraints + ([extra] if extra else [])),
            media=media, retriever=retriever, use_llm=use_llm)

res = st.session_state.get("res")
if res:
    r, rk = res["result"], res["risk"]
    if rk["level"] == "emergency":
        st.error("⚠️ What you describe may involve immediate danger. If you can, call **112** now or go to a safe place. "
                 "Help is also available on **181** (Women Helpline). The rest of this page can wait.")
    elif rk["level"] == "elevated":
        st.warning("Some patterns you describe can escalate. The options below may help you plan.")
    if res["error"]: st.caption(f"LLM unavailable ({res['error'][:80]}); showing safe fallback.")
    for w in r.get("warnings", []): st.caption(f"⚙️ Guardrail: {w}")

    st.header("Situation summary"); st.write(r.get("summary", ""))
    st.header("Safety concerns")
    for c in r.get("concerns", []): st.markdown(f"- **{c['label']}** — {c.get('why','')}")
    st.header("Possible options")
    for i, o in enumerate(r.get("options", []), 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {o['title']}**\n\n{o['why']}")
            st.markdown(f"✅ **Benefits:** {o['benefits']}\n\n⚠️ **Risks / trade-offs:** {o['risks']}")
            st.caption("Sources: " + (", ".join(o["sources"]) if o["sources"] else "⚠️ none — treat with caution"))
    if r.get("uncertainty_notes"):
        st.header("What I'm not sure about")
        for n in r["uncertainty_notes"]: st.markdown(f"- {n}")
    st.header("Relevant resources")
    for d in res["sources"].values():
        with st.expander(f"{d['title']}  [{d['id']}]"):
            st.write(d["text"]); st.caption(f"{d['source']} · last verified: {d['last_verified']}")
    st.header("Personalized safety plan")
    for i, s in enumerate(r.get("plan", []), 1): st.markdown(f"{i}. **{s['step']}** — {s['detail']}")
    for q in r.get("clarifying_questions", []): st.caption(f"💬 {q}")
    st.download_button("Download plan (.md)", pipeline.plan_markdown(res), "safety_plan.md")
    with st.expander("Debug: extracted facts & risk flags"):
        st.json({"risk": rk, "facts": res["facts"], "mode": res["mode"]})
