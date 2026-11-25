from . import llm, risk

BANNED = ["you must", "you have to", "you should confront", "guaranteed", "i promise",
          "definitely illegal", "just ignore", "nothing to worry"]

SYSTEM = """You are SafeSense, a safety-planning support assistant for people facing harassment,
threats, stalking, abuse or image-based coercion. Follow these rules strictly:
- Use ONLY the provided SOURCES for facts (helplines, laws, procedures). Cite source ids.
- If the sources don't cover something, say so in uncertainty_notes. Never invent phone numbers or laws.
- Offer OPTIONS with benefits and risks. Never command, never decide for the user, never promise outcomes.
- Never advise confronting, retaliating against, or negotiating with the abuser.
- Respect the user's stated constraints (e.g. cannot tell family, cannot go to police).
- You are not a lawyer, counsellor, or emergency service. Be warm, plain, non-judgmental.
- If immediate danger is indicated, step 1 of the plan must be contacting emergency services."""

def extract(story, context, constraints, media):
    prompt = f"""Extract facts ONLY from what the user wrote. Use null if unknown. Return JSON:
{{"relationship": str|null, "threat_types": [str], "timeline": str|null, "evidence_available": [str],
 "immediate_danger": bool, "user_constraints": [str], "location_hint": str|null}}

STORY: {story}
CONTEXT: {context}
CONSTRAINTS: {constraints}"""
    return llm.generate(prompt, system="You extract structured facts. Do not infer beyond the text.",
                        json_mode=True, media=media)

def generate_guidance(story, context, constraints, facts, rk, docs):
    src = "\n\n".join(f"[{d['id']}] {d['title']} ({d['source']}, verified {d['last_verified']})\n{d['text']}" for d in docs)
    prompt = f"""USER STORY: {story}
CONTEXT: {context}
USER CONSTRAINTS: {constraints}
EXTRACTED FACTS: {facts}
RULE-BASED RISK: {rk}

SOURCES:
{src}

Return JSON:
{{"summary": "2-3 sentence plain summary of what the user described",
 "concerns": [{{"label": str, "why": "which words/facts in the story indicate this"}}],
 "options": [{{"title": str, "why": str, "benefits": str, "risks": str, "sources": ["source ids"]}}],
 "uncertainty_notes": ["what you could not determine or the sources don't cover"],
 "plan": [{{"step": str, "detail": str}}],
 "clarifying_questions": ["max 2 gentle questions that would improve the guidance"]}}
Give 3-5 options, 4-7 plan steps. Plan must respect the constraints."""
    return llm.generate(prompt, system=SYSTEM, json_mode=True, temperature=0.3)

def demo_guidance(story, rk, docs):
    """No-API fallback: assembles output straight from retrieved sources."""
    opts = [{"title": d["title"], "why": d["text"][:260] + ("…" if len(d["text"]) > 260 else ""),
             "benefits": "See source for details.", "risks": "Consider your own safety and privacy before acting.",
             "sources": [d["id"]]} for d in docs[:4]]
    concerns = [{"label": f.replace("_", " ").title(), "why": "Detected by keyword rules (demo mode)."}
                for f in rk["emergency_flags"] + rk["elevated_flags"]]
    plan = [{"step": "Check immediate safety", "detail": "If in danger now, call 112."},
            {"step": "Preserve evidence", "detail": "Save messages/screenshots somewhere the other person cannot access."},
            {"step": "Choose a support option", "detail": "Pick the option above that feels safest and most realistic."}]
    return {"summary": "Demo mode: summary requires the LLM. Your description was analysed with keyword rules only.",
            "concerns": concerns, "options": opts, "uncertainty_notes": ["Demo mode: no LLM reasoning was used."],
            "plan": plan, "clarifying_questions": []}

def validate(out, docs):
    """Post-generation guardrails."""
    valid = {d["id"] for d in docs}
    warnings = []
    for o in out.get("options", []):
        o["sources"] = [s for s in o.get("sources", []) if s in valid]
        o["grounded"] = bool(o["sources"])
        if not o["grounded"]:
            warnings.append(f"Option '{o.get('title')}' has no supporting source.")
    blob = str(out).lower()
    for b in BANNED:
        if b in blob:
            warnings.append(f"Overconfident/directive phrase detected: '{b}'.")
    out["warnings"] = warnings
    return out

def analyze(story, context="", constraints="", evidence_text="", media=None, retriever=None, use_llm=None):
    use_llm = llm.available() if use_llm is None else use_llm
    rk = risk.assess(story, context, evidence_text)          # deterministic, always runs
    facts, error = {}, None
    try:
        if use_llm:
            facts = extract(story + "\n" + evidence_text, context, constraints, media)
            if facts.get("immediate_danger") and rk["level"] != "emergency":
                rk["level"] = "emergency"; rk["emergency_flags"].append("llm_immediate_danger")
        query = " ".join([story, context, " ".join(facts.get("threat_types", []) or [])])
        docs = retriever.search(query, k=6)
        if rk["level"] == "emergency":                       # always include emergency docs
            have = {d["id"] for d in docs}
            docs += [d for d in retriever.docs if d["topic"] == "emergency" and d["id"] not in have]
            have = {d["id"] for d in docs}
            if "self_harm" in rk["emergency_flags"]:         # always surface mental-health support
                docs += [d for d in retriever.docs if d["id"] == "KB15" and d["id"] not in have]
        out = (generate_guidance(story, context, constraints, facts, rk, docs)
               if use_llm else demo_guidance(story, rk, docs))
    except Exception as e:                                   # fail safe: never leave user empty-handed
        error = str(e)
        docs = [d for d in retriever.docs if d["topic"] == "emergency"]
        out = demo_guidance(story, rk, docs)
    out = validate(out, docs)
    return {"risk": rk, "facts": facts, "result": out, "sources": {d["id"]: d for d in docs},
            "mode": "llm" if use_llm and not error else "demo", "error": error}

def plan_markdown(res):
    r = res["result"]
    lines = ["# My Safety Plan", "", "_Information and options only — not legal, medical or emergency advice._", "",
             "## Summary", r.get("summary", ""), "", "## Steps"]
    lines += [f"{i}. **{s['step']}** — {s['detail']}" for i, s in enumerate(r.get("plan", []), 1)]
    lines += ["", "## Support contacts"]
    lines += [f"- {d['title']}: {d['text'][:200]}" for d in res["sources"].values() if d["topic"] in ("emergency", "helpline")]
    return "\n".join(lines)
