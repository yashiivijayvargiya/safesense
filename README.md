# SafeSense
GenAI safety-planning & decision-support assistant (Streamlit + Gemini + RAG).

## Run
```
pip install -r requirements.txt
cp .env.example .env        # add GEMINI_API_KEY for LLM mode (optional)
python -m streamlit run app.py
python -m eval.run_eval          # demo mode
python -m eval.run_eval --llm    # with Gemini
```
## Architecture
1. `risk.py`: deterministic red-flag rules (run first, independent of the LLM)
2. `pipeline.extract`: LLM -> structured JSON facts
3. `rag.py`: retrieve from `knowledge_base.json` (Gemini embeddings or TF-IDF fallback)
4. `pipeline.generate_guidance`: grounded options/risks/plan with source ids
5. `pipeline.validate`: drops invalid citations, flags ungrounded options and directive phrases
6. Fail-safe: any LLM error falls back to emergency resources.

## Before any real use
- Every KB entry has `last_verified: "VERIFY"`: check each number, URL and law, then add the date.
- Have a counsellor / legal-aid professional review the KB and sample outputs.
- Use only synthetic scenarios with free-tier API keys. Add a paid tier / data-protection terms before real users.
- Extend `eval/scenarios.json` to 30-50 cases (vague, minimizing, Hindi/Hinglish, immediate danger).
