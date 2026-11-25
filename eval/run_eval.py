"""Run:  python -m eval.run_eval [--llm]
Checks: risk level, retrieval coverage, groundedness (all option sources valid), no banned phrases."""
import json, pathlib, sys
from safesense import rag, pipeline, llm

use_llm = "--llm" in sys.argv and llm.available()
ret = rag.Retriever()
rows = json.loads((pathlib.Path(__file__).parent / "scenarios.json").read_text())
passed = 0
for s in rows:
    res = pipeline.analyze(s["text"], constraints=s["constraints"], retriever=ret, use_llm=use_llm)
    topics = {d["topic"] for d in res["sources"].values()}
    checks = {
        "risk_level": res["risk"]["level"] == s["expect_level"],
        "retrieval": all(t in topics for t in s["expect_topics"]),
        "grounded": all(o.get("grounded") for o in res["result"]["options"]),
        "no_banned_phrases": not any("Overconfident" in w for w in res["result"]["warnings"]),
    }
    ok = all(checks.values()); passed += ok
    print(f"{s['id']} {'PASS' if ok else 'FAIL'}  {checks}")
print(f"\n{passed}/{len(rows)} passed  (mode: {'llm' if use_llm else 'demo'}, retrieval: {ret.mode})")
