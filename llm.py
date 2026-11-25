"""Single provider wrapper. To switch to OpenAI, only edit generate() and embed()."""
import json, os, re
import numpy as np

GEN_MODEL = os.getenv("SAFESENSE_MODEL", "gemini-2.5-flash")
EMB_MODEL = os.getenv("SAFESENSE_EMBED_MODEL", "gemini-embedding-001")

def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))

def _client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def _safety():
    # Abuse-related text often trips default filters. Loosen only harassment/hate;
    # keep dangerous-content & sexual categories at defaults. Test with your scenarios.
    from google.genai import types
    cats = ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH"]
    return [types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH") for c in cats]

def generate(prompt: str, system: str | None = None, json_mode: bool = False,
             media: list[tuple[bytes, str]] | None = None, temperature: float = 0.2):
    """media = [(bytes, mime_type)] for screenshots/audio. Returns str, or dict if json_mode."""
    from google.genai import types
    parts = [types.Part.from_bytes(data=b, mime_type=m) for b, m in (media or [])]
    parts.append(prompt)
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
        safety_settings=_safety())
    resp = _client().models.generate_content(model=GEN_MODEL, contents=parts, config=cfg)
    text = (resp.text or "").strip()
    if not json_mode:
        return text
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    return json.loads(text)

def transcribe(audio: bytes, mime: str = "audio/wav") -> str:
    return generate("Transcribe this audio exactly. Output only the transcript.",
                    media=[(audio, mime)], temperature=0.0)

def embed(texts: list[str]) -> np.ndarray:
    res = _client().models.embed_content(model=EMB_MODEL, contents=texts)
    v = np.array([e.values for e in res.embeddings], dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)
