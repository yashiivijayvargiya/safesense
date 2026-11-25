"""Deterministic red-flag layer. Runs BEFORE and independently of the LLM."""
import re

RULES = {
    "emergency": {
        "weapon": r"\b(gun|knife|weapon|acid|pistol)\b",
        "strangulation": r"\b(strangl|chok(e|ed|ing)|choked)\b",
        "kill_threat": r"\b(kill (me|him|her|us|myself)|going to kill|threaten(ed|s)? to kill|murder)\b",
        "self_harm": r"\b(suicid|end my life|kill myself|want to die|hurt myself)\b",
        "imminent": r"\b(outside my (house|home|door)|at my door|breaking in|right now|is following me now|locked me|won'?t let me leave|trapped)\b",
        "sexual_violence": r"\b(raped|forced me to have sex|sexual assault)\b",
    },
    "elevated": {
        "stalking": r"\b(stalk|follow(ing|ed)? me|keeps? showing up|watching me|tracking me|tracker|gps)\b",
        "intimate_images": r"\b(private (photo|picture|video|image)s?|nude|intimate (photo|image|video)|leak|morph(ed)?|share my (photo|video|pictures?))\b",
        "blackmail": r"\b(blackmail|extort|or else|if i don'?t (meet|send|pay|come))\b",
        "escalation": r"\b(getting worse|more often|escalat|threatened again)\b",
        "physical_abuse": r"\b(hit me|beat me|beats me|slapp|punch|pushed me|physically)\b",
        "financial_control": r"\b(controls? my (money|salary|phone)|takes? my (salary|phone)|won'?t let me work)\b",
        "online_harassment": r"\b(troll|dox|fake (profile|account)|abusive (dm|messages?)|online harass)\b",
    },
}

def assess(*texts: str) -> dict:
    blob = " ".join(t or "" for t in texts).lower()
    flags = {lvl: [k for k, p in rules.items() if re.search(p, blob)] for lvl, rules in RULES.items()}
    level = "emergency" if flags["emergency"] else "elevated" if flags["elevated"] else "standard"
    return {"level": level, "emergency_flags": flags["emergency"], "elevated_flags": flags["elevated"]}
