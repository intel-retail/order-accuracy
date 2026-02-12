from openvino_genai import GenerationConfig

SEMANTIC_PROMPT = """
You are matching grocery product names.

Expected product: "{expected}"
Detected product: "{detected}"

Question:
Could these refer to the same real-world product on a grocery bill,
even if one name is shorter, simplified, or missing adjectives?

Examples:
"green apple" and "apple" -> YES
"cola" and "coca cola bottle" -> YES
"bread" and "milk" -> NO

Answer ONLY YES or NO.
"""

def semantic_match(vlm_pipeline, expected_name, detected_name):
    print(f"[SEMANTIC-MATCH] Comparing '{expected_name}' ↔ '{detected_name}'", flush=True)

    prompt = SEMANTIC_PROMPT.format(
        expected=expected_name,
        detected=detected_name
    )

    # Small deterministic config for text-only classification
    gen_config = GenerationConfig(
        max_new_tokens=8,
        temperature=0.0,
        do_sample=False
    )

    try:
        out = vlm_pipeline.generate(
            prompt,
            generation_config=gen_config
        )
        answer = out.texts[0].strip().upper()
        print(f"[SEMANTIC-MATCH] Answer: {answer}", flush=True)
        return answer.startswith("YES")

    except Exception as e:
        print("[SEMANTIC-MATCH] ERROR:", e, flush=True)
        return False
