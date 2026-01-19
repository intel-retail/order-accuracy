from semantic_matcher import semantic_match

def validate_order(expected_items, detected_items, vlm_pipeline):
    missing = []
    extra = []
    quantity_mismatch = []
    matched_detected = set()

    # ---- Pass 1: exact match ----
    for exp in expected_items:
        exp_name = exp["name"].lower()
        exp_qty = exp["quantity"]

        found = False
        for det in detected_items:
            det_name = det["name"].lower()
            det_qty = det["quantity"]

            if det_name == exp_name:
                found = True
                matched_detected.add(det_name)
                if det_qty != exp_qty:
                    quantity_mismatch.append({
                        "name": exp_name,
                        "expected": exp_qty,
                        "detected": det_qty
                    })
                break

        if not found:
            missing.append(exp)

    # ---- Pass 2: semantic match on leftovers ----
    if missing:
        still_missing = []

        for exp in missing:
            exp_name = exp["name"].lower()
            exp_qty = exp["quantity"]
            matched = False

            for det in detected_items:
                det_name = det["name"].lower()
                if det_name in matched_detected:
                    continue

                if semantic_match(vlm_pipeline, exp_name, det_name):
                    matched = True
                    matched_detected.add(det_name)
                    if det["quantity"] != exp_qty:
                        quantity_mismatch.append({
                            "name": exp_name,
                            "expected": exp_qty,
                            "detected": det["quantity"]
                        })
                    break

            if not matched:
                still_missing.append(exp)

        missing = still_missing

    # ---- Extras ----
    for det in detected_items:
        if det["name"].lower() not in matched_detected:
            extra.append(det)

    return {
        "missing": missing,
        "extra": extra,
        "quantity_mismatch": quantity_mismatch
    }
