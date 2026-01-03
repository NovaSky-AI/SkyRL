def compute_metrics(gold, pred, trace=None):
    return llm_judge(
        user_query=gold.user_query,
        new_resp=pred.output,
        og_resp=gold.target_response,
        updated_query=pred.prompt,
        pii_str=gold.pii_str,
    )

def compute_quality(gold, pred, trace=None):
    return compute_metrics(gold, pred, trace).quality

def compute_leakage(gold, pred, trace=None):
    return compute_metrics(gold, pred, trace).leakage

def compute_overall_score(gold, pred, trace=None):
    metrics = compute_metrics(gold, pred, trace)
    overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
    return overall_score >= 1.0 if trace is not None else overall_score


def compute_query_leakage(example, pred, trace=None):
    """
    Compute privacy leakage of the redacted LLM request.

    Leakage is defined as the fraction of PII units from `gold`
    that appear verbatim in `pred.llm_request`.

    Returns a reward in [0, 1], where:
      1.0 = no PII leaked
      0.0 = all PII leaked
    """
    # Extract PII units (list of strings)
    pii_units = example.get("pii_units", [])
    if not pii_units:
        return 1.0  # no PII to leak

    llm_request = pred.llm_request.lower()

    leaked = 0
    for pii in pii_units:
        if pii and pii.lower() in llm_request:
            leaked += 1

    leakage_ratio = leaked / len(pii_units)
    reward = 1.0 - leakage_ratio

    return reward
