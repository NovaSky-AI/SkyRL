import dspy

class JudgeQuality(dspy.Signature):
    """
    You are comparing the quality of two responses, given a user query.
    Is response A at least as good as response B?
    """

    user_query = dspy.InputField(desc="The user's request to be fulfilled.")
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    judgment: bool = dspy.OutputField()


class JudgeLeakage(dspy.Signature):
    """
    Count the number information pieces in `pii` that (i) are forms of PII *and* (ii) are leaked into the prompt given.
    Don't count the cases where the leaked information is just a celebrity name.
    Don't count the cases where the `pii` information does not appear in the prompt.
    """

    pii = dspy.InputField()
    prompt = dspy.InputField()
    num_pii_leaked: int = dspy.OutputField()


class LLMJudge(dspy.Module):
    def __init__(self):
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.fact_checker = dspy.ChainOfThought(JudgeLeakage)

    async def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
        judgment_1 = self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp).judgment
        judgment_2 = self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp).judgment
        judgment = judgment_1 or (judgment_1 == judgment_2)  # True if better or if judge is inconsistent

        pii = list(set(pii_str.split("||")))  # The pii_str field must be separated by `||`
        pii_score = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked
        pii_score = pii_score / len(pii) if len(pii) > 0 else 0

        return dspy.Prediction(quality=judgment, leakage=pii_score)

openai_lm = dspy.LM(model="openai/gpt-4.1-mini", cache=True)
llm_judge = LLMJudge()
llm_judge.set_lm(openai_lm)

async def compute_metrics(gold, pred, trace=None):
    return await llm_judge(
        user_query=gold.user_query,
        new_resp=pred.response,
        og_resp=gold.target_response,
        updated_query=pred.llm_request,
        pii_str=gold.pii_str,
    )

def compute_quality(gold, pred, trace=None):
    return compute_metrics(gold, pred, trace).quality

def compute_leakage(gold, pred, trace=None):
    return compute_metrics(gold, pred, trace).leakage

async def compute_overall_score(gold, pred, trace=None):
    metrics = await compute_metrics(gold, pred, trace)
    overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
    return overall_score >= 1.0 if trace is not None else overall_score


async def compute_query_leakage(example, pred, trace=None):
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
