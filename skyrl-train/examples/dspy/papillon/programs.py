import dspy


class CraftRedactedRequest(dspy.Signature):
    """
    Given a private user query, create a privacy-preserving request for a powerful external LLM.
    The LLM may assist without learning private information about the user.
    """

    user_query = dspy.InputField()
    llm_request = dspy.OutputField()


class RespondToQuery(dspy.Signature):
    """
    Respond to a user query.
    For inspiration, we found a potentially related request to a powerful external LLM and its response.
    """

    related_llm_request = dspy.InputField()
    related_llm_response = dspy.InputField(
        desc="information from a powerful LLM responding to a related request"
    )
    user_query = dspy.InputField(desc="the user's request you need to fulfill")
    response = dspy.OutputField(desc="your final response to the user's request")


class PAPILLON(dspy.Module):
    def __init__(self, untrusted_model):
        self.craft_redacted_request = dspy.ChainOfThought(CraftRedactedRequest)
        self.respond_to_query = dspy.Predict(RespondToQuery)
        self.untrusted_model = untrusted_model

    def forward(self, user_query):
        llm_request = self.craft_redacted_request(user_query=user_query).llm_request
        llm_response = self.untrusted_model(llm_request)[0]
        response = self.respond_to_query(
            related_llm_request=llm_request,
            related_llm_response=llm_response,
            user_query=user_query,
        ).response

        return dspy.Prediction(
            llm_request=llm_request,
            llm_response=llm_response,
            response=response,
        )
