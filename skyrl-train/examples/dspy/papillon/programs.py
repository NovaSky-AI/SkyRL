import dspy
from dspy.adapters.xml_adapter import XMLAdapter


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
    def __init__(self, untrusted_model=dspy.LM(model="openai/gpt-4.1-mini"), cache=True):
        self.craft_redacted_request = dspy.ChainOfThought(CraftRedactedRequest)
        self.respond_to_query = dspy.Predict(RespondToQuery)
        self.untrusted_model = untrusted_model
        self.adapter = XMLAdapter()

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

class PAPILLON_request_gen(PAPILLON):
    def __init__(self):
        super().__init__()
        request_lm = dspy.LM(
            model="openai/Qwen/Qwen2.5-0.5B-Instruct",
            api_base="http://0.0.0.0:8002/v1",
            api_key="fake-key",
            temperature=1.0,
            model_type="chat",
            max_tokens=4096,
            cache=False,
        )
        self.craft_redacted_request.set_lm(request_lm)
        self.respond_to_query.set_lm(request_lm)
        
    async def forward(self, example):
        user_query = example.get("user_query")
        print("[Program] Generating LLM request")
        self.llm_request = await self.craft_redacted_request.acall(user_query=user_query)
        llm_request = self.llm_request.llm_request
        print("[Program] Generating untrusted response")
        llm_response = await self.untrusted_model.acall(llm_request)
        llm_response = llm_response[0]
        print("[Program] Generating response")
        response = await self.respond_to_query.acall(
            related_llm_request=llm_request,
            related_llm_response=llm_response,
            user_query=user_query,
        )
        response = response.response

        return dspy.Prediction(
            llm_request=llm_request,
            llm_response=llm_response,
            response=response,
        )
    
    def collect_trace(self, kwargs, pred):
        # Get formatted finetune data which contains both input and output messages
        import pdb; pdb.set_trace()
        finetune_data = self.adapter.format_finetune_data(
                                signature=self.craft_redacted_request.predictors()[0].signature,
                                inputs=kwargs,
                                outputs=self.llm_request,
                                demos=[] # TODO: Add support for demos
                            )
        
        all_messages = finetune_data.get('messages', [])
        
        return all_messages, self.llm_request

    
    