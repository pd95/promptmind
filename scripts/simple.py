from langchain.chat_models import init_chat_model

model = "phi4-mini"
llm_model = init_chat_model(model=model, model_provider="ollama")

user_query = "What is python in AI? Respond with 3 bullet points."
result = llm_model.invoke(user_query)
result.pretty_print()