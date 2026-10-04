from fitness_agent.llm import get_chat_model

ANSWER_PROMPT = """You are a helpful fitness assistant.
Answer the question using only the context below and mention the source numbers you used, like [1].
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}"""


def generate_answer(question: str, context: str) -> str:
    response = get_chat_model().invoke(ANSWER_PROMPT.format(context=context, question=question))
    return response.text.strip()
