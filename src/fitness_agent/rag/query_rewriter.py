from fitness_agent.llm import get_chat_model

REWRITE_PROMPT = """Rewrite the user's question as a short search query for a fitness knowledge base.
Keep the key fitness terms, expand abbreviations, and drop filler words.
Return only the search query.

Question: {question}"""


def rewrite_query(question: str) -> str:
    response = get_chat_model().invoke(REWRITE_PROMPT.format(question=question))
    return response.text.strip().strip('"')
