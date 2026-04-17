from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)

SYSTEM_PROMPT = """You are LoanIQ, an expert mortgage underwriting assistant.
Answer the user's question using ONLY the context provided below.
Be concise, accurate, and cite relevant facts from the context.
If the context does not contain enough information, say "I don't have enough information in the indexed documents to answer this."

Context:
{context}
"""


def get_rag_prompt() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("Question: {question}"),
        ]
    )
    return prompt


def get_chatbot_prompt() -> ChatPromptTemplate:
    """Fallback prompt when no context (no docs indexed yet)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are LoanIQ, a mortgage underwriting AI assistant. "
                "Answer questions directly and concisely."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("Question: {question}"),
        ]
    )
    return prompt
