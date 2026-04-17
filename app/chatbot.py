from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from app.services.get_model import get_llm
from app.services.get_prompt import get_rag_prompt
from app.services.get_session import get_memory
from app.services.retriever import hybrid_retrieve
from app.core.logger import get_logger

log = get_logger(__name__)


def _format_context(docs) -> str:
    """Join retrieved doc chunks into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def chat_bot(question: str, session_id: str) -> dict:
    log.info(f"session={session_id} | question='{question[:80]}'")

    # Stage 1-5: Hybrid retrieval
    docs = hybrid_retrieve(question)
    context = _format_context(docs)
    log.info(f"Context built from {len(docs)} docs.")

    # Build chain
    prompt = get_rag_prompt()
    llm = get_llm()

    # Inject context + question, keep chat_history from memory
    chain = RunnablePassthrough.assign(
        answer=prompt | llm | StrOutputParser()
    )

    main_chain = RunnableWithMessageHistory(
        chain,
        get_memory,
        output_messages_key="answer",
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    response = main_chain.invoke(
        {"question": question, "context": context},
        config={"configurable": {"session_id": session_id}},
    )

    log.info(f"Response generated for session={session_id}")
    return {
        "session_id": session_id,
        "question": question,
        "answer": response["answer"],
        "sources": [doc.metadata.get("source", "unknown") for doc in docs],
    }
