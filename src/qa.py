from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def load_qa():
    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(
        "rag_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )
