from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "rag_db",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = Ollama(model="mistral")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False
)

while True:
    q = input("Ask SideEffectNet: ")
    if q.lower() == "exit":
        break
    print(qa.run(q))
