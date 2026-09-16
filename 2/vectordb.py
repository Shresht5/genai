from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()
docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]


embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vectorStore=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory='chroma-db'
)

result = vectorStore.similarity_search("what is used for data analysis?",k=2)
for r in result:
    print(r)

retriver = vectorStore.as_retriever()

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)
  
# vector = embeddings.embed_query("hello gamers system")

# print(vectorStore)
# print(len(vectorStore))