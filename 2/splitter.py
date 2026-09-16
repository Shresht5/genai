from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader# pdf loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import CharacterTextSplitter

# splitter = CharacterTextSplitter(
#     separator= "",# default '\n\n'
#     chunk_size = 100,
#     chunk_overlap=1
# )#1

# splitter= TokenTextSplitter(
#     chunk_size=100,
#     chunk_overlap=10
#)#2

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)#3

data= PyPDFLoader("2/GRU.pdf")
# data = TextLoader("2/notes.txt")

docs = data.load()

chunks = splitter.split_documents(docs)

for i in chunks[:5]:
    print(i.page_content)
    print()
    print()
    print()
    
print(len(chunks))