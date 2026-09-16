from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader# txt loader
from langchain_core.prompts import ChatPromptTemplate # 3 mwssage temple s/h/ai
from langchain_community.document_loaders import PyPDFLoader# pdf loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# data = TextLoader("2/notes.txt")
data= PyPDFLoader("2/GRU.pdf")
docs = data.load()
print("docs loaded")
print(len(docs))
templete=ChatPromptTemplate.from_messages([
    ("system","you are a AI that summarizes the text"),
    ("human","{data}")
])

model = ChatMistralAI(model = 'ministral-8b-2410')
prompt=templete.format_messages(data=docs)
res=model.invoke(prompt)
print(res.content)