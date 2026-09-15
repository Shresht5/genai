import langchain
from fastapi import FastAPI
from typing import Union
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

from langchain.chat_models import init_chat_model

# model = init_chat_model("open-mistral-7b", model_provider="mistralai" ,temperature=0.6,max_tokens=20)
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=100,
)
model = ChatHuggingFace(llm=llm)
res=model.invoke("what is cricket?" )
# print(model)
print(res)

# app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id,"q": q}
