import langchain
from fastapi import FastAPI
from typing import Union
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("open-mistral-7b", model_provider="mistralai")
res=model.invoke("what is cricket?")
# print(model)
print(res)

# app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id,"q": q}
