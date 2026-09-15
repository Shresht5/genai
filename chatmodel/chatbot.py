import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={        
        "temperature": 0.7,
    },
)
model = ChatHuggingFace(llm=llm)

messages=[]

def fix_message(response):
    answer = response.split("<|im_start|>assistant")[-1]
    answer = answer.split("<|im_end|>")[0].strip()
    return (answer)

print("-----------________-----------")
print("Type 0 to exit")

while True:
    prompt=input("you : ")
    if(prompt=="0"):
        break
    messages.append(HumanMessage(content=prompt))
    res=model.invoke(messages)
    msg=fix_message(res.content)
    print("bot : ",msg)
    messages.append(AIMessage(content=msg))
    
print(messages)