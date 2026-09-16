from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.7,
    },
)
model = ChatHuggingFace(llm=llm)

res = model.invoke("What is cricket?")

print(res)