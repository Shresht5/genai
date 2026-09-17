from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda , RunnablePassthrough

# Components
model = ChatMistralAI(model="ministral-8b-2410")
parser = StrOutputParser()

#Runable passthrough
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms"),
    ("human", "Explain the following code in simple words:\n{code}")#code actually is whole prompt of first invoke
])

seq = code_prompt | model | parser 

seq2 = RunnableParallel(
    {"code" :  RunnablePassthrough(),
     "explanation" : explain_prompt | model | parser
    }
)

chain = seq | seq2

result = chain.invoke({"topic" : "please write a code of palindrome in python "})

print(result['code'])
print(result['explanation'])

# PARALEL/LAMBDA runable
# # Two different prompts
# short_prompt = ChatPromptTemplate.from_template(
#     "Explain {topic} in 1-2 lines"
# )

# detailed_prompt = ChatPromptTemplate.from_template(
#     "Explain {topic} in detail"
# )

# # Input
# topic = "Machine Learning"

# chain = RunnableParallel({
#     "short" :RunnableLambda(lambda x :x['short']) |short_prompt | model | parser ,
#     "detailed" :RunnableLambda(lambda x: x['detailed']) |detailed_prompt |model |parser
# })

# result = chain.invoke({
#     "short" : {"topic":"Machine Learning"},
#     "detailed" : {"topic":"Deep Learning"}
# })

# print(result['short'])
# print(result['detailed'])