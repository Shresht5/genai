from dotenv import load_dotenv
load_dotenv()
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rich import print 

#1 creating a tool 

@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

tools = {
    "get_text_length" : get_text_length
}
llm = ChatMistralAI(model = "ministral-8b-2410")

#tool binding 
llm_with_tool = llm.bind_tools([get_text_length])

message = []
prompt = input("You: ")
query = HumanMessage(prompt)
message.append(query)

result = llm_with_tool.invoke(message)
# result2 = llm.invoke(message)
# print(result)
# print("++++++++++++++++++++++++++++++++++")
# print(result2)

message.append(result)

if result.tool_calls:#tool calling
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
    result = llm_with_tool.invoke(message)
    

print(result.content)
# print("++++++++++++++++++++++++++++++++++")

# print(message)






#custom tool
# @tool #decorator for creating tool 
# def get_greeting(name : str) -> str: #type hints
#     """Generate a greeting message for a user""" #docstring

#     return f"Hello {name}, Welcome to the AI world"


# result = get_greeting.invoke({"name":"akarsh"})
# print(result)

# print(get_greeting.name)
# print(get_greeting.description)
# print(get_greeting.args)



















#search api with aiprompt
# search_tool = TavilySearch(max_result = 5) #***

# llm = ChatMistralAI(model = "ministral-8b-2410")

# prompt = ChatPromptTemplate.from_template("""
# You are a helpful assistant
# summarize the following news into clear bullet points
# {news}""")

# chain = prompt | llm | StrOutputParser()

# news_result = search_tool.run("Latest AI news of 2026 ")

# result = chain.invoke({"news" : news_result})

# print(result)
# print()
# print('tool description:  \n')

# print(search_tool.description)
# print()
# print('tool name:  \n')

# print(search_tool.name)
# print()
# print('tool ... :  \n')
# print(search_tool.args) 