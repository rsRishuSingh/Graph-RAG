# main.py
import os
import getpass
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain import LLMChain, PromptTemplate


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    

def main():
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",  
        temperature=0.7,
        api_key=GROQ_API_KEY         
    ) 

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Tell me a fun fact about {topic}."
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"topic": 'India'})

    print("\n🧠 Groq says:", result)

if __name__ == "__main__":
    main()




'''
import os
import getpass
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.schema import RunnableSequence

def get_groq_api_key() -> str:
    """Retrieve GROQ_API_KEY from env or prompt the user."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        key = getpass.getpass("Enter your Groq API key: ")
    return key

def main():
    # Load .env if present
    load_dotenv()

    # 1) Securely fetch API key
    groq_api_key = get_groq_api_key()

    # 2) Instantiate Groq-powered chat model
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",   # choose any Groq model
        temperature=0.7,
        api_key=groq_api_key                # explicitly pass in key
    )

    # 3) Define a PromptTemplate (for structured chat prompts, use ChatPromptTemplate similarly)
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Tell me a fun fact about {topic}."
    )

    # 4a) Old-style chain (still supported, but we’ll show the new pattern next)
    chain_legacy = LLMChain(llm=llm, prompt=prompt)
    result_legacy = chain_legacy.invoke({"topic": "India"})
    print("\n🧠 [legacy] Groq says:", result_legacy)

    # 4b) New RunnableSequence pattern (preferred for v1.0+)
    seq: RunnableSequence = prompt | llm
    result_new = seq.invoke({"topic": "India"})
    print("\n🧠 [runnable seq] Groq says:", result_new)

if __name__ == "__main__":
    main()

'''


'''
import os, getpass
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

def get_groq_key():
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        key = getpass.getpass("Enter your Groq API key: ")
    return key

# 1) init
groq_key = get_groq_key()
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=groq_key
)

# 2) build messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a fun fact about India.")
]

# 3) call
response = llm.predict_messages(messages)
print("🧠 Groq:", response.content)

'''