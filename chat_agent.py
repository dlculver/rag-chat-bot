# import libraries
import os
import openai
import asyncio

# llama index libraries
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks.base import CallbackManager
from llama_index.llms.openai import OpenAI

# chain lit libraries
import chainlit as cl
from chainlit.input_widget import Select, TextInput

# your libraries
from index_wikipages import create_index


index = None
agent = None

@cl.on_chat_start
async def on_chat_start():
    print("on_chat_start")
    global index
    # Settings
    settings = await cl.ChatSettings(
        [
            Select(
                id= "MODEL",
                label= "Choose which model to use.",
                values=["gpt-3.5-turbo", "gpt-4"],
                initial_index=0,
            ),
            
            TextInput(
                id="WikiPageRequest", 
                label="Request Wikipage"
            ),
        ]
    ).send()

def wikisearch_engine(index):
    print("Creating query engine...")
    query_engine = index.as_query_engine(
        response_mode="compact", 
        verbose=True,
        similarity_top_k=10,
    )
    return query_engine

def create_react_agent(MODEL):
    query_engine_tools =[
        QueryEngineTool(
            query_engine=wikisearch_engine(index), 
            metadata=ToolMetadata(name="Wikipedia",
                                  description="Useful for performing searches on Wikipedia pages."
                                ),
        )
    ]

    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(model=MODEL)
    agent = ReActAgent(
        tools=query_engine_tools, 
        memory=None,
        llm=llm, 
        callback_manager=CallbackManager([
            # Add your callbacks here
            cl.LlamaIndexCallbackHandler()
        ]), 
        verbose=True
    )

    return agent

@cl.on_settings_update
async def setup_agent(settings):
    print("Setting up agent...")
    global agent
    global index
    query = settings["WikiPageRequest"]
    index = create_index(query)

    print("on_settings_update", settings)
    MODEL = settings["MODEL"]
    agent = create_react_agent(MODEL)
    await cl.Message(
        author="Agent", content=f"""Wikipage(s) "{query}" successfully indexed"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    print("on_message main function...")
    print(f"Message received: {message.content}")
    if agent:
        print(f"Agent: {agent}")
        print(f"Agent.chat: {agent.chat}")
        print(f"Awaiting agent.chat({message.content})...")
        response = await cl.make_async(agent.chat)(message.content)
        await cl.Message(author="Agent", content=response).send()

