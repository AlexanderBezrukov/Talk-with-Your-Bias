import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .prompts import PromptTemplates
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


class Chat_Generator:
    def __init__(self, model_name:str, prompt_template:type(PromptTemplates)):
        self.prompt = prompt_template
        self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_name, temperature=0.00001)
    async def predict(self, query, session_id="session1") -> dict:

        try:
            CONDENSE_PROMPT = self.prompt.TALK_CONDENSE_PROMPT
            condense_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        CONDENSE_PROMPT,
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            chain = condense_prompt | self.llm

            with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="messages",
            )

            config = {"configurable": {"session_id": f"{session_id}"}}

            # Respond to user query
            if query:
                input = f"{query}"

                response = with_message_history.invoke(
                    {"messages": [HumanMessage(content=input)]},
                    config=config,
                )

                return response.content.replace('"""', '"')
        except Exception as e:
            print(f"Error: {e}")
            raise
