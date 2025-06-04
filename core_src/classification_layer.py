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

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def get_json(output_: str) -> dict:
    try:
        output = output_.replace("```", "").replace("```json","").replace("json","").replace("\\\n","\n").replace("\\n","\n")
        json_output = json.loads(output, strict=False)
        json_output["LLM_err"] = ""
        return json_output
    except ValueError as _:
        try:
            output = output.split(",")
            print(output, len(output), _)
            json_output = {
                "warning": "",
                "module_name": "",
                "LLM_err": output_
            }
            return json_output
        except ValueError as e:
            print(f"Error in JSON parsing: {e}")
            raise

def validation(model_output_dict):
    for i in "text2sql,datagen,biasdet,visual,biasinj,None".split(","):
        if i in model_output_dict["module_name"]:
            return True
    return False

def parse_final_answer(output: str) -> str:
    return (
        output.strip().replace("```", "")
    )

class CLASS_Generator:
    def __init__(self, model_name:str, prompt_template:type(PromptTemplates)):
        self.prompt = prompt_template
        self.llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name, temperature=0.0000001)
    async def predict(self, query, session_id="classification_layer") -> dict:

        try:
            CONDENSE_PROMPT = self.prompt.CLASSIFICATION_CONDENSE_PROMPT

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
                input = f"User prompt: {query}." + """Return in format:{"warning": <warnings>,"module_name": <one of text2sql,datagen,biasdet,visual,biasinj,None>}"""
                response = with_message_history.invoke(
                    {"messages": [HumanMessage(content=input)]},
                    config=config,
                )

                output_dict = get_json(response.content.replace('"""', '"'))
                LLM_err = output_dict["LLM_err"]
                if LLM_err != "":
                    input = f"Double check the final answer. Strictly follow format."
                    response = with_message_history.invoke(
                        {"messages": [HumanMessage(content=input)]},
                        config=config,
                    )

                    output_dict = get_json(response.content.replace('"""', '"'))
                    LLM_err = output_dict["LLM_err"]

                if LLM_err == "":
                    validation_passed = validation(output_dict)
                    if validation_passed is False:
                        errors = "module_name should be one of text2sql,datagen,biasdet,visual,biasinj,None"
                        correction_prompt = f"""
                        Double check the correctness of the format of your answer, fix errors:
                         - {errors} """
                        response = with_message_history.invoke(
                            {"messages": [HumanMessage(content=correction_prompt)]},
                            config=config,
                        )

                        output_dict = get_json(response.content.replace('"""', '"'))

                return output_dict
        except Exception as e:
            print(f"Error: {e}")
            raise
