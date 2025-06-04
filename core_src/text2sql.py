import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .prompts import PromptTemplates
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import os

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def get_json(output_: str) -> dict:
    try:
        output = output_.replace("```", "").replace("```json","").replace("json","").replace("sql", "").replace("\\\n","\n").replace("\\n","\n")
        json_output = json.loads(output, strict=False)
        json_output["LLM_err"] = ""
        return json_output
    except ValueError as _:
        try:
            output = output.split(",")
            print(output, len(output), _)
            if len(output) == 4:
                quality_value = "0"
                json_output = {
                "quality_value":quality_value,
                "confidence_value":output[0].split(":")[1].strip().replace('"', '').replace("'", """"""),
                "clarifying_questions":output[1].split(":")[1].strip().replace('"', '').replace("null", ""),
                "First draft":output[2].split(":")[1].replace('"', '').replace('}', '').strip(),
                "Final answer":output[3].split(":")[1].replace('"', '').replace('}', '').strip(),
                "LLM_err":""
                }
            elif len(output) == 5:

                json_output = {
                    "quality_value":output[0].split(":")[1].strip().replace('"', '').replace("'", """"""),
                    "confidence_value":output[1].split(":")[1].strip().replace('"', '').replace("'", """"""),
                    "clarifying_questions":output[2].split(":")[1].strip().replace('"', '').replace("null", ""),
                    "First draft":output[3].split(":")[1].replace('"', '').replace('}', '').strip(),
                    "Final answer":output[4].split(":")[1].replace('"', '').replace('}', '').strip(),
                    "LLM_err":""
                }
            else:
                json_output = {
                    "quality_value":"0",
                    "confidence_value":"100",
                    "clarifying_questions":"",
                    "First draft":"",
                    "Final answer":"",
                    "LLM_err": output_
                }

            return json_output
        except ValueError as e:
            print(f"Error in JSON parsing: {e}")
            raise

def parse_final_answer(output: str) -> str:
    return (
        output.strip().replace("```", "").replace("sql", "").replace("SQL", "")
    )

class SQL_Generator:
    def __init__(self, model_name:str, prompt_template:type(PromptTemplates)):
        # self.session_id = session_id
        self.prompt = prompt_template
        self.llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name, temperature=0.0001)
    async def predict(self, query, session_id="clarificationclient31") -> dict:

        try:
            CONDENSE_PROMPT = self.prompt.CONDENSE_PROMPT
            quality_control_prompt = self.prompt.quality_control_prompt
            clarifying_prompt = self.prompt.clarifying_prompt
            todo_prompt = self.prompt.todo_prompt
            validation_prompt = self.prompt.validation_prompt
            format_prompt = self.prompt.format_prompt

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

            config = {"configurable": {"session_id": f"{session_id}id6"}}

            # Respond to user query
            if query:
                sql_query = ""
                clarifying_questions = ""
                confidence_value = 0
                quality_value = 0
                input = f"{query} {clarifying_prompt} {todo_prompt} {validation_prompt} {quality_control_prompt} {format_prompt}"

                response = with_message_history.invoke(
                    {"messages": [HumanMessage(content=input)]},
                    config=config,
                )

                output_dict = get_json(response.content.replace('"""', '"'))
                LLM_err = output_dict["LLM_err"]
                if LLM_err != "":
                    input = f"Double check the final sql query. Strictly {format_prompt}"
                    response = with_message_history.invoke(
                        {"messages": [HumanMessage(content=input)]},
                        config=config,
                    )

                    output_dict = get_json(response.content.replace('"""', '"'))
                    LLM_err = output_dict["LLM_err"]

                if LLM_err == "":
                    quality_value = int(output_dict["quality_value"])
                    validation_passed = True
                    if "confidence_value" in output_dict:
                        confidence_value = int(output_dict["confidence_value"])
                        if confidence_value > 50:
                            sql_query = parse_final_answer(output_dict["Final answer"])

                        else:
                            clarifying_questions = output_dict["clarifying_questions"]

                    else:
                        validation_passed = False
                        errors = '"confidence_value" is missing in output'

                    if validation_passed is False:
                        sql_query = ""
                        correction_prompt = f"""
                        Double check the correctness of the format of your answer and the Final answer, fix ALL errors and errors including:
                         - {errors} """
                        response = with_message_history.invoke(
                            {"messages": [HumanMessage(content=correction_prompt)]},
                            config=config,
                        )


                        output_dict = get_json(response.content.replace('"""', '"'))
                        confidence_value = int(output_dict["confidence_value"])
                        if confidence_value > 50:
                            sql_query = parse_final_answer(output_dict["Final answer"])
                        else:
                            clarifying_questions = output_dict["clarifying_questions"]


                return {
                    "quality_value": quality_value,
                    "confidence_value": confidence_value,
                    "clarifying_questions": clarifying_questions,
                    "sql_query": sql_query,
                    "LLM_err": LLM_err
                }
        except Exception as e:
            print(f"Error: {e}")
            raise
