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
import io
import sys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
store = {}



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]



class Data_Generator:
    def __init__(self, model_name:str):
        self.llm1 = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_name, temperature=0.00001)
        self.generate_code_prompt = """
            # Context:
            You are a highly skilled data analyst and Python programmer. You will receive a DataFrame description along with the conversation history, column information, and sample data. Your task is to generate Python code that performs a fairness analysis of the data. The analysis should go beyond basic calculations and provide deep insights through the use of aggregations, bias statistical metrics, or data transformations.
            
            # Instructions:
            1. Review the provided table information and conversation history thoroughly. Ensure that you understand the critical domain of dataset and data context, the types of columns present, and any specific requirements for handling them. Use the correct column names provided in the table description and avoid introducing new columns.
            2. Pay particular attention to data types (e.g., categorical vs. numerical) and ensure these types are preserved throughout the generation. Avoid unintentional data type conversions by converting the data back to the original type after operations like shifting or aggregating.
            3. Generate Python code that generate realistic DataFrame from the critical domain. The generated dataset should include but is not limited to:
                 - Make sure that the numbers make sense, respect common logic connections between attributes.
                 - Make sure that data has good diversity and correspond real-world distributions.
            4. Structure the code so that each block is executed incrementally: perform an operation, then immediately print raw generated DataFrame.
            5. Do not use the return statement or generate visualizations. Focus solely on generating dataset.
            6. Review the generated code for any syntax or logical errors before finalizing it. Ensure that the code is error-free and adheres to best practices.
            7. Call dataset df
            
            # Objective:
            Your primary goal is to generate Python code that realistic dataset based on the critical domain, protected attributes, columns and target variable type. The code should be robust, error-free, and ready for execution without any further modifications. 
            # Important Note:
            Do not include any additional comments or non-coding text in the final code. The code will be directly executed to generate the dataset.
            """
        self.correct_code_prompt = """
        # Context:
        You are a highly skilled Python programmer and data analyst. You will receive a Python code snippet that resulted in an error when executed. Your task is to correct the code by identifying the issue and generating a revised version that works correctly.

        # Instructions:
        1. Carefully review the provided code, error message, and relevant table information to understand the context and the cause of the error.
        2. Correct the code by addressing the specific issue, ensuring that the solution is robust and maintains the overall logic of the original code.
        3. Structure the corrected code to execute incrementally: perform each key operation, then immediately print the results or validate the outcome. This ensures that partial results are available and errors are easier to trace.
        4. Make sure to review the corrected code for any syntax or logical errors. The code should handle potential issues gracefully, allowing the analysis to proceed as far as possible even if an error occurs later.
        5. The final code should be ready for execution without any further modifications. Avoid adding any additional comments or non-coding text.

        # Objective:
        Your goal is to generate a corrected version of the Python code that performs the required analysis without errors, using an incremental approach to ensure robustness and clarity. No extra comments or non-coding text should be included in the final code.
        """

    async def predict(self, query, session_id="data_gen"):

        try:
            condense_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.generate_code_prompt,
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            code_generation_message = query

            chain = condense_prompt | self.llm1

            with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="messages",
            )

            config = {"configurable": {"session_id": session_id}}

            # Respond to user query
            if code_generation_message:
                response = with_message_history.invoke(
                    {"messages": [HumanMessage(content=code_generation_message)]},
                    config=config,
                )
                #print(f"Generated code response: {response}")
                generated_code = response.content


        except Exception as e:
            return {"error": str(e) + code_generation_message}

        results = self._execute_generated_code(generated_code)
        if "Error executing code" in results or "An unexpected error occurred:" in results:
            # Regenerate the code by providing the error context along with the relevant table info and conversation history
            #error_context_message = f"The following code resulted in an error:\n```python\n{generated_code}\n```\nThe error was: {analysis_results}\n\nHere is the relevant table information and conversation history:\n\n{self._prepare_code_generation_message(table_df_path, table_df, conversation_history)}\n\nPlease correct the code and try again."
            print(f"Error encountered: {results}")
            try:
                # Request new code generation with error context using the specific correction prompt
                response = with_message_history.invoke(
                    {"messages": [HumanMessage(content=f"There is an error in code: {results}. Fix it.")]},
                    config=config,
                )

                generated_code = response.content
                # print(f"Regenerated code response: {generated_code}")
            except Exception as e:
                return {"error": f"Error during code regeneration: {str(e)}"}

            # Attempt to execute the regenerated code
            results = self._execute_generated_code(generated_code)
            # print(f"Regenerated analysis results: {analysis_results}")

            # If there's still an error, return without invoking the analysis model
            if "Error executing code" in analysis_results or "An unexpected error occurred:" in analysis_results:
                return "Error: Unable to generate analysis due to code errors."


        return results, generated_code

    def _execute_generated_code(self, code: str) -> str:
        # Create a string buffer to capture the output
        code = code.strip().strip("```").replace("python", "").strip()

        buffer = io.StringIO()
        # Save the current stdout
        old_stdout = sys.stdout
        try:
            # Redirect stdout to the buffer
            sys.stdout = buffer
            local_vars = {}
            exec(code, globals(), local_vars)

            return local_vars.get("df")   # Remove any extra leading/trailing whitespace
        except Exception as e:
            return f"Error executing code: {e}"
        finally:
            # Restore the original stdout
            sys.stdout = old_stdout
            buffer.close()

