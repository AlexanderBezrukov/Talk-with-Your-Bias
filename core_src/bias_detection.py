import os
import pandas as pd
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
import io
import sys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

#
class BiasDetection:
    def __init__(self, model_name: str):
        self.generate_code_prompt = """
            # Context:
            You are a highly skilled data analyst and Python programmer. You will receive a DataFrame along with the conversation history, column information, and sample data. Your task is to generate Python code that performs a fairness analysis of the data. The analysis should go beyond basic calculations and provide deep insights through the use of aggregations, bias statistical metrics, or data transformations.
            
            # Instructions:
            1. Review the provided table information and conversation history thoroughly. Ensure that you understand the critical domain of dataset and data context, the types of columns present, and any specific requirements for handling them. Use the correct column names provided in the table description and avoid introducing new columns.
            2. Pay particular attention to data types (e.g., categorical vs. numerical) and ensure these types are preserved throughout the analysis. Avoid unintentional data type conversions by converting the data back to the original type after operations like shifting or aggregating.
            3. Generate Python code that conducts a comprehensive analysis of the DataFrame. The analysis should include but is not limited to:
                - **Descriptive Statistics for different protected attributes:** Provide measures such as mean, median, standard deviation, and range for numerical columns.
                - **Disparate Treatment Metrics:** Provide Disparate Treatment Metrics measures depending on target variable type, for Binary: Group-wise Mean Difference (GMD), Chi-Square Test for Dependence; for Categorical: Conditional Entropy Difference, for regression:Mean Outcome Difference (MOD).
                - **Disparate Impact Analysis:** Measure Disparate Impact Metrics such as Disparate Impact Ratio (DIR), Statistical Parity Difference (SPD) if Target Variable  is Binary; Conditional Statistical Parity (CSP) if Target Variable is Categorical; and Kullback-Leibler (KL) Divergence, Theil Index if if Target Variable is Continious. Summarize the key comparisons and insights.
                - **Counterfactual Fairness Test:** Measure Counterfactual Fairness Test. implementation depends on Target Variable type.
                - **Outliers and Anomalies:** Identify and highlight any outliers or anomalies in the data that deviate significantly from the norm, and explain their potential impact in natural language.
            4. Structure the code so that each analysis block is executed incrementally: perform an operation, then immediately print the results or insights in a clear, natural language format. Avoid printing raw DataFrames or tables directly.
            5. Replace all direct DataFrame print statements with detailed descriptions of the analysis results. Use print statements to explain the significance of the data, trends, and statistics in an informative, narrative style. The print statements should not simply list values (e.g., "Measure1: Value1"). Instead, they should be full sentences that describe the findings (e.g., "The average revenue per customer in XY was XY for product group XY in country XY.").
            6. If necessary, fill null values with a placeholder or use appropriate methods to handle missing data.
            7. Do not use the return statement or generate visualizations. Focus solely on generating fairness analysis code.
            9. The code must load the DataFrame from the specified path, perform the fairness analysis, and use print statements to communicate the insights gained. Do not save any output to a file.
            10. Review the generated code for any syntax or logical errors before finalizing it. Ensure that the code is error-free and adheres to best practices.
            
            # Objective:
            Your primary goal is to generate Python code that provides meaningful and actionable fairness insights based on the data, domain,protected attributes and target variable type. The code should be robust, error-free, and ready for execution without any further modifications. The results should be communicated clearly and effectively through descriptive print statements, avoiding the direct printing of raw tables or DataFrames. The print statements should read as informative, natural language explanations, making the insights easy to understand and interpret.
            
            # Important Note:
            Do not include any additional comments or non-coding text in the final code. The code will be directly executed to generate the analysis results.
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

        self.analyze_results_prompt = """
        # Context:
        You are a highly skilled biased-data analyst. You will be provided with the results from a Bias detection analysis of a DataFrame, including column information, sample data, conversation history, schema of the full table. Your task is to interpret these bias metric results and identify key insights and trends.

        # Instructions:
        1. Carefully review the provided bias analysis results, column information, or schema description. Ensure that you understand the critical domain specs, context of the data, protected attributes and the implications of the analysis performed.
        2. Provide a detailed and concise analysis of the results, highlighting any significant trends, patterns, or anomalies that you observe.
        3. Structure your response strictly according to the following format and do not include any additional information:

        **Key Findings**:
        - Bullet points summarizing the most significant fairness insights. Each bullet should be clear and concise, focusing on the most important aspects of the analysis.
        - Do not use any special formatting effects like bold, italics, underlines, or special characters. Provide the analysis in plain text only.
    
        # Objective:
        Your goal is to provide a clear and actionable summary of the analysis results, offering fairness insights that can guide further decision-making or investigation. Do not include any extraneous information or unnecessary details.
        """
        self.llm1 = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_name, temperature=0.00001)
        self.llm2 = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_name, temperature=0.05)


    async def detect_bias(self, session_id: str, table_df_path: str, conversation_history: str) -> str:
        # Step 1: Generate Python code for analysis
        table_df = pd.read_csv(table_df_path)
        code_generation_message = self._prepare_code_generation_message(table_df_path, table_df, conversation_history)
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

            chain = condense_prompt | self.llm1

            with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="messages",
            )

            config = {"configurable": {"session_id": session_id}}

            # Respond to user query
            if conversation_history:
                response = with_message_history.invoke(
                    {"messages": [HumanMessage(content=code_generation_message)]},
                    config=config,
                )
                generated_code = response.content

        except Exception as e:
            return {"error": str(e) + code_generation_message}

        # Step 2: Execute the generated code
        analysis_results = self._execute_generated_code(generated_code)
        #print(f"Analysis results: {analysis_results}")

        if "Error executing code" in analysis_results or "An unexpected error occurred:" in analysis_results:
            # Regenerate the code by providing the error context along with the relevant table info and conversation history
            print(f"Error encountered: {analysis_results}")
            try:
                # Request new code generation with error context using the specific correction prompt
                response = with_message_history.invoke(
                    {"messages": [HumanMessage(content=f"There is an error in code: {analysis_results}. Fix it.")]},
                    config=config,
                )
                generated_code = response.content
                # print(f"Regenerated code response: {generated_code}")
            except Exception as e:
                return {"error": f"Error during code regeneration: {str(e)}"}

            # Attempt to execute the regenerated code
            analysis_results = self._execute_generated_code(generated_code)
            # If there's still an error, return without invoking the analysis model
            if "Error executing code" in analysis_results or "An unexpected error occurred:" in analysis_results:
                return "Error: Unable to generate analysis due to code errors."
        # Step 3: Analyze the results using the LLM
        analysis_message = self._prepare_analysis_message(analysis_results, table_df, conversation_history)
        try:
            final_analysis_response = self.llm2.invoke([
                ("system", self.analyze_results_prompt),
                ("user", analysis_message)
            ])
            final_analysis = final_analysis_response.content
            return (final_analysis, analysis_results, generated_code)
        except Exception as e:
            return {"error": str(e)}

    def _prepare_code_generation_message(self, table_df_path: str, table_df: pd.DataFrame, conversation_history: str) -> str:
        table_description = self._create_table_description(table_df)
        return f"Table Path: {table_df_path}\n\nTable Description:\n{table_description}\n\nConversation History:\n{conversation_history}"

    def _prepare_analysis_message(self, analysis_results: str, table_df: pd.DataFrame, conversation_history: str) -> str:
        table_description = self._create_table_description(table_df)
        return f"Analysis Results:\n{analysis_results}\n\nTable Description:\n{table_description}\n\nConversation History:\n{conversation_history}"

    def _create_table_description(self, table_df: pd.DataFrame) -> str:
        description = "\nColumns and Data Types:\n"
        for column in table_df.columns:
            description += f"- {column} (Type: {table_df[column].dtype})\n"

        description += "\nSample Values for Each Column:\n"
        for column in table_df.columns:
            # Drop null values and check if there are any non-null values left
            non_null_values = table_df[column].dropna()
            if len(non_null_values) > 0:
                sample_values = non_null_values.sample(min(3, len(non_null_values))).tolist()
                description += f"- {column}: {sample_values}\n"
            else:
                description += f"- {column}: No non-null values available\n"

        return description

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
            # Get the entire content of the buffer
            output = buffer.getvalue()
            return output.strip()  # Remove any extra leading/trailing whitespace
        except Exception as e:
            return f"Error executing code: {e}"
        finally:
            # Restore the original stdout
            sys.stdout = old_stdout
            buffer.close()
