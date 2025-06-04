import os
import json
from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_json(output: str) -> dict:
    output = output.replace("```", "").replace("```json","").replace("json","").replace("sql", "").replace("\\\n","\n").replace("\\n","\n")
    return json.loads(output)


class Visualisation:
    def __init__(self, model_name: str):
        self.llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name, temperature=0.0001)
        self.CONDENSE_PROMPT = "You are high qualified Data Visualizer."
        self.todo_prompt = """
        Visualize bias detection metrics and statistical disparities from the report and select the best type of plot (indicator, bar, scatter or pie) from plotly.graph_objects to make reasonable visualisation of the metrics to show fairness across protected attributes. Add colors to improve user experience.
        if user asks to visualize dataset, you Analyze all columns and info about dataframe from the prompt and select the best type of plot (bar, scatter or pie) from plotly.graph_objects to make reasonable visualisation of dataframe df answering User question. Remember name of dataframe is df.
        If there is a column with week, yearweek, datetime - use it as x axis to plot time series [example:g o.Bar(x=df['week'], y = ...)] 
        Use fig = go.Figure(). Respond only following the format, not extra text, double check the columns names:
        {
            "Text_for_user":"<<suggested visualisation>>",
            "add_traces":"[fig.add_trace(<<code to add>>), fig.add_trace(<<code to add>>),..]",
            "fig.update_layout":"fig.update_layout(<<code to add>>)"
        }
        Examples:
        {
            "Text_for_user": "Indicators are used to display key bias detection metrics for protected attributes.",
            "add_traces": "[fig.add_trace(go.Indicator(mode='number', value=0.15, title={'text': 'Disparate Impact (Gender)'}, domain={'row': 0, 'column': 0})), fig.add_trace(go.Indicator(mode='number', value=0.25, title={'text': 'Disparate Impact (Race)'}, domain={'row': 0, 'column': 1})), fig.add_trace(go.Indicator(mode='number', value=0.10, title={'text': 'Disparate Impact (Age)'}, domain={'row': 1, 'column': 0})), fig.add_trace(go.Indicator(mode='number', value=0.12, title={'text': 'Statistical Parity (Gender)'}, domain={'row': 1, 'column': 1}))]",
            "fig.update_layout": "fig.update_layout(title='Bias Detection Metrics Indicators', grid={'rows': 2, 'columns': 2, 'pattern': 'independent'})"
        }
        {
            "Text_for_user": "A bar chart is used to compare bias detection metrics across protected attributes.",
            "add_traces": "[fig.add_trace(go.Bar(x=['Gender', 'Race', 'Age'], y=[0.15, 0.25, 0.10], name='Disparate Impact')), fig.add_trace(go.Bar(x=['Gender', 'Race', 'Age'], y=[0.12, 0.20, 0.08], name='Statistical Parity'))]",
            "fig.update_layout": "fig.update_layout(title='Bias Detection Metrics Comparison', xaxis_title='Protected Attributes', yaxis_title='Metric Value', barmode='group')"
        }
        {
          "Text_for_user": "The suggested visualization is a histogram showing the percentage of individuals within each gender group who have a positive class label (income >50K).",
          "add_traces": "[fig.add_trace(go.Bar(x=df[df['Class-label'] == 1]['gender'].value_counts().index, y=df[df['Class-label'] == 1]['gender'].value_counts().values/df['gender'].value_counts().values, name='>50K Income'))]",
          "fig.update_layout": "fig.update_layout(title='Number of Individuals with >50K Income by Gender', xaxis_title='Gender', yaxis_title='Count', xaxis_type='category', barmode='group')"
        }
        

  
        """
        self.store = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    async def predict(self, query: str, session_id="visualisation") -> dict:

        try:
            condense_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.CONDENSE_PROMPT,
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            chain = condense_prompt | self.llm

            with_message_history = RunnableWithMessageHistory(
                chain,
                self.get_session_history,
                input_messages_key="messages",)

            config = {"configurable": {"session_id": session_id}}

            # Respond to user query
            if query:
                input = f"{query}\n {self.todo_prompt}"
                response = with_message_history.invoke(
                    {"messages": [HumanMessage(content=input)]},
                    config=config,
                )
                output_dict = get_json(response.content)

                return output_dict

        except Exception as e:
            print(f"Error occurred: {e}")
            raise
