import streamlit as st
from datetime import datetime as dt
import pandas as pd
import asyncio
from dotenv import load_dotenv
import argparse
import subprocess
import json
import runpy
import sys
from io import StringIO
load_dotenv("credentials.env")
from core_src.classification_layer import CLASS_Generator
from core_src.chat import Chat_Generator
from core_src.text2sql import SQL_Generator
from core_src.visualisation_metrics import Visualisation
from core_src.dataset_gen import Data_Generator
from core_src.bias_detection import BiasDetection
from core_src.bias_injection import BiasInject
from PIL import Image

from sql_formatter.core import format_sql
from core_src import prompts
from langdetect import detect
from core_src.utils import *
import os
import sqlite3

DATASET_PATHS = {
    "COMPAS": "sota_tabular_datasets/law_and_criminal/compas-scores-two-years_clean.csv",
    "ProPublica Violent Recidivism": "sota_tabular_datasets/law_and_criminal/compas-scores-two-years-violent_clean.csv",
    "Communities and Crime": "sota_tabular_datasets/law_and_criminal/communities_crime.csv",
    "Diabetes": "sota_tabular_datasets/healthcare/diabetes-clean.csv",
    "Adult Census Income": "sota_tabular_datasets/finance/adult-clean.csv",
    "Dutch Census": "sota_tabular_datasets/employment_and_worker_management/dutch.csv",
    "Ricci v. DeStefano": "sota_tabular_datasets/employment_and_worker_management/ricci_race.csv",
    "Law School Admission Council (LSAC)": "sota_tabular_datasets/education/law_school_clean.csv",
    "Student—Mathematics": "sota_tabular_datasets/education/student_mat_new.csv",
    "Student—Portuguese": "sota_tabular_datasets/education/student_por_new.csv",
    "Adult German Credit": "sota_tabular_datasets/finance/german_data_credit.csv",
    "Bank Marketing": "sota_tabular_datasets/finance/bank-full.csv",
    "Credit Card Clients": "sota_tabular_datasets/finance/credit-card-clients.csv",
    "Boston Housing": "sota_tabular_datasets/housing_and_socioeconomics/housing.csv"
}
# Define a dictionary of critical domains, datasets, and protected attributes
CRITICAL_DOMAINS = {
    "Law Enforcement & Criminal Justice": {
        "datasets": {
            "COMPAS": ["race", "sex"],
            "ProPublica Violent Recidivism": ["race", "sex"],
            "Communities and Crime": ["Black"]
        }
    },
    "Healthcare": {
        "datasets": {
            "Diabetes": ["Gender"]
        }
    },
    "Employment & Worker Management": {
        "datasets": {
            "Adult Census Income": ["gender", "age", "race"],
            "Dutch Census": ["sex"],
            "Ricci v. DeStefano": ["Race"]
        }
    },
    "Education": {
        "datasets": {
            "Law School Admission Council (LSAC)": ["race", "male"],
            "Student—Mathematics": ["age", "sex"],
            "Student—Portuguese": ["age", "sex"]
        }
    },
    "Financial Services": {
        "datasets": {
            "Adult German Credit": ["age", "foreign_worker"],
            "Bank Marketing": ["age", "marital"],
            "Credit Card Clients": ["SEX", "EDUCATION", "MARRIAGE"]
        }
    },
    "Housing": {
        "datasets": {
            "Boston Housing": ["Race"]
        }
    }
}

# Define file paths for CSVs
bias_metrics_file = "knowledge_db_files/bias_metrics.csv"
bias_tools_file = "knowledge_db_files/bias_detection_tools.csv"
datasets_file = "knowledge_db_files/datasets.csv"
critical_domains_file = "knowledge_db_files/critical_domains.csv"
laws_file = "knowledge_db_files/laws.csv"
protected_attributes_file = "knowledge_db_files/protected_attributes.csv"
protected_attributes_laws_file = "knowledge_db_files/protected_attributes_laws.csv"

# Read CSV files into DataFrames
df_bias_metrics = pd.read_csv(bias_metrics_file)
df_bias_tools = pd.read_csv(bias_tools_file)
df_datasets = pd.read_csv(datasets_file)
df_critical_domains = pd.read_csv(critical_domains_file)
df_laws = pd.read_csv(laws_file)
df_protected_attributes = pd.read_csv(protected_attributes_file)
df_protected_attributes_laws = pd.read_csv(protected_attributes_laws_file)

# Create SQLite connection
conn = sqlite3.connect(":memory:")  # In-memory database

# Load DataFrames into SQLite tables
df_bias_metrics.to_sql("bias_metrics_data", conn, index=False, if_exists='replace')
df_bias_tools.to_sql("bias_detection_tools", conn, index=False, if_exists='replace')
df_datasets.to_sql("datasets_data", conn, index=False, if_exists='replace')
df_critical_domains.to_sql("critical_domains_data", conn, index=False, if_exists='replace')
df_laws.to_sql("laws_data", conn, index=False, if_exists='replace')
df_protected_attributes.to_sql("protected_attributes_data", conn, index=False, if_exists='replace')
df_protected_attributes_laws.to_sql("protected_attributes_laws_data", conn, index=False, if_exists='replace')

def query_database(sql_query):
    return pd.read_sql_query(sql_query, conn)

def initialize_variables():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()
    if "visualisation_code" not in st.session_state:
        st.session_state.visualisation_code = []
    if "sql_to_explain" not in st.session_state:
        st.session_state.sql_to_explain = {}
    if "last_dataset_name" not in st.session_state:
        st.session_state.last_dataset_name = ""
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = True
    if "protected_attr" not in st.session_state:
        st.session_state.protected_attr = ""
    if "filename" not in st.session_state:
        st.session_state.filename = ""
    if "clicked_analysis" not in st.session_state:
        st.session_state.clicked_analysis = False
    if "bias_detection_context" not in st.session_state:
        st.session_state.bias_detection_context = ""
    if "bias_detection_metric_values" not in st.session_state:
        st.session_state.bias_detection_metric_values = ""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "prompts_template" not in st.session_state:
        st.session_state.prompts_template = prompts.PromptTemplates(
            rules_file="agent_specs/rules.txt",
            schema_file="agent_specs/database_schema_description.txt",
            examples_file="agent_specs/examples.txt",
            examples_clarification_file="agent_specs/examples_clarifications.txt"
        )
    if "model_name" not in st.session_state:
        st.session_state.model_name = None

    st.session_state.model_name = "gpt-4o-mini"

# Async function to interact with data model
async def llm_query_generation(message, session_id):
    try:
        result = await st.session_state.sql_generation.predict(
            query=message, session_id=session_id
        )
        return result
    except Exception as e:
        st.error(f"Error in LLM function: {e}")
        return None

async def llm_chat_generation(message, session_id):
    try:
        result = await st.session_state.chat.predict(
            query=message, session_id=session_id
        )
        return result
    except Exception as e:
        st.error(f"Error in LLM function: {e}")
        return None

async def llm_class_generation(message, session_id):
    try:
        result = await st.session_state.class_.predict(
            query=message, session_id=session_id
        )
        return result
    except Exception as e:
        st.error(f"Error in LLM function: {e}")
        return None

async def llm_visualisation_generation(message, session_id):
    try:
        result = await st.session_state.visualisation.predict(query=message, session_id=session_id)
        # st.write(result)
        return result
    except Exception as e:
        st.error(f"Error in visualisation function: {e}")
        return None

async def llm_data_generation(query,session_id):
    try:
        result = await st.session_state.data_generator.predict(
            query=query, session_id=session_id
        )
        return result
    except Exception as e:
        st.error(f"Error in data gen function: {e}")
        return None

async def llm_bias_analysis(session_id, table_df_path, conversation_history):
    try:
        result = await st.session_state.bias_analyzer.detect_bias(
            session_id=session_id, table_df_path=table_df_path, conversation_history=conversation_history
        )
        return result
    except Exception as e:
        st.error(f"Error in table_analysis function: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error in Table analysis function: {e}"})
        return None

async def llm_bias_injection(session_id, table_df_path, conversation_history):
    try:
        result = await st.session_state.bias_inject.inject_bias(
            session_id=session_id, table_df_path=table_df_path, conversation_history=conversation_history
        )
        return result
    except Exception as e:
        st.error(f"Error in table_analysis function: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error in injection function: {e}"})
        return None


def display_chat_messages(logo_picture):
    # Display chat messages from history on app rerun
    index_plot = 0
    for message in st.session_state.messages:
        role = message["role"]
        avatar = None
        if role == "assistant":
            avatar = logo_picture
        with st.chat_message(message["role"], avatar=avatar):
            if message["content"] is None:
                pass
            elif message["content"].startswith("WARN"):
                st.warning(message["content"][4:])
            elif message["content"].startswith("py:"):
                expander = st.expander("Generated Python Code: ", expanded=False)
                expander.code(message["content"][3:], language="py")
            elif message["content"] == "_visualization_plot_here_":
                print(st.session_state.visualisation_code[index_plot])
                exec(st.session_state.visualisation_code[index_plot])
                index_plot += 1
            elif is_sql(message["content"]):
                expander = st.expander("Generated SQL Query: ", expanded=False)
                sql_query = message["content"].replace("sql_query:", "")
                expander.code(sql_query, language="sql")
            elif is_figure(message["content"]):
                image = Image.open(message["content"]).convert("RGB")
                st.image(
                    image, caption=os.path.basename("Visualization"), channels="RGB"
                )
            elif is_csv(message["content"]):

                if os.path.exists(message["content"]):
                    df = pd.read_csv(message["content"])
                    st.dataframe(df)

            else:
                st.markdown(message["content"])


    st.session_state.chat = Chat_Generator(st.session_state.model_name, st.session_state.prompts_template)
    st.session_state.class_ = CLASS_Generator(st.session_state.model_name, st.session_state.prompts_template)
    st.session_state.sql_generation = SQL_Generator(
        st.session_state.model_name, st.session_state.prompts_template
    )
    st.session_state.visualisation = Visualisation(st.session_state.model_name)
    st.session_state.bias_analyzer = BiasDetection(st.session_state.model_name)
    st.session_state.data_generator = Data_Generator(st.session_state.model_name)
    st.session_state.bias_inject = BiasInject(st.session_state.model_name)

# Streamlit app
def main():

    initialize_variables()
    logo_picture = "./logo/bot_logo.png"
    # Set page configuration
    st.set_page_config(
        page_title="Talk with Your Bias",
        page_icon="⚖️",
        layout="wide"
    )

    # Layout for title and description
    col1, col2 = st.columns([2, 4])
    with col2:
        st.title("Welcome to Talk with Your Bias!")
        st.text(" Context-Aware Agent for Bias Detection and Generation in Tabular Data")
    with col1:
        st.image("logo/picture.png", width=200)

    st.markdown("""
        **Talk with Your Bias** is a framework designed to analyze, measure and generate bias in tabular datasets.
        It allows users to upload their own datasets or generate synthetic data with controlled bias for evaluation and stress-testing.
        The framework provides interactive explanations about bias types, protected attributes, etc.
    """)

    st.header("Why Talk with Your Bias?")
    st.markdown("""
        Our framework provides:
        - **Bias detection** using fairness metrics and custom measures.
        - **Interactive explanations** to help users understand bias impact.
        - **Data generation with controlled bias level** for testing and model fairness evaluation.
    """)

    # Sidebar instructions
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
        # Talk with Your Bias: Workflow Overview

The **Talk with Your Bias** framework enables discussions about different bias concerns even without selecting a dataset. However, to analyze, detect, or inject bias into a dataset within a critical domain, you must follow a structured workflow.

## General Workflow

1. **Select a Critical Domain**  
   - Before performing any bias-related tasks, you must define the domain in which bias will be analyzed or injected.  
   - This ensures that discussions and insights remain relevant to the specific field of interest.  

2. **Prepare Your Dataset**  
   Choose one of the following options:  
   - **Upload Your Dataset**: Use your own data for bias analysis.  
   - **Upload an Existing Biased Dataset**: Work with a dataset known to contain biases.  
   - **Generate a Custom Dataset**: Create a dataset tailored to your specific needs and bias concerns.  

3. **Start Talking About Your Bias**  
   - Once the first step is completed, you can begin analyzing bias in the dataset.  
   - The framework allows for interactive discussions and insights into bias-related issues.  

4. **Bias Analysis & Injection**  
   - **Bias Analysis**: Identify and quantify bias in your dataset.  
   - **Bias Injection**: Introduce specific biases intentionally for testing or simulation purposes.  
   - These actions can only be performed after step 1 (selecting a critical domain).  

5. **Visualization of Bias**  
   - Visualization tools help interpret bias in the dataset.  
   - Visual representation only makes sense after performing bias analysis.  

## Key Notes:
- The workflow is **not strictly linear**, meaning users can explore bias concerns flexibly.  
- However, **bias analysis and bias injection require a selected critical domain first**.  
- **Visualization should follow bias analysis** for meaningful insights.  

This structured yet flexible approach ensures a comprehensive examination of bias while allowing for interactive discussions and deeper analysis.
    """)

    # Sidebar: Step 1 - Select a Critical Domain
    st.sidebar.header("Step 1: Select a Critical Domain")
    critical_domain = st.sidebar.selectbox(
        "Choose a domain to analyze bias:",
        list(CRITICAL_DOMAINS.keys())
    )

    st.sidebar.markdown("---")

    # Sidebar: Step 2 - Prepare Your Dataset
    st.sidebar.header("Step 2: Prepare Your Dataset")
    dataset_option = st.sidebar.radio(
        "Choose a dataset option:",
        [
            "Upload Your Dataset",
            f"Upload an Existing Biased Dataset from the {critical_domain}",
            "Generate a Custom Dataset"
        ]
    )
    df = None  # Initialize dataset variable
    # Option 1: Upload Your Own Dataset
    if dataset_option == "Upload Your Dataset":
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"**Uploaded Dataset Preview ({critical_domain}):**")
            st.dataframe(df)

    # Option 2: Upload an Existing Biased Dataset
    elif dataset_option == f"Upload an Existing Biased Dataset from the {critical_domain}":
        biased_dataset_name = st.sidebar.selectbox(
            "Choose a biased dataset:",
            list(CRITICAL_DOMAINS[critical_domain]["datasets"].keys())
        )
        df = pd.read_csv(DATASET_PATHS[biased_dataset_name])

        st.write(f"**Selected Biased Dataset: {biased_dataset_name} ({critical_domain})**")
        st.dataframe(df)
        st.write(f"**Protected attributes**: {CRITICAL_DOMAINS[critical_domain]['datasets'][biased_dataset_name]}")
        st.session_state.protected_attr = f"Protected attributes: {CRITICAL_DOMAINS[critical_domain]['datasets'][biased_dataset_name]}"
    # Option 3: Generate a Custom Dataset
    elif dataset_option == "Generate a Custom Dataset":
        n_samples = st.sidebar.slider("Number of Samples", 10, 10000, 100)
        st.sidebar.write(f"Write in chat: **Generate Custom Dataset in {critical_domain} with {n_samples} raw.** You can add description, attribute names, protected attributes, target variable or ML task.")

    st.sidebar.markdown("---")
    st.sidebar.write("💡 After dataset selection, proceed with bias analysis.")

    st.markdown("---")
    st.markdown("🔍 **Explore bias in datasets and improve AI fairness with Talk with Your Bias!**")
    display_chat_messages(logo_picture)
    if type(df)!=type(None):
        st.session_state.dataset = df
        st.session_state.dataset.to_csv("csv_files/original.csv", index=False)

    def inject_bias(filename, conversation_history):
        try:
            st.write("Bias Injection started .... wait, please.")
            st.session_state.filename_analysis = filename
            # if not "Protected attributes:" in conversation_history:

            dataset, code = asyncio.run(llm_bias_injection(session_id=st.session_state.session_id, table_df_path=filename, conversation_history=conversation_history))
            expander = st.expander("Generated Python Code: ", expanded=True)
            expander.code(code, language="py")
            st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"py:{code}",
                }
            )
            #st.markdown(metric_values)
            content = st.write_stream(response_generator("Bias Injection Result:"))
            st.session_state.messages.append({"role": "assistant", "content": content})
            st.dataframe(dataset)
            name = f"csv_files/injected{generate_session_id(7)}.csv"
            dataset.to_csv(name, index=False)
            st.session_state.last_dataset_name = name
            st.session_state.messages.append({"role": "assistant", "content": name})

        except Exception as e:
            st.error(f"Error while generating the analysis: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error while bias injection: {e}"})


    def detect_bias(filename, conversation_history):
        try:
            st.write("Bias Analysis started .... wait, please.")
            st.session_state.filename_analysis = filename
            # if not "Protected attributes:" in conversation_history:
            conversation_history += "detect protected attributes"

            analysis_result, metric_values, code = asyncio.run(llm_bias_analysis(session_id=st.session_state.session_id, table_df_path=st.session_state.filename_analysis, conversation_history=conversation_history))
            st.session_state.cutoff_analysis = None
            if analysis_result is not None:
                expander = st.expander("Generated Python Code: ", expanded=True)
                expander.code(code, language="py")
                st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"py:{code}",
                    }
                )
                #st.markdown(metric_values)
                content = st.write_stream(response_generator("Bias Analysis Result:"))
                st.session_state.messages.append({"role": "assistant", "content": content})
                st.markdown(analysis_result)
                st.session_state.bias_detection_context = analysis_result
                st.session_state.bias_detection_metric_values = metric_values
                st.session_state.messages.append({"role": "assistant", "content": analysis_result})

            else:
                st.warning("No bias analysis result.")
                st.session_state.messages.append({"role": "assistant", "content": "No bias analysis result."})


        except Exception as e:
            st.error(f"Error while generating the analysis: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error while generating the bias analysis: {e}"})

        finally:
            st.session_state.clicked_analysis = False


    def visualization(prompt):
        print("Starting visualisation... ")
        # The message and nested widget will remain on the page
        import plotly.graph_objects as go
        code = []
        # Create a line plot using Plotly
        fig = go.Figure()
        code.append(
            f"""import plotly.graph_objects as go\n\nfig = go.Figure()"""
        )

        try:
            prompt += "Measured: " + st.session_state.bias_detection_metric_values
            session_id_vis = generate_session_id(5)
            r = asyncio.run(llm_visualisation_generation(prompt, session_id_vis))
            print(r)
            content = st.write_stream(response_generator(r["Text_for_user"]))
            print(r["add_traces"])
            ldict = {}
            exec(r["add_traces"], locals(), ldict)
            print("ok")
            # Update the layout
            eval(r["fig.update_layout"])
            print("ok")
            if check_empty_plot(fig):
                raise Exception("I got empty plot. Fix it.")
            code.append(f"""ldict = {{}}\nexec("{r["add_traces"]}", locals(), ldict)\n{r['fig.update_layout']}""")

        except Exception as _:
            try:
                if str(_) != "name 'fig' is not defined":
                    print(prompt + f"Avoid an error: {str(_)[:40]}.")
                    r_ = asyncio.run(
                        llm_visualisation_generation(prompt + f"Avoid an error: {str(_)[:40]}.", session_id_vis)
                    )
                    content = st.write_stream(response_generator(r["Text_for_user"]))
                    exec(r_["add_traces"])
                    # Update the layout
                    exec(r_["fig.update_layout"])
                    code.append(f"""{r_['add_traces']}\n{r_['fig.update_layout']}""")

            except Exception as e:
                print(e)
                st.write(
                    f"Sorry, something wrong with visualisation. Could you click the button again, please."
                )
        finally:
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            if check_empty_plot(fig):
                st.write(
                    "It was not possible to generate a visualization for this table."
                )
                to_show_plot = "st.write('It was not possible to generate a visualization for this table.')"
            else:
                # Display the figure in Streamlit
                st.plotly_chart(fig)
                to_show_plot = "st.plotly_chart(fig)"
            # Save the figure
            code.append(
                f"""fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')\n{to_show_plot}"""
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": "_visualization_plot_here_"}
            )
            st.session_state.visualisation_code.append("\n".join(code))

    prompt = st.chat_input(f"Message Talk With Your Bias …")

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container

        with st.chat_message("assistant", avatar=logo_picture):
            prompt = prompt.strip()
            write_log_file(prompt + "," + st.session_state.session_id + "\n")
            note = ""
            if type(st.session_state.dataset)!=type(None):
                note = " Dataset is loaded. "
                st.write(note)
            result = asyncio.run(
                    llm_class_generation(prompt + note, st.session_state.session_id)
                )
            print(result)
            # Display the result from LLM
            if result is not None:
                try:
                    LLM_err = result["LLM_err"]
                    if LLM_err == "":
                        if result['module_name'] == "None":
                            if st.session_state.bias_detection_context:
                                context = "take into account the context: bias detection metrics results: " + st.session_state.bias_detection_context + "\n"
                            else:
                                context = ""
                            st.write(f"Calling chat module.")
                            result_ = asyncio.run(llm_chat_generation(context + '\n' + prompt, st.session_state.session_id))
                            content = st.markdown(result_)
                            st.session_state.messages.append({"role": "assistant", "content": result_})
                        elif result["warning"]:
                            st.warning(result["warning"])
                            st.session_state.messages.append({"role": "assistant", "content": "WARN" + result["warning"]})

                        #text2sql,datagen,biasdet,visual,biasinj
                        if st.session_state.bias_detection_context:
                            context = "bias detection metrics results: " + st.session_state.bias_detection_context + "\n"
                        else:
                            context = ""

                        if result['module_name'] == "text2sql":
                            st.write(f"Calling module: Text-to-sql")
                            result_ = asyncio.run(llm_query_generation(prompt, st.session_state.session_id))
                            print(result_)
                            if result_ is not None:
                                LLM_err = result_["LLM_err"]
                                if LLM_err == "":
                                    sql_query = result_["sql_query"]
                                    if not sql_query:
                                        st.markdown(result_['clarifying_questions'])
                                        st.session_state.messages.append({"role": "assistant", "content": result_['clarifying_questions']})
                                        return
                                    sql_query_formated = format_sql(sql_query)
                                    expander = st.expander("Generated SQL Query: ", expanded=True)
                                    expander.code(sql_query_formated, language="sql")
                                    st.session_state.messages.append(
                                            {
                                                "role": "assistant",
                                                "content": f"sql_query:{sql_query_formated}",
                                            }
                                        )
                                    st.session_state.filename = f"csv_files/knowl_database{len(st.session_state.messages)}{st.session_state.session_id}.csv"
                                    content = st.write_stream(response_generator("Result from knowledge database:"))
                                    st.session_state.messages.append({"role": "assistant", "content": content})

                                    results_df = query_database(sql_query)
                                    if results_df is not None:
                                        # Display results
                                        print(results_df.info())
                                        st.dataframe(results_df)
                                        results_df.to_csv(
                                            st.session_state.filename, index=False
                                        )
                                        st.session_state.messages.append(
                                            {
                                                "role": "assistant",
                                                "content": st.session_state.filename,
                                            })
                                        context += results_df.to_csv(index=False)
                            result_ = asyncio.run(llm_chat_generation(prompt + "take into account the context, summarize it and answer. context:\n" + context, st.session_state.session_id))
                            content = st.markdown(result_)
                            st.session_state.messages.append({"role": "assistant", "content": result_})
                        if result['module_name'] == "datagen":
                            st.write(f"Calling module: {result['module_name']}")
                            result_, code = asyncio.run(llm_data_generation(prompt, st.session_state.session_id))
                            expander = st.expander("Generated Python Code: ", expanded=True)
                            expander.code(code, language="py")
                            st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"py:{code}",
                                }
                            )

                            st.dataframe(result_)
                            st.session_state.filename = f"csv_files/generated_{len(st.session_state.messages)}{st.session_state.session_id}.csv"
                            result_.to_csv(
                                st.session_state.filename, index=False
                            )
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": st.session_state.filename,
                                })
                            st.session_state.dataset = result_
                            if type(st.session_state.dataset) != type( None):
                                st.session_state.dataset.to_csv("csv_files/original.csv", index=False)

                        if result['module_name'] == "biasdet":
                            if type(st.session_state.dataset)!=type(None):
                                st.write(f"Calling module: Bias Detection")
                                path = "csv_files/original.csv"
                                if st.session_state.last_dataset_name:
                                    path = st.session_state.last_dataset_name
                                df = pd.read_csv(path)
                                st.dataframe(df.head(5))
                                detect_bias(path, prompt + st.session_state.protected_attr)
                            else:
                                st.warning("Bias Analysis is available only dataset is selected.")
                        if result['module_name'] == "visual":
                            st.write(f"Calling module: Visualization")
                            if st.session_state.bias_detection_metric_values:
                                visualization(prompt)
                            else:
                                st.warning("Visualization is available only after completed bias analysis.")
                        if result['module_name'] == "biasinj":
                            if type(st.session_state.dataset)!=type(None):
                                if st.session_state.suggestion:
                                    st.write(f"Calling module: Bias Injection")
                                    df = pd.read_csv("csv_files/original.csv")
                                    result_ = asyncio.run(llm_chat_generation(prompt + f"Analyze the dataset: {df.head(5)}. Write top 10 {critical_domain} domain stereotype-based bias scenarios tailored to dataset context and protected attributes. And suggest select the severity of bias injection from 10 to 90%", st.session_state.session_id))
                                    content = st.markdown(result_)
                                    st.session_state.messages.append({"role": "assistant", "content": result_})
                                    st.session_state.suggestion = False
                                else:
                                    inject_bias("csv_files/original.csv", prompt)

                            else:
                                st.warning("Bias Injection is available only dataset is selected.")


                        return
                    else:
                        print(LLM_err)

                except Exception as e:
                    raise e

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"The following error occurred during running main: {e}")
        st.write("An error occurred. Please reload the page and try again.")
