class PromptTemplates:
    def __init__(self, rules_file, schema_file, examples_file, examples_clarification_file):
        # Reading instruction files
        with open(rules_file, "r") as file:
            self.rules = file.read()

        with open(schema_file, "r") as file:
            self.database_schema_description = file.read()

        with open(examples_file, "r") as file:
            self.examples = file.read()

        with open(examples_clarification_file, "r") as file:
            self.examples_clarification = file.read()

        self.CONDENSE_PROMPT = f"""
        # Context:
        You have access to previous conversation history with the user in the Chat_history.
        
        {self.database_schema_description}
        
        {self.rules}
        
        {self.examples_clarification}
        
        ## If an essential details are not provided, make sure to return the SQL query to fetch all relevant data from the database.
        ## Here is an example structure to follow:
        ### Identify the main table and relevant columns.
        ### Identify any related tables and necessary joins.
        ### Form the query using 'SELECT'.
        ### Ensure proper join conditions to link related tables.Give proper column names.
        ### Give as much data as possible if details are not specified and it complies with rules.
        ### Respond with SQL queries adhering to the above guidelines.
        ### Maintain consistency in your response.
        ### If client id or any value is not found simply give data in a list.DO NOT apply conditions there.
        
        {self.examples}
        Chat_history is described below. """
        self.TALK_CONDENSE_PROMPT = f"""
        # Context:
        You are “Talk with Your Bias” designed by Oleksandr Bezrukov. This agent interacts through a natural language interface, guiding users in identifying potential biases and applying fairness-enhancing techniques. It addresses diverse user needs, enabling intuitive controlled dataset modification with bias severity levels, domain-specific bias analysis, and bias detection without requiring programming expertise.
        Key application scenarios include:
           - Education and Awareness – The framework serves as an interactive tool for students, educators, and policymakers to study bias in AI. Users can explore real and synthetic datasets, analyze fairness metrics, and examine bias manifestations through hands-on experimentation.
           - Bias Research and Algorithm Evaluation – Researchers can generate and modify datasets with controlled bias levels, including biased versions of real-world datasets. The framework supports stereotype-based bias injection for systematic evaluation of bias detection and mitigation algorithms, facilitating reproducible research.
           - Bias Management for AI Practitioners – Data scientists can detect, visualize, and manipulate bias in datasets used for AI product development, particularly in critical domains. The tool enables bias-aware model evaluation, stress-testing under different bias conditions, and integration of legal considerations on %related to 
protected attributes.
           - Exploratory Bias Analysis for Non-Experts – Non-technical users can engage with bias analysis through interactive discussions, dataset uploads, and scenario-based bias injection.
        You have access to previous conversation history with the user in the Chat_history.
        NOT Answer questions not related to your scope.
        Chat_history is described below. """

        self.CLASSIFICATION_CONDENSE_PROMPT = f"""
        # Context:
        You get the user prompt. You analyze the user prompt and determining the appropriate processing flow. Based on the user input, you direct the request to one of the following modules:
        1. Text-to-SQL (text2sql) - if user asks any info questions about bias, fairness, tools, existing SOTA datasets, law acts, protected attributes, critical domains.
        2. Dataset Generation (datagen) - if user asks to generate dataset based on his reqs or existing SOTA datasets. 
        3. Context-Aware Bias Detection (biasdet) - if user asks to detect or assess bias in data but it available if user uploaded custom dataset or you generated it using (datagen) otherwise write warning To generate or upload data first.
        4. Visualization Engine (visual) - if user asks to Visualize data or metrics but it available if user uploaded custom dataset or you generated it using (datagen) otherwise write warning To generate or upload data first.
        5. Context-Aware Bias Injection Engine (biasinj) if user asks to inject, simulate, insert bias but it available if user uploaded custom dataset or you generated it using (datagen) otherwise write warning To generate or upload data first.
        6. None - if you cannot understand modules which module to choose, generate warning.
        RULE: if user asks more than one module, generate warning that you do it step by step and ask what user want to do first and choose "None" module.
        example: Which existing datasets have bias concerns, and which laws do they potentially violate? its text2sql
        Chat_history is described below. """

        self.DATA_GEN_CONDENSE_PROMPT = f"""
        # Context:
        Your task is to generate  You analyze the user prompt and determining the appropriate processing flow. Based on the user input, you direct the request to one of the following modules:
        1. Text-to-SQL (text2sql) - if user asks any info questions about bias, fairness, tools, existing SOTA datasets, law acts, protected attributes, critical domains.
        2. Dataset Generation (datagen) - if user asks to generate dataset based on his reqs or existing SOTA datasets. 
        3. Context-Aware Bias Detection (biasdet) - if user asks to detect or assess bias in data but it available if user uploaded custom dataset or you generated it using (datagen) otherwise write warning To generate or upload data first.
        4. Visualization Engine (visual) - if user asks to Visualize data or metrics but it available if user uploaded custom dataset or you generated it using (datagen) otherwise write warning To generate or upload data first.
        5. Context-Aware Bias Injection Engine (biasinj) if user asks to inject, simulate, insert bias but it available if user uploaded custom dataset or you generated it using (datagen) otherwise write warning To generate or upload data first.
        6. None - if you cannot understand modules which module to choose, generate warning.
        RULE: if user asks more than one module, generate warning that you do it step by step and ask what user want to do first and choose "None" module.
        example: Which existing datasets have bias concerns, and which laws do they potentially violate? its text2sql
        Chat_history is described below. """


        self.quality_control_prompt = """Let quality_value is integer value in range from 0 to 100 which quantifies how accurate the sql query answers the user question and complies with ALL RULES. quality_value = 100 means the highest quality of the sql query, otherwise quality_value = 0 means not relevant sql query.
        Compute the quality_value. If quality_value is less than 50, improve the {dialect} sql query so that quality_value > 90.
        """

        self.clarifying_prompt = """Let confidence_value is integer value in range from 0 to 100 which reflects your confidence of understanding the question and having enough info from client to generate the correct sql query following ALL RULES. confidence_value = 100 means the highest confidence, otherwise confidence_value = 0 means no confidence at all.
        Compute the confidence_value. If confidence is less than 50, rephrase the client question and ask all info you need.
        """

        self.todo_prompt = """
        Return sql query for it and no extra text around it as the resulting query will be executed in db connection to run and give response so response should be according to the context provided.
        Make sure to give correct column tables."""

        self.validation_prompt = """
        Write an FIRST_DRAFT_QUERY of the query. Then double check the FIRST_DRAFT_QUERY for following the RULES, especially compliance with the HARD Constraints. Then double check the FIRST_DRAFT_QUERY for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Follow MariaDB specific syntax, for example the FULL OUTER JOIN operation is not supported
        - Never use aliases that correspond to reserved keywords. For example do not use 'as' as an alias.
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins
        - MAKE SURE to provide correct column names.
        - MAKE SURE that joins and relationships between tables are correctly formed
        - MAKE SURE Every time join the sales table with the prediction table, first to aggregate by week to avoid creating duplicates.
        - IF FIRST_DRAFT_QUERY contains join the sales table with the prediction data table MAKE SURE to JOIN by a primary key for the prediction data table using the columns: product id, yearweek and sales category id.
        """

        self.format_prompt = """
        Use format:
         {
            "quality_value": "<<quality_value>>",
            "confidence_value": "<<confidence_value>>",
            "clarifying_questions": "<<clarifying_questions>>",
            "First draft": "<<FIRST_DRAFT_QUERY>>",
            "Final answer": "<<FINAL_ANSWER_QUERY>>"
        }
        """


