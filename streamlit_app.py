import streamlit as st
import requests # Keep requests for potential future external APIs if needed, but not for local chat
import pandas as pd
import duckdb
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import kagglehub
# from dotenv import load_dotenv # Removed as st.secrets is used for API key

# Load environment variables (for local testing, not directly for Streamlit Cloud deployment)
# load_dotenv() # Removed as st.secrets is used for API key

# --- Global Constants and Variables ---
CHROMA_PERSIST_PATH = './chroma_store'
collection_name = 'schema_embeddings'
dataset_parent_folder = 'Ecommerce Order Dataset'
dataset_subfolder = 'train'

csv_files = {
    'customers': 'df_Customers.csv',
    'orders': 'df_Orders.csv',
    'order_items': 'df_OrderItems.csv',
    'payments': 'df_Payments.csv',
    'products': 'df_Products.csv'
}

# Download Kaggle dataset path (will use cached version if already downloaded)
path = kagglehub.dataset_download("bytadit/ecommerce-order-dataset")
base_path_for_csvs = os.path.join(path, dataset_parent_folder, dataset_subfolder)

# --- Cached Resource Initialization Functions ---

@st.cache_resource
def get_duckdb_connection():
    """Establishes an in-memory DuckDB connection and loads CSVs."""
    con = duckdb.connect(database=':memory:', read_only=False)
    print("Loading CSVs into pandas DataFrames and then into DuckDB...")
    for table_name, file_name in csv_files.items():
        full_path = os.path.join(base_path_for_csvs, file_name)
        df = pd.read_csv(full_path)
        con.execute(f'CREATE TABLE {table_name} AS SELECT * FROM df');
        print(f"Table '{table_name}' created in DuckDB.")
    print("DuckDB tables loaded successfully.")
    return con

@st.cache_resource
def get_schema_descriptions(con):
    """Extracts and formats schema descriptions from DuckDB."""
    table_names = con.execute('PRAGMA show_tables;').fetchdf()['name'].tolist()
    schema_descriptions = []
    for table_name in table_names:
        column_info = con.execute(f"PRAGMA table_info('{table_name}');").fetchdf()
        columns = []
        for index, row in column_info.iterrows():
            col_name = row['name']
            col_type = row['type']
            columns.append(f"`{col_name}` ({col_type})")
        columns_str = '; '.join(columns)
        table_description = f"Table `{table_name}` has columns: {columns_str}"
        schema_descriptions.append(table_description)
    print("Schema descriptions extracted.")
    return schema_descriptions

@st.cache_resource
def get_embedding_model():
    """Loads the pre-trained SentenceTransformer embedding model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("'all-MiniLM-L6-v2' embedding model loaded.")
    return model

@st.cache_resource
def get_schema_embeddings(embedding_model, schema_descriptions):
    """Generates embeddings for schema descriptions."""
    embeddings = embedding_model.encode(schema_descriptions)
    print("Schema embeddings generated.")
    return embeddings

@st.cache_resource
def get_chromadb_collection(schema_descriptions, schema_embeddings):
    """Initializes ChromaDB client and populates the collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    collection = client.get_or_create_collection(name=collection_name)
    ids = [f"schema_{i}" for i in range(len(schema_descriptions))]
    # Check if collection is empty before adding
    if collection.count() == 0:
        collection.add(
            documents=schema_descriptions,
            embeddings=schema_embeddings.tolist(),
            ids=ids
        )
        print(f"Added {collection.count()} items to '{collection_name}' collection.")
    else:
        print(f"'{collection_name}' collection already contains {collection.count()} items. Skipping add.")
    return collection

@st.cache_resource
def get_llm():
    """Initializes the Google Generative AI LLM."""
    # Use st.secrets for Streamlit Cloud deployment or os.getenv for local testing
    # api_key = os.getenv('GOOGLE_API_KEY') # For local testing - Removed
    api_key = ""

    if not api_key:
        st.error("Google API Key not found. Please set GOOGLE_API_KEY in .streamlit/secrets.toml or Streamlit secrets.")
        st.stop()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=api_key)
    print("LLM 'gemini-2.5-flash' initialized.")
    return llm

# --- Core Logic Functions ---

def generate_sql_query(user_question, llm, embedding_model, chroma_collection):
    """Generates an SQL query based on a natural language question and relevant schema information."""
    question_embedding = embedding_model.encode([user_question]).tolist()

    retrieved_schemas = chroma_collection.query(
        query_embeddings=question_embedding,
        n_results=2
    )

    context = "\n".join(retrieved_schemas['documents'][0]) if retrieved_schemas['documents'] else "No relevant schema found."

    prompt = f"""Given the following database schema:
{context}

Generate a DuckDB SQL query to answer the following question: "{user_question}"

Only return the SQL query, without any additional text or explanations.
"""

    response = llm.invoke(prompt)
    sql_query = response.content.strip()

    # Strip markdown code block delimiters
    if sql_query.startswith("```sql") and sql_query.endswith("```"):
        sql_query = sql_query[len("```sql"): -len("```")].strip()
    elif sql_query.startswith("```") and sql_query.endswith("```"):
        sql_query = sql_query[len("```"): -len("```")].strip()

    return sql_query

def execute_sql_query(sql_query, duckdb_con):
    """Executes a given SQL query against the DuckDB database."""
    try:
        result_df = duckdb_con.execute(sql_query).fetchdf()
        return result_df.to_dict(orient="records")
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="Text-to-SQL E-commerce Analyzer", layout="wide")
st.title("🛒 Text-to-SQL E-commerce Analyzer")

st.markdown("Ask natural language questions about your e-commerce data and get SQL queries and results!")

# Initialize all cached resources
con = get_duckdb_connection()
schema_descriptions = get_schema_descriptions(con)
embedding_model = get_embedding_model()
schema_embeddings = get_schema_embeddings(embedding_model, schema_descriptions)
chroma_collection = get_chromadb_collection(schema_descriptions, schema_embeddings)
llm = get_llm()

# --- Sidebar for Schema Display ---
st.sidebar.header("Database Schema")

if schema_descriptions:
    with st.sidebar.expander("View Schema Details"):
        for desc in schema_descriptions:
            st.write(desc)
else:
    st.sidebar.warning("Schema not loaded. Please ensure all setup steps are correct.")

# --- Main Chat Interface ---

# Initialize chat history if not present
if "messages" not in st.session_state: # Typo corrected here
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "generated_sql" in message:
            st.code(message["generated_sql"], language="sql")
        if "results" in message and message["results"] is not None:
            if message["results"]:
                st.dataframe(pd.DataFrame(message["results"])) # Convert list of dicts to DataFrame
            else:
                st.info("Query executed successfully, but returned no results.")

# React to user input
if prompt := st.chat_input("What would you like to know about the e-commerce data?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Generating SQL and fetching results..."):
        try:
            # Generate SQL directly
            generated_sql = generate_sql_query(prompt, llm, embedding_model, chroma_collection)

            # Execute SQL directly
            results = execute_sql_query(generated_sql, con)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                if generated_sql and results is not None:
                    st.markdown("Here is the generated SQL query and its results:")
                    st.code(generated_sql, language="sql")
                    if results:
                        st.dataframe(pd.DataFrame(results))
                    else:
                        st.info("Query executed successfully, but returned no results.")
                elif generated_sql and results is None:
                    st.markdown("Generated SQL query:")
                    st.code(generated_sql, language="sql")
                    st.error("SQL execution failed. Check the console for errors.")
                else:
                    st.error("Could not generate SQL or execute query.")

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here is the generated SQL query and its results:",
                "generated_sql": generated_sql,
                "results": results
            })

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
