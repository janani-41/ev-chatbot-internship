import streamlit as st
import pandas as pd
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EV Charger Finder", page_icon="⚡", layout="wide")
st.title("⚡ EV Charging Station Finder")
st.markdown("""
**Status:** Live | **Data Source:** Sample of User Dataset
*Ask about location, connector types (CCS, Tesla), or charging speed.*
""")

# --- 1. SETUP API KEY ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("🚨 API Key missing! Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. LOAD & INDEX DATASET (SAFE MODE) ---
@st.cache_resource
def load_db():
    try:
        # Initialize Vector DB
        chroma_client = chromadb.Client()
        embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
        collection = chroma_client.create_collection(name="ev_stations_safe", embedding_function=embedding_func)

        # CHECK FOR FILE
        file_name = "detailed_ev_charging_stations.csv"
        if not os.path.exists(file_name):
            st.error(f"❌ ERROR: '{file_name}' not found.")
            return None

        # READ CSV
        df = pd.read_csv(file_name)
        
        # --- CRITICAL FIX FOR 429 ERROR ---
        # We only take the first 50 rows to respect Free Tier limits.
        # This is enough for a demo.
        df = df.head(50) 
        # ----------------------------------

        # PREPARE TEXT FOR AI
        documents = []
        ids = []
        
        for idx, row in df.iterrows():
            info = (
                f"Station at {row['Address']}. "
                f"Operator: {row['Station Operator']}. "
                f"Connectors: {row['Connector Types']}. "
                f"Power: {row['Charging Capacity (kW)']} kW ({row['Charger Type']}). "
                f"Cost: ${row['Cost (USD/kWh)']}/kWh. "
                f"Availability: {row['Availability']}. "
                f"Rating: {row['Reviews (Rating)']}/5."
            )
            documents.append(info)
            ids.append(str(idx))
            
        # Add to Database
        collection.add(documents=documents, ids=ids)
        print(f"✅ Indexed {len(documents)} stations.")
        return collection

    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

collection = load_db()

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_query = st.chat_input("Ex: Find a fast charger with CCS connectors.")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    if collection is None:
        st.error("Database not loaded.")
        st.stop()

    # RAG: RETRIEVAL
    try:
        results = collection.query(query_texts=[user_query], n_results=3)
        
        if not results['documents'] or not results['documents'][0]:
            context = "No stations found matching the query."
        else:
            context = "\n\n".join(results['documents'][0])

        # RAG: GENERATION
        prompt = f"""
        You are an expert EV Assistant. Use the retrieved station details below to answer the user.
        
        1. If the user asks for a location, suggest the closest matches from the list.
        2. Mention the Cost, Power (kW), and Connector Type for every station you suggest.
        3. If no relevant station is found in the context, clearly say "I couldn't find a station in that area in my database."
        
        RETRIEVED STATIONS:
        {context}
        
        USER QUESTION:
        {user_query}
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
