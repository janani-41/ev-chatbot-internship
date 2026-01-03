import streamlit as st
import pandas as pd
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import os
import time  # <--- Added for the delay

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EV Charger Finder", page_icon="⚡", layout="wide")
st.title("⚡ EV Charging Station Finder")
st.markdown("""
**Status:** Live | **Data:** Top 20 Stations (Safe Mode)
*Ask about location, connector types (CCS, Tesla), or charging speed.*
""")

# --- 1. SETUP API KEY ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("🚨 API Key missing! Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. SLOW LOAD & INDEX (The Fix) ---
@st.cache_resource
def load_db_safely():
    try:
        # Initialize Vector DB
        chroma_client = chromadb.Client()
        embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
        
        # Use a new name to avoid conflicts with old broken collections
        collection_name = "ev_stations_turtle_v1" 
        
        # Check if it already exists (to avoid re-loading if not needed)
        try:
            collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_func)
            print("✅ Collection found. Skipping re-indexing.")
            return collection
        except:
            pass # Collection doesn't exist, create it below

        collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_func)

        # CHECK FOR FILE
        file_name = "detailed_ev_charging_stations.csv"
        if not os.path.exists(file_name):
            st.error(f"❌ ERROR: '{file_name}' not found.")
            return None

        # READ CSV
        df = pd.read_csv(file_name)
        
        # LIMIT TO 20 ROWS (Strict Safety)
        df = df.head(20) 
        
        # PROGRESS BAR (So you know it's working)
        progress_text = "Indexing stations slowly to avoid Rate Limits... Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            # Construct Text
            info = (
                f"Station at {row['Address']}. "
                f"Operator: {row['Station Operator']}. "
                f"Connectors: {row['Connector Types']}. "
                f"Power: {row['Charging Capacity (kW)']} kW ({row['Charger Type']}). "
                f"Cost: ${row['Cost (USD/kWh)']}/kWh. "
                f"Availability: {row['Availability']}. "
                f"Rating: {row['Reviews (Rating)']}/5."
            )
            
            # ADD ONE BY ONE
            collection.add(documents=[info], ids=[str(idx)])
            
            # UPDATE PROGRESS
            percent = int(((idx + 1) / total_rows) * 100)
            my_bar.progress(percent, text=f"Indexed {idx+1}/{total_rows} stations...")
            
            # PAUSE FOR 4 SECONDS (The Rate Limit Fix)
            time.sleep(4)
            
        my_bar.empty() # Clear bar when done
        st.success(f"✅ Successfully indexed {total_rows} stations!")
        return collection

    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

collection = load_db_safely()

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
