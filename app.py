import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import time
from groq import Groq

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EV Charger Bot", page_icon="⚡", layout="wide")
st.title("⚡ EV Charging Station Finder")
st.markdown("""
**Status:** Live | **AI:** Groq (Llama 3) | **Embeddings:** HuggingFace (Local)
*Ask about location, connector types (CCS, Tesla), or charging speed.*
""")

# --- 1. SETUP GROQ CLIENT ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("🚨 Groq API Key missing! Please add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- 2. LOAD & INDEX DATA (RAG) ---
@st.cache_resource
def load_db():
    try:
        # Initialize Vector DB
        chroma_client = chromadb.Client()
        
        # USE LOCAL EMBEDDINGS (No API Key needed for this part!)
        # This downloads a small model (all-MiniLM-L6-v2) to run on the server.
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        collection_name = "ev_stations_groq_v1"
        
        # Check if collection exists
        try:
            collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_func)
            return collection
        except:
            pass # Create new if not found

        collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_func)

        # CHECK FOR FILE
        file_name = "detailed_ev_charging_stations.csv"
        if not os.path.exists(file_name):
            st.error(f"❌ ERROR: '{file_name}' not found on GitHub.")
            return None

        # READ CSV
        df = pd.read_csv(file_name)
        
        # LIMIT DATA FOR SPEED (Top 50 is fine here as embeddings are local)
        df = df.head(50)
        
        # PROGRESS BAR
        progress_text = "Indexing stations locally... Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        total_rows = len(df)
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
            
            # Update bar every 10 items
            if idx % 5 == 0:
                percent = int(((idx + 1) / total_rows) * 100)
                my_bar.progress(percent, text=f"Indexed {idx+1}/{total_rows} stations...")

        # Add all to DB at once (Faster with local embeddings)
        collection.add(documents=documents, ids=ids)
        
        my_bar.empty()
        st.success(f"✅ Successfully indexed {len(documents)} stations!")
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
user_query = st.chat_input("Ex: I have a Tesla in Chicago. Any chargers?")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    if collection is None:
        st.error("Database not loaded.")
        st.stop()

    # RAG: RETRIEVAL
    results = collection.query(query_texts=[user_query], n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        context = "No stations found matching the query."
    else:
        context = "\n\n".join(results['documents'][0])

    # RAG: GENERATION (Using Groq)
    prompt = f"""
    You are a helpful EV Assistant. Use the retrieved station details below to answer the user.
    
    1. If the user asks for a location, suggest the closest matches from the list.
    2. Mention the Cost, Power, and Connector Type.
    3. If no station is found, say "I couldn't find a station in that area."
    
    RETRIEVED STATIONS:
    {context}
    
    USER QUESTION:
    {user_query}
    """
    
    # Call Groq API (Llama 3 is very fast)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful EV assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant", 
        )
        
        response_text = chat_completion.choices[0].message.content
        
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
    except Exception as e:
        st.error(f"Groq API Error: {e}")

