import streamlit as st
import pandas as pd
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="EV Charger Bot", page_icon="⚡")
st.title("⚡ EV Charging Station Finder")
st.write("I can find charging stations based on your location and vehicle type.")

# --- 1. SETUP API KEY ---
# This looks for the key in the Cloud Secrets.
# If not found (running locally without setup), it warns you.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("API Key not found! Please set it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. PREPARE DATA (RAG) ---
@st.cache_resource
def load_db():
    # Setup Vector DB (ChromaDB)
    chroma_client = chromadb.Client()
    embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
    collection = chroma_client.create_collection(name="ev_stations", embedding_function=embedding_func)

    # Load CSV
    if os.path.exists("ev_stations.csv"):
        df = pd.read_csv("ev_stations.csv")
        
        # Convert rows to text for the AI to read
        documents = []
        ids = []
        for idx, row in df.iterrows():
            info = f"{row['Station Name']} located at {row['Location']}. Type: {row['Charger Type']}. Power: {row['Power']}. Status: {row['Status']}."
            documents.append(info)
            ids.append(str(idx))
            
        collection.add(documents=documents, ids=ids)
        return collection
    return None

collection = load_db()

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_query = st.chat_input("Ex: I have a Tesla near Main St. Any chargers?")

if user_query:
    # Show User Message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # RETRIEVAL: Search for relevant stations
    results = collection.query(query_texts=[user_query], n_results=2)
    found_data = "\n".join(results['documents'][0])

    # GENERATION: Ask Gemini
    prompt = f"""
    You are an EV Assistant. Answer the user based ONLY on the station info below.
    
    STATION INFO FOUND:
    {found_data}
    
    USER QUESTION:
    {user_query}
    """
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    
    # Show AI Response
    with st.chat_message("assistant"):
        st.markdown(response.text)
    st.session_state.messages.append({"role": "assistant", "content": response.text})