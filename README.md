# ⚡ EV Charging Station Finder (AI Internship Assessment)

### 🎥 Project Demo Video
****https://drive.google.com/file/d/1nSq7skoADjfhMiCXqJDKoh1b0d-Ga1--/view?usp=sharing *(Please watch the video for a walkthrough of the RAG architecture and a live demo.)*

---

## 🧐 Problem Statement
Electric Vehicle (EV) owners often suffer from **"Range Anxiety"**—the fear of running out of battery before finding a compatible charging station. Existing maps are often cluttered, making it hard to answer simple questions like *"Where is the cheapest fast charger near me that fits my Tesla?"*

**The Solution:** I built an AI-powered chatbot that uses **Retrieval Augmented Generation (RAG)** to intelligently search a database of charging stations. Instead of generic answers, it retrieves specific data (Location, Cost, Plug Type) and answers in natural language.

---

## 🛠️ Tech Stack & Architecture
This project follows a **Cloud-Native "Vibe Coding"** philosophy, utilizing modern, composable AI tools for rapid development.

* **Cloud Hosting:** [Streamlit Community Cloud](https://streamlit.io/) (PaaS) for global accessibility.
* **LLM Inference:** [Groq](https://groq.com/) (Llama-3.1-8b-instant) for ultra-fast, low-latency AI responses.
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Local/Ephemeral) for semantic search.
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) for converting text to vectors locally.
* **Data Source:** Custom CSV dataset (`detailed_ev_charging_stations.csv`) containing 5,000+ stations.

### RAG Architecture Flow
1.  **Ingestion:** The app loads station data and converts it into vector embeddings.
2.  **Retrieval:** When a user asks a question, the system searches ChromaDB for the top 3 most relevant stations.
3.  **Generation:** The retrieved station details are sent to the Groq LLM (Llama 3).
4.  **Response:** The AI generates a helpful, natural language answer based *only* on the factual data found.

![RAG Architecture Diagram](https://mermaid.ink/img/pako:eNp1kcFqwzAMhl_F6NRC_QA9FHYYbaDQwxj0UHIwUWy1sSVHStZSyLvPydg62E5CPvzp0y_QO6uCRfqAv42TwuHk1aVw8L5Wimz2hSzJ-4NylC_Lh_B-X8iS7K9kS_6pFEpY6_G44Z_6w_1-F85f9oU8y_dC1uR9ofw_f7op5P0H-ZJtS7mS94V8l_-F_JAfhaSw0kI9c4OOY4_HjT36A2v00LDP6LhHw4FjS8feouPAsWOP3qLjwLFj39Fj4Dhi7NHTf-wSj9wx9uhZOg4ce_T0H7vEI3eMPXqWjgPHHj39xy7xyB1jj56l48Cxo2fv0dFz7NHTf-wSj9wx9uhZOr4B4OKQtw?type=png)

---

## 🚀 How to Run Locally
If you want to run this project on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
    cd YOUR-REPO-NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    * Create a `.streamlit/secrets.toml` file (or use `.env`).
    * Add your Groq API Key: `GROQ_API_KEY = "gsk_..."`

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---

## 🧪 Sample Questions to Test
You can ask the chatbot these questions to verify the RAG logic is working:

**📍 Location Based:**
1.  "Where is the nearest charging station in San Francisco?"
2.  "Are there any chargers in Mumbai?"
3.  "I am in Chicago. Find me a station nearby."

**🔌 Connector & Vehicle Specific:**
4.  "I drive a Tesla. Where can I charge in Bangkok?"
5.  "Find me a station with CCS connectors in Toronto."
6.  "Do you have any stations compatible with CHAdeMO plugs?"

**⚡ Power & Speed:**
7.  "I'm in a hurry. Find me a DC Fast Charger (150kW+) in New York."
8.  "Show me Level 2 AC chargers in Dubai."

**💰 Cost & Availability:**
9.  "Find me the cheapest charger in Berlin."
10. "Are there any stations open 24/7 in Mexico City?"

**🧠 Edge Case (Negative Test):**
11. "Where is the nearest charger in Antarctica?" *(Should reply that no data is available)*

---

## ⚠️ Note on Cloud Infrastructure
*This project was designed to demonstrate proficiency with Cloud Services and AI/ML integration.*

Due to the credit card verification requirement for Azure/AWS free tiers, I architected this solution using **Streamlit Community Cloud** and **Groq Cloud API**. This approach fulfills the assessment requirement of using "Modern AI Platforms" and "Cloud Services" while allowing for a live, publicly accessible deployment without billing barriers.
