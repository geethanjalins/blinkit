import streamlit as st
import chromadb
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="Feedback Analyst", layout="wide")
st.title("Business Feedback Analyst")

st.markdown("""
This application uses Semantic Search to retrieve relevant customer feedback and an LLM to state the root cause of business performance issues.
""")

# ==================================
# 1. Initialize Vector Database
# ==================================
@st.cache_resource
def get_db_collection():
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        # ChromaDB automatically handles embedding user queries 
        # using the default sentence-transformers model
        collection = client.get_collection(name="customer_feedback")
        return collection
    except Exception as e:
        st.error(f"Error connecting to Vector DB: {e}. Please run build_vector_db.py first.")
        return None

collection = get_db_collection()

# ==================================
# 2. LLM Configuration Sidebar
# ==================================
st.sidebar.header("LLM Settings")
api_key = st.sidebar.text_input("Enter your API Key", type="password")
provider = st.sidebar.selectbox("LLM Provider", ["Groq"])

# Groq offers fast inference for Llama 3 models
base_url = "https://api.groq.com/openai/v1"
model_name = "llama-3.1-8b-instant"

# ==================================
# 3. Chat Interface & RAG Pipeline
# ==================================
query = st.chat_input("E.g., Why are fruit sales down?")

if query:
    st.chat_message("user").write(query)
    
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("Please enter your API Key in the sidebar to generate analysis.")
            st.stop()
            
        if not collection:
            st.stop()
            
        with st.spinner("Retrieving relevant feedback from vector database..."):
            # A. Convert user query to embedding & Retrieve relevant feedback
            # chroma handles embedding the string automatically
            results = collection.query(
                query_texts=[query],
                n_results=15
            )
            
            # Extract feedback strings
            retrieved_docs = results['documents'][0]
            feedback_context = "\n".join([f"- {doc}" for doc in retrieved_docs])
            
        with st.spinner(f"Analyzing with {provider}..."):
            # B. Insert retrieved feedback into Generation prompt
            system_prompt = "You are an AI business analyst. Your task is to analyze customer feedback and identify root causes behind business performance issues."
            
            user_prompt = f"""
            Task:
            1. Read all customer complaints carefully.
            2. Identify recurring issues and patterns.
            3. Determine the main root cause behind the issue asked in the query.
            4. Mention product names and regions if available.
            5. Summarize the finding clearly in business language.
            
            Rules:
            - Use only the information present in the feedback.
            - Do not assume or invent details.
            
            Output Requirement:
            Provide a concise root cause summary in 1-2 sentences answering the user query.
            
            User Query: {query}
            
            Retrieved Customer Feedback:
            {feedback_context}
            """
            
            try:
                # C. Send prompt to LLM (OpenAI/Groq/Llama)
                
                
                # ADD api_key=api_key HERE:
                client = OpenAI(base_url=base_url, api_key=api_key) 
    
                response = client.chat.completions.create(
                model=model_name,
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
               ],
             temperature=0.3
       )
                
                analysis_result = response.choices[0].message.content
                
                # D. Display summarized root cause in chat UI
                st.write("### Root Cause Analysis")
                st.info(analysis_result)
                
                with st.expander("View Retrieved Feedback Context"):
                    st.write(feedback_context)
                    
            except Exception as e:
                st.error(f"Error during LLM generation: {e}")
