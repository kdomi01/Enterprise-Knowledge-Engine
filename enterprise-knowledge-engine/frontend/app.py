import streamlit as st
import requests

# Configure core page layouts
st.set_page_config(
    page_title="Enterprise Knowledge Engine",
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8000/api/v1"

st.title("Enterprise Knowledge Engine")
st.markdown("---")

# Setup layout columns: Left side for processing, Right side for Chat execution
col_left, col_right = st.columns([1, 2])

# ==========================================
# LEFT COLUMN: DOCUMENT INGESTION WORKSPACE
# ==========================================
with col_left:
    st.header("Ingestion Workspace")
    st.subheader("Upload PDF Document")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your PDF here", 
        type=["pdf"], 
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        if st.button("Process & Index Document", use_container_width=True):
            with st.spinner("Extracting chunks, generating embeddings..."):
                try:
                    # Package file binary payload for FastAPI endpoint compliance
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/ingest/upload", files=files)
                    
                    if response.status_code == 201:
                        res_data = response.json()
                        st.success(f"Successfully processed: {res_data.get('filename')}")
                        st.info(f"Generated {res_data.get('total_chunks')} Parent-Child vectors.")
                        
                        # NESTED DIAGNOSTIC EXPANDER: Only displays right here inside the successful pass!
                        with st.expander("🔬 Inspect Raw Extracted Text Engine Output", expanded=True):
                            st.text_area(
                                "This is exactly what the backend extracted from your PDF:",
                                value=res_data.get("preview_text", "No text preview returned by backend."),
                                height=300
                            )
                    else:
                        st.error(f"Ingestion failed with status code: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Could not connect to backend server: {str(e)}")

# ==========================================
# RIGHT COLUMN: SYSTEM COGNITION CHAT INTERFACE
# ==========================================
with col_right:
    st.header("Context-Grounded Query Engine")
    
    # Initialize UI session states for persistent message logging
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_audit" not in st.session_state:
        st.session_state.last_audit = None

    # Render previous conversation histories dynamically
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Stream user text entries
    if user_query := st.chat_input("Ask a question regarding your uploaded knowledge files..."):
        # Append and display user input
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Call LangGraph API execution tracks
        with st.chat_message("assistant"):
            with st.spinner("Orchestrating graph state execution..."):
                try:
                    payload = {"query": user_query}
                    response = requests.post(f"{BACKEND_URL}/query/engine", json=payload)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        generation = res_data.get("generation")
                        
                        # Render generation output
                        st.markdown(generation)
                        st.session_state.messages.append({"role": "assistant", "content": generation})
                        
                        # Store metadata separately into the active session state for audit sidebar tracking
                        st.session_state.last_audit = {
                            "steps": res_data.get("visited_steps", []),
                            "context": res_data.get("retrieved_context", [])
                        }
                    else:
                        st.error("Engine failed to resolve an optimized response.")
                except Exception as e:
                    st.error(f"Pipeline connectivity error: {str(e)}")

# ==========================================
# SIDEBAR: SYSTEM COMPLIANCE & RE-RANK AUDIT
# ==========================================
with st.sidebar:
    st.header("🔍 Pipeline Execution Audit")
    st.markdown("---")
    
    if st.session_state.last_audit is not None:
        audit = st.session_state.last_audit
        
        # 1. Traveled Path Logging Layout
        st.subheader("⚡ LangGraph Routing History")
        for step in audit["steps"]:
            st.caption(f"🟩 Node Completed: `{step}`")
            
        st.markdown("---")
        
        # 2. Reranking Alignment Scoring Profiles
        st.subheader("📊 Cross-Encoder Scoring (v2-m3)")
        if not audit["context"]:
            st.info("No text contexts retained past the current threshold.")
        else:
            for i, match in enumerate(audit["context"]):
                with st.expander(f"Chunk #{i+1} (Score: {match.get('score', 0.0):.2f})"):
                    st.caption(f"**Source:** {match.get('source')}")
                    st.markdown(f"*{match.get('text')}*")
    else:
        st.info("Initiate a query to populate active LangGraph telemetry logs.")