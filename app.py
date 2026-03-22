import streamlit as st

from src.llm_service.pipeline.graph import FinancialAgent


st.set_page_config(
    page_title="Financial QA Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial QA Agent")
st.write("Ask a finance question and send it to the FinancialAgent.")

# Initialize the agent once (avoid reloading on every interaction)
@st.cache_resource
def get_agent():
    return FinancialAgent()

agent = get_agent()

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input(
    "Your question",
    placeholder="Example: Show me the revenue of 3M in 2023",
)

col1, col2 = st.columns([1, 1])

with col1:
    ask_clicked = st.button("Ask", use_container_width=True)

with col2:
    clear_clicked = st.button("Clear history", use_container_width=True)

if clear_clicked:
    st.session_state.history = []
    st.rerun()

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke(query)

                record = {
                    "query": query,
                    "answer": result.get("answer", "No answer generated."),
                    "metadata": result.get("metadata"),
                    "route": result.get("route"),
                    "source": result.get("source"),
                    "error": result.get("error"),
                }

                st.session_state.history.insert(0, record)

            except Exception as e:
                st.error(f"Failed to call FinancialAgent: {str(e)}")

# Show latest result
if st.session_state.history:
    latest = st.session_state.history[0]

    st.subheader("Latest Result")
    st.markdown(f"**Question:** {latest['query']}")
    st.markdown(f"**Route:** {latest['route']}")
    st.markdown(f"**Source:** {latest['source']}")

    st.markdown("**Metadata:**")
    st.json(latest["metadata"] if latest["metadata"] is not None else {})

    if latest["error"]:
        st.error(latest["error"])

    st.markdown("**Answer:**")
    st.write(latest["answer"])

    st.divider()

    st.subheader("History")
    for i, item in enumerate(st.session_state.history, start=1):
        with st.expander(f"{i}. {item['query']}"):
            st.markdown(f"**Route:** {item['route']}")
            st.markdown(f"**Source:** {item['source']}")
            st.markdown("**Metadata:**")
            st.json(item["metadata"] if item["metadata"] is not None else {})
            if item["error"]:
                st.error(item["error"])
            st.markdown("**Answer:**")
            st.write(item["answer"])