import streamlit as st
import requests
import json

st.set_page_config(page_title="ResearchCrew", page_icon="🔬", layout="wide")

API_BASE = "http://localhost:8000"

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM Provider", ["gemini", "openai"])
    api_key = st.text_input("LLM API Key", type="password", help="Your API key for the selected LLM provider")
    tavily_key = st.text_input("Tavily API Key", type="password", help="Required for web search")

    model_map = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini"}
    model = st.text_input("Model", value=model_map.get(provider, ""), help="Override default model")

    st.divider()
    st.caption("ResearchCrew v1.0")
    st.caption("Multi-agent research with LangGraph")

# --- Main area ---
st.title("🔬 ResearchCrew")
st.caption("Multi-agent research system — enter a topic and watch 4 AI agents research it for you.")

query = st.text_area(
    "Research Topic",
    placeholder="e.g., Compare RAG vs fine-tuning for domain-specific LLM applications",
    height=100,
)

can_research = query and api_key and tavily_key

if st.button("🚀 Research", type="primary", disabled=not can_research, use_container_width=True):
    if not can_research:
        st.warning("Please enter a research topic, LLM API key, and Tavily API key.")
    else:
        settings = {
            "provider": provider,
            "llm_api_key": api_key,
            "tavily_api_key": tavily_key,
            "model": model,
        }

        progress_container = st.container()
        report_container = st.container()

        with progress_container:
            st.subheader("🤖 Agent Activity")

            status_placeholder = st.empty()
            status_placeholder.info("Starting research...")

            try:
                response = requests.post(
                    f"{API_BASE}/research",
                    json={"query": query, "settings": settings},
                    timeout=300,
                )

                if response.status_code == 200:
                    result = response.json()
                    status_placeholder.success("Research complete!")

                    with report_container:
                        st.subheader("📄 Research Report")
                        st.markdown(result["report"])

                        if result.get("sources"):
                            with st.expander("📚 Sources", expanded=True):
                                for i, source in enumerate(result["sources"], 1):
                                    url = source.get("url", "N/A")
                                    title = source.get("title", url)
                                    st.markdown(f"{i}. [{title}]({url})")

                        st.divider()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Iterations", result.get("iterations", 0))
                        with col2:
                            st.metric("Sources Found", len(result.get("sources", [])))

                        st.download_button(
                            label="📥 Download Report",
                            data=result["report"],
                            file_name="research_report.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                else:
                    status_placeholder.error(f"API error: {response.status_code} — {response.text}")

            except requests.exceptions.ConnectionError:
                status_placeholder.error(
                    "Could not connect to the API server. "
                    "Make sure it's running: `uvicorn src.api.main:app --port 8000`"
                )
            except requests.exceptions.Timeout:
                status_placeholder.error("Request timed out. The research may be taking too long.")
            except Exception as e:
                status_placeholder.error(f"Unexpected error: {e}")

elif not can_research and query:
    st.info("Please provide both API keys in the sidebar to start researching.")
