# ui/app.py

import streamlit as st
from scripts.rag_core import rag_query_pipeline

st.set_page_config(page_title="KCC Query Assistant", page_icon="🌾")

st.title("🌾 KCC Query Assistant")
st.markdown("Ask agricultural queries using offline Kisan Call Center data.")

query = st.text_input("Enter your question:")
submit = st.button("Get Answer")

if submit and query.strip():
    with st.spinner("Thinking..."):
        try:
            answer, sources = rag_query_pipeline(query)
            if answer:
                st.success("✅ Answer from Local KCC Data:")
                st.markdown(answer)
                st.markdown("**🔍 Retrieved Context:**")
                for i, src in enumerate(sources, 1):
                    st.markdown(f"{i}. {src}")
            else:
                st.warning("⚠️ No relevant KCC data found.")
                st.info("Try rephrasing the question or use fallback online search.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
