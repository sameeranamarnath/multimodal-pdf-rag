import streamlit as st
import requests

st.title("Integrated PDF Analytics with FastAPI Endpoints")

FASTAPI_UPLOAD_ENDPOINT = 'http://localhost:8000/upload/'
FASTAPI_SEARCH_ENDPOINT = 'http://localhost:8000/search/'

if "collection" not in st.session_state:
    st.session_state["collection"] = ""

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if st.button("Process PDFs"):
    if uploaded_files:
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]

        response = requests.post(FASTAPI_UPLOAD_ENDPOINT, files=files)

        if response.status_code == 200:
            st.success("PDFs processed successfully.")
            st.session_state["collection"] = response.json().get("collection")
        else:
            st.error("Error processing PDFs")
    else:
        st.warning("Please upload at least one PDF to proceed.")

query = st.text_input("Enter your query:")

if st.button("Search"):
    collection_name = st.session_state.get("collection", None)

    if not collection_name:
        st.error("Please process PDFs before searching.")
    else:
        response = requests.post(
            FASTAPI_SEARCH_ENDPOINT,
            params={"query": query, "collection": collection_name}
        )

        if response.status_code == 200:
            result = response.json()
            st.subheader("Answer:")
            st.write(result.get("answer", "No answer found"))
            st.subheader("Sources:")
            sources = result.get("sources", [])
            for src in sources:
                st.markdown(f"- {src}")
        else:
            st.error("Error retrieving search results")