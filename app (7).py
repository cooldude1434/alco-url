
import streamlit as st
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import AIMessage, HumanMessage

from common.utils import data_ingestion, get_embeddings, build_vector_store_database, get_llm, get_response_llm


def main():
    # Configure the settings of the webpage
    st.set_page_config(page_title="Chat with Websites", page_icon= "🧊", layout="wide")

    # Add a header
    st.header("Chat with websites using  Gemini-Pro💬🤖")

    # Input question from the user
    # user_question = st.text_input("Ask a question from the URLs")

    # Vertex AI - Google Palm text embeddings
    embeddings = get_embeddings()

    # Create a sidebar
    with st.sidebar:
        # Title of the sidebar
        st.title("URL Chatbot:")

        # List of urls
        urls = []

        for i in range(5):
            url = st.sidebar.text_input(f"URL {i+1}")
            # Append URLs if provided
            if url:
                urls.append(url)

        if st.button("Process URLs"):
            with st.spinner("Data Ingestion...Started...✅✅✅"):
                # Ingest data
                docs = data_ingestion(urls = urls)
                # Create vector store database
                build_vector_store_database(documents = docs, embeddings = embeddings)
                st.success("Done")

#     if st.button("Google Palm Output"):
#         with st.spinner("Thinking"):
#             # Load data
#             faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#             # Load LLM
#             llm  = get_llm(model_name = "text-bison@002")

#             st.write(get_response_llm(llm, faiss_index, user_question))

#             st.success("Done")
    # elif st.button("Gemini-Pro Output"):
    with st.spinner("Thinking"):
        faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                # Load LLM
        llm = get_llm(model_name="gemini-pro")

        
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?"),
                                             ]
        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            # print(f"**ragu{llm},{faiss_index},{user_query}")
            response,source = get_response_llm(llm, faiss_index,user_query)
            source_urls = set()

            for document in source:
                metadata = document.metadata
                if 'source' in metadata:
                    source_urls.add(metadata['source'])

            # Join all URLs into a single string separated by newline
            source_urls_string = '\n'.join(source_urls)
            
            response = f"{response}\nSource:\n{source_urls_string}"  

            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            print(source)
            
         # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)


        # st.write(get_response_llm(llm, faiss_index, user_question))

        # st.success("Done")

if __name__ == "__main__":
    main()
