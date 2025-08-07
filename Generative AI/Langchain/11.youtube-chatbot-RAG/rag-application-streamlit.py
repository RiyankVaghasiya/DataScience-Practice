import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Set Streamlit page configuration
st.set_page_config(page_title="RAG using LangChain", layout="centered")
st.title("🎥 RAG from YouTube Video Transcript")

# Sidebar for API key input
api_key = st.sidebar.text_input("🔑 Enter your Groq API Key", type="password")
video_id = st.text_input("📹 Enter YouTube Video ID (e.g., 3dhcmeOTZ_Q)")

question = st.text_input("❓ Ask a question based on the transcript")

if st.button("Generate Answer"):
    if not api_key or not video_id or not question:
        st.error("Please fill in all required fields.")
    else:
        try:
            # Step 1a - Fetch Transcript
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
            transcript = " ".join(chunk.text for chunk in transcript_list)

            # Step 1b - Split Text
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            # Step 1c - Embed and Vector Store
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )

            # Step 2 - Retrieval
            retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 4})

            # Step 3 - Prompt and LLM
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            )

            llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, api_key=api_key)

            # Building Chain
            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            parser = StrOutputParser()

            main_chain = parallel_chain | prompt | llm | parser

            # Step 4 - Run the Chain
            with st.spinner("Generating answer..."):
                answer = main_chain.invoke(question)
                st.success("✅ Answer:")
                st.markdown(answer)

        except TranscriptsDisabled:
            st.error("❌ Transcript not available for this video.")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
