from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import flag


def main():
    load_dotenv()
    st.set_page_config(page_title="Arete Global HR")
    st.header("Ask Arete Global HR 💬")

    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    # upload file
    # pdf = st.file_uploader("Upload your PDF", type="pdf")
    option = st.selectbox(
        'Please select the county',
        ('None', 'Egypt',
         'KSA', 'UAE',
         'India'), label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled)

    filename = option + '.pdf'
    faiisname = option + '.faiis'

    if option is not 'None':

        st.session_state.visibility = "hidden"
        st.session_state.disabled = True
        st.session_state.update()
        isFile = os.path.isfile('docs/' + faiisname)

        with st.spinner('Reading ' + option + ' policies..'):
            # extract the text
            if not isFile:
                #pdf_reader = PdfReader(pdf)
                pdf_reader = PdfReader('docs/' + filename)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # split into chunks
                text_splitter = CharacterTextSplitter(
                    separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
                )
                chunks = text_splitter.split_text(text)

                # create embeddings
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                knowledge_base.save_local('docs', option)
            else:
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.load_local('docs', embeddings, option)
                # show user input

            user_question = st.text_input("Please ask your question here:")

            with st.spinner('Checking ' + option + ' office policies...'):
                if user_question:
                    docs = knowledge_base.similarity_search(user_question)

                    llm = OpenAI()
                    chain = load_qa_chain(llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(
                            input_documents=docs, question=user_question
                        )
                        print(cb)

                        st.write(response)


if __name__ == "__main__":
    main()
