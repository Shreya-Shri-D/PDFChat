import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from google.cloud import vision
import io

# Set up Google Cloud Vision API client
def setup_google_vision():
    return vision.ImageAnnotatorClient()

def setup_openai(api_key):
    openai.api_key = api_key

def extract_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_images_from_pdf(pdf):
    images = []
    try:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            for img in page.images:
                images.append({
                    "data": img.data,
                    "name": img.name if hasattr(img, 'name') else f"Image {len(images)+1}"
                })
    except Exception as e:
        st.error(f"Error extracting images: {e}")
    return images

# Function to analyze images using Google Vision API (e.g., for labels, etc.)
def analyze_image(image_content):
    client = setup_google_vision()
    image = vision.Image(content=image_content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    return labels

def create_knowledge_base(text, openai_api_key):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    setup_openai(openai_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def ChatPDF(text, images, openai_api_key):
    knowledge_base = create_knowledge_base(text, openai_api_key)
    st.write("Knowledge Base created")

    def ask_question(i=0):
        user_question = st.text_input("Ask a question about your PDF?", key=i)
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(openai_api_key=openai_api_key)
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)
                print(cb)

            # If the user asks for an image
            if "image of" in user_question.lower():
                keyword = user_question.split("image of")[-1].strip()
                st.write(f"Searching for an image related to {keyword}...")

                # Check if any extracted images match the query
                matching_images = [img for img in images if keyword.lower() in img['name'].lower()]
                if matching_images:
                    for img in matching_images:
                        st.image(img['data'], caption=f"Image matching {keyword}")
                else:
                    # Use Google Vision API for label detection
                    st.write(f"Analyzing images for relevance to {keyword}...")
                    for img in images:
                        labels = analyze_image(img['data'])
                        for label in labels:
                            if keyword.lower() in label.description.lower():
                                st.image(img['data'], caption=f"Found matching image for {keyword}: {label.description}")

            ask_question(i + 1)

    ask_question()

def main():
    st.set_page_config(page_title="Ask your PDF", page_icon="📄")

    hide_st_style = """
            <style>
            #mainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.header("Ask your PDF 🤔💭")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        option = st.selectbox("What you want to do with PDF📜", [
            "Extract Raw Text📄",
            "Extract Images🖼️",
            "ChatPDF💬"
        ])
        text = extract_text_from_pdf(pdf)
        images = extract_images_from_pdf(pdf)

        if option == "Extract Raw Text📄":
            st.write(text)
        elif option == "Extract Images🖼️":
            for i, img in enumerate(images):
                st.image(img['data'], caption=f"{img['name']}")
        elif option == "ChatPDF💬":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                ChatPDF(text, images, openai_api_key)

if __name__ == "__main__":
    main()
