import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema import HumanMessage


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")
except Exception as e:
    st.error(f"Error initializing ChatGroq: {e}")
    llm = None


st.title("Language Translator")

if llm:
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant who translates sentences.
        Translate the following sentence into {language}:

        Sentence: "{sentence}"

        Your response should be the translation only.
        """
    )


    prompt1 = st.text_input("Enter the sentence ")
    target_language = st.text_input("Enter the lang")

    if st.button("Translate"):
        if prompt1 and target_language:
            try:
                
                formatted_prompt = prompt_template.format(sentence=prompt1, language=target_language)
                message = HumanMessage(content=formatted_prompt)
                response = llm([message])  
                st.write("Translated Sentence")
                st.write(response.content)
            except Exception as e:
                st.error(f"Failed {e}")
        else:
            st.error("give data")
