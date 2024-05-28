import streamlit as st
import os
import openai
from langchain_community.chat_models import AzureChatOpenAI
from dotenv import load_dotenv

import query_data

st.set_page_config(layout="wide")

st.subheader("DRC Meeting Minute Explorer")

query = st.text_input("What do you want to know?")

answer, context = query_data.run_query(query)

st.write(answer)
st.write("The following context was used to answer your question:\n\n" + context)




