
import requests

from langchain import PromptTemplate

from langchain.chains import LLMChain
import pandas as pd
import numpy as np
import streamlit as st
from langchain_community.llms import HuggingFaceHub






if 'list_df' not in st.session_state:
    st.session_state['list_df'] =''

if 'e_list' not in st.session_state:
    st.session_state['e_list'] = ''
e_list = []
list_df = []

if 'profile' not in st.session_state:
    st.session_state['profile'] = ''


token = st.secrets['HUGGINGFACEHUB_API_TOKEN']
# print(token)


def scrape_linkedin_profile(linkedin_profile_url:str):
    
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""
    headers = {'Authorization': 'Bearer ' + st.secrets["PROXYCURL_API_KEY"]}
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=headers
    )

    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
    

summary_template = """

Name of the person is {full_name}.
Given input information {information} about {full_name} from I want you to create:
Summarize {information}.
"""

experience_template = """

Name of the person is {full_name}.
Summarize {information} in 2 lines. You have to mention names of companies where the person has worked.
"""

education_template = """
Name of the person is {full_name}.
Given input information {information} about {full_name} from I want you to create:
Summarize {information} with the insitutes where education was pursued in 2 lines
"""

p1 = st.text_input('Enter the LinkedIn profile')


if st.button('Click for summary'):
    with st.spinner("Generating response.."):
    
        llm = HuggingFaceHub(repo_id=st.secrets['repo_id'],\
                        huggingfacehub_api_token = token, model_kwargs={"temperature":1e-10, "max_length":512})
        linkedin_data1 = scrape_linkedin_profile(p1)
        full_name = linkedin_data1.get('full_name')
        
        summary_prompt_template = PromptTemplate(input_variables = ["full_name","information"],template = summary_template)
        chain = LLMChain(llm=llm, prompt = summary_prompt_template)
        df1 = chain.invoke({'full_name':full_name, 'information':linkedin_data1.get('summary')})
        df1 = df1.get('text')

        experience_prompt_template = PromptTemplate(input_variables = ["full_name","information"],template = experience_template)
        chain = LLMChain(llm=llm, prompt = experience_prompt_template)
        df2 = chain.invoke({'full_name':full_name, 'information':linkedin_data1.get('experiences')})
        df2= df2.get('text')

        education_prompt_template = PromptTemplate(input_variables = ["full_name","information"],template = education_template)
        chain = LLMChain(llm=llm, prompt = education_prompt_template)
        df3 = chain.invoke({'full_name':full_name, 'information':linkedin_data1.get('education')})
        df3= df3.get('text')

        st.write(df1+df2+df3)
