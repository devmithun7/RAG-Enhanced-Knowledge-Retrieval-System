from dotenv import load_dotenv
import os
import requests
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
#from fastapi import FastAPI
import requests
import numpy as np
#app=FastAPI()
import sys
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
import random
import openai
from pinecone import PodSpec
from typing import Any, List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
import boto3
from io import StringIO
import re
import string
import pinecone
from langchain import VectorDBQA, OpenAI

def process_qasetb_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Find all questions, options, and correct answers using regular expressions
    qa_pattern = re.compile(r'(\d+\..*?)(?=\n\d+\.|\Z)', re.DOTALL)
    question_option_pattern = re.compile(r'(\d+\..*?)\n([a-d])\)\s*(.*?)\n', re.DOTALL)

    # Extract questions, options, and correct answers
    question_option_tuples = [(match.group(1).strip(), match.group(2).strip(), match.group(3).strip()) 
                              for match in question_option_pattern.finditer(text)]

    # Create a DataFrame
    dataframe_setb = pd.DataFrame(question_option_tuples, columns=['Question', 'Option', 'Answer'])

    return dataframe_setb

api_key = os.environ.get("PINECONE_API_KEY")
print("hi",api_key)
pc = Pinecone(api_key='6421fb24-dfb4-4ed0-a3be-578eb0fedbc1')
file_path = "Extracted_details.csv"
df = pd.read_csv(file_path)
# & C:/Users/devmi/.virtualenvs/ice_breaker-4_Y-Iwbs/Scripts/Activate.ps
# Initialize the environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = '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'
# Constants for S3
S3_BUCKET_NAME = "bigdatacasestudy5"

# Initialize session state for storing filenames and page navigation
if 'listfilename' not in st.session_state:
    st.session_state.listfilename = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'

# Create a session using environment variables or configured AWS credentials
session = boto3.Session(
     aws_access_key_id='AKIA4MTWHTESY6QTXXEJ',
    aws_secret_access_key='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'
)
s3 = session.client('s3')
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA4MTWHTESY6QTXXEJ'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'En86bl0rvxZfiGtS60pa7t/zWM2UCyy3NyUFmVxR'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'


# Apply custom CSS styles
def apply_custom_css():
    # Your existing CSS function remains the same
    pass

def run_llm(query: str) -> Any:
    index_name = 'test2'
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})




# Login page
def login_page():
    st.title('Login to Q/A Service')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if username == "User123" and password == "Hello123":
            st.session_state.authenticated = True
            st.session_state.current_page = 'URL_page'  # Set the current page to home after login
        else:
            st.error("Login Failed. Please check your credentials.")

# Home page
def home_page():
    st.title('Welcome to the Home Page')
    # Additional home page content goes here

# Page to trigger an Airflow DAG
def URL_page():
    with st.container():
        st.title('Welcome to the File Upload Service')
        URL = st.text_input('URL: ')
        def trigger_airflow_dag(URL):
            AIRFLOW_ENDPOINT_URL = 'http://localhost:8080/api/v1/dags/process_web_data_dag/dagRuns'
            AIRFLOW_USERNAME = 'airflow'
            AIRFLOW_PASSWORD = 'airflow'
            payload = {
                "conf": {
                    "URL": URL
                }
            }
            try:
                response = requests.post(
                    AIRFLOW_ENDPOINT_URL,
                    json=payload,
                    auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD))
                if response.status_code == 200:
                    st.success("DAG triggered successfully.")
                else:
                    st.error(f"Failed to trigger DAG: {response.text}")
            except requests.ConnectionError as e:
                st.error(f"Connection error: {e}")
        if st.button('Submit URL'):
            trigger_airflow_dag(URL)

       
       


# Page for searching and processing content
def search_page():
    st.title('Generate Customized Markdown')
    # The rest of the search_page function remains the same
    Enter_Topic_Name = st.text_input('Enter Topic Name: ')
    complete = st.text_input('Are you Done ?')
    # Assuming the file is at the provided path in the virtual environment
    file_path = r"C:\Users\devmi\Documents\fastapi\Extracted_details.csv"
    if 'markdown_content' not in st.session_state:
        st.session_state.markdown_content = ""
    try:
        df = pd.read_csv(file_path)
         # Display the first few rows of the dataframe in the app
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return  # Exit the function if there is an error
    # Add a process button
    if st.button('Process'):  # This button is named 'Process'
        # Check if the entered topic name matches any value in 'Name of the Topic' column
        # Check if the entered topic name matches any value in 'Name of the Topic' column
        if Enter_Topic_Name in df['Name of the Topic'].values:
                learning_output = df.loc[df['Name of the Topic'] == Enter_Topic_Name, 'Learning Outcomes'].values[0]
                st.success(f"Learning outcomes for {Enter_Topic_Name}: {learning_output}")
                information = learning_output
        summary_template = """
        A financial analyst with an MBA interested in learning more about the {information}
● Given a {information}, create a technical note that summarizes the key Learning outcome
statement {information}. Note: Be sure to include tables, figures and equations as you
see fit. Consolidate the entire note into a document in markdown format with one heading and no subheading.
        """

        summary_prompt_template = PromptTemplate(
            input_variables=["information"], template=summary_template
        )

        openai_api_key= os.environ['OPENAI_API_KEY']
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)



        chain = LLMChain(llm=llm, prompt=summary_prompt_template)

        res = chain.invoke(input={"information": information})

            # Update the session state variable instead of a local variable
        st.session_state.markdown_content += "\n" + res.get('text', 'No content available.')
        st.write(st.session_state.markdown_content)
    else:
            st.warning("Entered topic name does not match any topic in the list.")

    if st.button('Create Markdown') and complete.lower() == 'yes':
        # Write the session state markdown content to a file
            with open('output.md', 'w') as file:
                file.write(st.session_state.markdown_content)

            st.success('The output has been stored in a document.')
            # Reset markdown_content for new session
            st.session_state.markdown_content = ""

               

            def load_local_markdown_file(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    return content

            file_path = r"C:\Users\devmi\Documents\fastapi\output.md"
            raw_documents = load_local_markdown_file(file_path)
            print(f"Loaded 1 document of length {len(raw_documents)} characters")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", "."]
            )

            # Define the Document class to wrap the document content
            class Document:
                def __init__(self, content, metadata=None):
                    self.page_content = content
                    self.metadata = metadata if metadata is not None else {}
            # Instantiate Document with raw_documents content
            document = Document(content=raw_documents)

            # Split the document into chunks
            documents = text_splitter.split_documents(documents=[document])
            print(f"Split into {len(documents)} chunks")

            # Assuming the splitter returns documents with a metadata structure
            for doc in documents:
                # Ensure this logic matches the actual structure of documents returned by the splitter
                if 'metadata' in doc and 'source' in doc['metadata']:
                    old_path = doc['metadata']['source']
                    new_url = old_path.replace("langchain-docs", "https://")
                    doc['metadata'].update({"source": new_url})
            embeddings = OpenAIEmbeddings()
            PineconeLangChain.from_documents(documents, embeddings, index_name = "test1")
            print("****Added to vectors*******")
        

#test2
    
def process_qa_file(file_path):
    # Read the entire text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Find all questions and answers using regular expressions
    qa_pattern = re.compile(r'(\d+\..*?)(?=\n\d+\.|\Z)', re.DOTALL)
    question_pattern = re.compile(r'(\d+\..*?)\nAnswer:', re.DOTALL)
    answer_pattern = re.compile(r'Answer:\s*(.*)', re.DOTALL)

    questions = [question_pattern.search(qa).group(1).strip() if question_pattern.search(qa) else None 
                 for qa in qa_pattern.findall(text)]
    answers = [answer_pattern.search(qa).group(1).strip() if answer_pattern.search(qa) else None 
               for qa in qa_pattern.findall(text)]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Question': questions,
        'Answer': answers,
        
    })
    
    return df	

 

index_name = 'test2'
api_key = '6421fb24-dfb4-4ed0-a3be-578eb0fedbc1'

# initialize connection to pinecone
pinecone_client = Pinecone(api_key=api_key)
index = pinecone_client.Index(index_name)
				





# Page for generating questions
def QA_page():
    if 'markdown_content1' not in st.session_state:
        st.session_state.markdown_content1 = ""
    st.title('Generate questions and answers')
    Enter_Topic_Name = st.text_input('Enter Topic Name: ')

    # Assuming the file is at the provided path in the virtual environment
    file_path = "Extracted_details.csv"
    df = pd.read_csv(file_path)
    #st.write(df.head())

    # Add a process button
    if st.button('Generate'):  
        # Check if the entered topic name matches any value in 'Name of the Topic' column
        if Enter_Topic_Name in df['Name of the Topic'].values:
            summary_output = df.loc[df['Name of the Topic'] == Enter_Topic_Name, 'Summary'].values[0]
            learning_output = df.loc[df['Name of the Topic'] == Enter_Topic_Name, 'Learning Outcomes'].values[0]
            introduction = df.loc[df['Name of the Topic'] == Enter_Topic_Name, 'Introduction'].values[0]
            #st.write(df.head())
            #st.success(f"Summary outcomes for {Enter_Topic_Name}: {summary_output}")
            #st.success(f"Learning outcomes for {Enter_Topic_Name}: {learning_output}")
            #st.success(f"Introduction for {Enter_Topic_Name}: {introduction}")
            
            # Concatenate summary, learning outcomes, and introduction
            information = summary_output + learning_output + introduction
            
            # Define the template for generating multiple-choice questions
            summary_template = """
            Based on the following {information}, generate strictly only 50 multiple-choice questions and not less than 50 questions. Try to be crative with the options incase if not able to generate 50 questions.
            Each question should have 4 options (a, b, c, d) in a separate line.  
            Mention answers of each question with the correct option listed explicitly on a separate line.  
    
            """
            
            # Create a PromptTemplate object
            summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

            # Initialize ChatOpenAI instance
            openai_api_key = os.environ['OPENAI_API_KEY']
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

            # Create LLMChain instance
            chain = LLMChain(llm=llm, prompt=summary_prompt_template)

            # Invoke the chain to generate questions
            res = chain.invoke(input={"information": information})
            st.session_state.markdown_content1 += "\n" + res.get('text', 'No content available.')
            st.write(st.session_state.markdown_content1)

    if st.button('Finished'):
            # Write the generated questions and answers to a file
                with open('generated_questions_answers.txt', 'w') as file:
                    file.write(st.session_state.markdown_content1)
                    st.success("Generated questions and answers have been saved to 'generated_questions_answers.txt' file.")
    
    if st.button('Push pincone'): 
        file_path = 'generated_questions_answers.txt'

        # Call the function to process the file and create a DataFrame
        # The actual call is commented out to prevent execution here
        df = process_qa_file(file_path)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)

        api_key = '6421fb24-dfb4-4ed0-a3be-578eb0fedbc1'
        pc = Pinecone(api_key=api_key)

        # Initialize OpenAI client
        openai.api_key = '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'

        # Define the Pinecone index name and namespace

        # Namespace for storing questions

        def generate_question_embedding(question_text, embed_model="text-embedding-3-small"):
            # Generate embedding for the question using the specified model
            response = openai.Embedding.create(input=question_text, engine=embed_model)
            embedding = response.data[0].embedding
            if isinstance(embedding, list):
                # If embedding is already a list, no need to convert
                return embedding
            elif isinstance(embedding, np.ndarray):
                # If embedding is a NumPy array, convert it to a list
                return embedding.tolist()
            else:
                # Handle other types of embeddings if necessary
                raise ValueError("Unsupported embedding type: {}".format(type(embedding)))



        def store_questions_in_namespace(df):
            # Specify your Pinecone API key
            api_key = '6421fb24-dfb4-4ed0-a3be-578eb0fedbc1'

            # Create a Pinecone index with the specified name
            index_name = 'test2'

            # Create an instance of Pinecone index
            pc = Pinecone(api_key=api_key)
            index = pc.Index(name=index_name)

            # Specify the namespace
            question_namespace = "questions"
            answer_namespace= 'answers'
            df['id'] = df['id'].astype(str)

            # Iterate through each question in the DataFrame
            for idx, row in df.iterrows():
                question_id = row['id']
                question_text = row['Question']
                answer_text= row['Answer']

                # Generate the embedding for the question
                embedding = generate_question_embedding(question_text)
                embeddings = generate_question_embedding(answer_text)

                # Convert embedding to a dictionary
                embedding_dict = {"id": question_id, "values": embedding}
                embeddings_dict = {"id": question_id, "values": embeddings}

                # Upsert the question embedding into the Pinecone index with the specified namespace
                index.upsert(vectors=[embedding_dict], namespace=question_namespace)
                index.upsert(vectors=[embeddings_dict], namespace=answer_namespace)


        # Call the function to store questions in the namespace
        store_questions_in_namespace(df)





             
             
   

def PartQA_page():
    st.title('View Answers')
    Enter_Topic_Name = st.text_input('Enter Query ')
    if st.button('Give output'):
            query = Enter_Topic_Name
            result = run_llm(query)
            st.write(result)


def FPartQA_page():
    st.title('View Answers')
    file_path = 'generated_questions_answers.txt'
    df = process_qa_file(file_path)  # Ensure this function reads the file into a DataFrame.
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    
    if st.button('Give output'):
        data = []  # List to store each row of data for the table.

        for idx, row in df.iterrows():
            question_text = row['Question']
            query = question_text
            result = run_llm(query)  # Ensure this function processes the query.
            # Parse the output into a more readable format.
            answer = result['result']
            data.append({"Question": question_text, "Answer": answer})  # Add the question and formatted answer.
        
        # Create a DataFrame from the collected data and display it.
        results_df = pd.DataFrame(data)
        st.table(results_df)

def FPartQB_page():
    st.title('View Answers')
    file_path = 'generated_questions_setb.txt'
    df = process_qa_file(file_path)  # Ensure this function reads the file into a DataFrame.
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    
    if st.button('Give output'):
        data = []  # List to store each row of data for the table.

        for idx, row in df.iterrows():
            question_text = row['Question']
            query = question_text
            result = run_llm(query)  # Ensure this function processes the query.
            # Parse the output into a more readable format.
            answer = result['result']
            data.append({"Question": question_text, "Answer": answer})  # Add the question and formatted answer.
        
        # Create a DataFrame from the collected data and display it.
        results_df = pd.DataFrame(data)
        st.table(results_df)


def run_llm1(query: str) -> Any:
    index_name = 'test1'
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


def Part4():
    st.title('View Answers')
    file_path = 'generated_questions_answers.txt'
    df = process_qa_file(file_path)  # Ensure this function reads the file into a DataFrame.
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    
    if st.button('Give output'):
        data = []  # List to store each row of data for the table.

        for idx, row in df.iterrows():
            question_text = row['Question']
            result = run_llm1("For the given {question_text} find the nearest answer and give option. The {question_text} is in the format with questions and options")  # Ensure this function processes the query.
            # Parse the output into a more readable format.
            answer = result['result']
            data.append({"Question": question_text, "Answer": answer})  # Add the question and formatted answer.
        
        # Create a DataFrame from the collected data and display it.
        results_df = pd.DataFrame(data)
        st.table(results_df)







def questionsetB_page():
    if 'markdown_contenttt' not in st.session_state:
        st.session_state.markdown_contenttt = ""
    st.title('Generate questions')
    Enter_Topic_Name = st.text_input('Enter Topic Name: ')
    subtopic = st.text_input('Enter sub-topic you want to generate')
    # Assuming the file is at the provided path in the virtual environment
    file_path = "Extracted_details.csv"
    dataframe_setb = pd.read_csv(file_path)
    dataframe_setb.head()
    if 'markdown_content' not in st.session_state:
        st.session_state.markdown_content = ""
    # Add a process button
    if st.button('Generate'):  # This button is named 'Generate'
        print("ddddddddddddddddddddddddddddddddddddddd",dataframe_setb['Name of the Topic'].values)
        # Check if the entered topic name matches any value in 'Name of the Topic' column
        if Enter_Topic_Name in dataframe_setb['Name of the Topic'].values:
            summary_output = dataframe_setb.loc[dataframe_setb['Name of the Topic'] == Enter_Topic_Name, 'Summary'].values[0]
            learning_output = dataframe_setb.loc[dataframe_setb['Name of the Topic'] == Enter_Topic_Name, 'Learning Outcomes'].values[0]
            introduction = dataframe_setb.loc[dataframe_setb['Name of the Topic'] == Enter_Topic_Name, 'Introduction'].values[0]
            st.write(dataframe_setb.head())
            st.success(f"Summary outcomes for {Enter_Topic_Name}: {summary_output}")
            st.success(f"Learning outcomes for {Enter_Topic_Name}: {learning_output}")
            st.success(f"Introduction for {Enter_Topic_Name}: {learning_output}")
            information = summary_output + learning_output + introduction
            summary_template ="""
            Based on the following {information}, generate 50 multiple-choice questions. Each question should have 4 options (a, b, c, d), each on a new line.
            """
            summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

            print(f"Is OPENAI_API_KEY in os.environ? {'OPENAI_API_KEY' in os.environ}")
            openai_api_key= os.environ['OPENAI_API_KEY']
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

            chain = LLMChain(llm=llm, prompt=summary_prompt_template)

            res = chain.invoke(input={"information": information})

            st.session_state.markdown_contenttt += "\n" + res.get('text', 'No content available.')
            st.write(st.session_state.markdown_contenttt)
            with open('generated_questions_setb.txt', 'w') as file:
                file.write(st.session_state.markdown_contenttt)
                st.success("Generated questions and answers have been saved to 'generated_questions_setb.txt' file.")
                file_path = 'generated_questions_setb.txt'
            
            # Read the content of the file
            with open(file_path, 'r') as file:
                file_content = file.read()

            # Split the file content into lines
            lines = file_content.split('\n')
            for line in lines:
                print(line)

            # Function to check if a line represents a question
            def is_question_line(line):
                return bool(re.match(r'^\d+\.\s', line))

            # Parsing the questions and options
            questions_with_options = []
            current_question = ""
            print("ssssssssssssssssssssssssssss",lines[0])
            for line in lines:
                if is_question_line(line):  # Check if it's a new question
                    if current_question:  # If there's a collected question, append it
                        questions_with_options.append(current_question.strip())
                    current_question = line  # Start a new question
                elif line.strip().startswith(('a)', 'b)', 'c)', 'd)')):  # If it's an option, append to current question
                    current_question += ' ' + line.strip()
                    
            # Append the last question
            if current_question:
                questions_with_options.append(current_question.strip())

            # Convert to DataFrame
            df_questions_with_options = pd.DataFrame(questions_with_options, columns=["Question"])

            # Show the first few formatted questions
            print(df_questions_with_options.head())  




          
        
   


# Function to show navigation sidebar after successful login
def show_navigation():
    with st.sidebar:
        st.title("Navigation")
        if st.button('Upload'):
            st.session_state.current_page = 'URL_page'
        if st.button('Search'):
            st.session_state.current_page = 'search'
        if st.button('Questions and Answers Generator'):
            st.session_state.current_page = 'QA_page'
        if st.button('Questions Generator'):
            st.session_state.current_page = 'Q_page'
        if st.button('View Answer'):
            st.session_state.current_page = 'PartQA_page'
        if st.button('View All Answer'):
            st.session_state.current_page = 'FPartQA_page'
        if st.button('Part 4'):
            st.session_state.current_page = 'Part4'
        

# Main application function
def main():
    apply_custom_css()

    # Initialize session state variables if they are not already set
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'

    # Show login page if not authenticated
    if not st.session_state.authenticated:
        login_page()
    else:
        show_navigation()
        # Display the selected page
        if st.session_state.current_page == 'home':
            login_page()
        elif st.session_state.current_page == 'URL_page':
            URL_page()
        elif st.session_state.current_page == 'search':
            search_page()
        elif st.session_state.current_page == 'QA_page':
            QA_page()
        elif st.session_state.current_page == 'Q_page':
            questionsetB_page()
        elif st.session_state.current_page == 'PartQA_page':
            PartQA_page()
        elif st.session_state.current_page == 'FPartQA_page':
            FPartQA_page()
        elif st.session_state.current_page == 'Part4':
            Part4()


# Run the main function
if __name__ == "__main__":
    main()

    
