# RAG-Enhanced-Knowledge-Retrieval-System
Architecture Diagram
![Architecture Diagram](https://github.com/BigDataIA-Spring2024-Sec1-Team1/Assignment5/blob/main/part2architecturediagram.jpg)

## Technologies Used

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Amazon AWS](https://img.shields.io/badge/Amazon_AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-<COLOR_CODE>?style=for-the-badge&logoColor=white)
![Langchain](https://img.shields.io/badge/Langchain-<COLOR_CODE>?style=for-the-badge&logoColor=white)


## Problem Statement
You are an enterprise experimenting with the use of Models as a service APIs
to build intelligent applications for knowledge retrieval and Q/A tasks.
In this assignment, we will leverage Pinecone and OpenAI api’s for:
1. Creating knowledge summaries using OpenAI’s GPT
2. Generating a knowledge base (Q/A) providing context
3. Using a vector database to find and answer questions.
4. Use the knowledge summaries from 1 to answer questions


## Project Goals
1. Your end-user application should use Streamlit as an interface to accept the pdf-file and store it in S3

2. You will then kickoff the pipeline (through Streamlit) to initiate the loading of the data into Snowflake

3. You will then invoke the second API service to interface with Snowflake and bring back results into Streamlit

## Question 1

### Prerequisites

Before running the script, ensure you have the following installed:
- OpenAI GPT
- Streamlit
- S3
- PineCone
- Langchain

## Description

Streamlit:

- It is used in our application as an User Interface, so users can login and upload file and as well as look into content.


PineCone:
- Excels in fast retrieval of relevant data through vector databases, crucial for tasks like question-answering and document similarity checks.
- Supports scalable, enterprise-level applications and seamlessly integrates with AI models like OpenAI's GPT, enhancing knowledge base generation and retrieval capabilities.

  Langchain:
- LangChain can help in generating summaries of the key Learning Outcome Statements (LOS) by leveraging GPT models.
- Generating a Knowledge Base (Q/A) providing context: LangChain can assist in generating question-answer sets by utilizing GPT models to generate questions based on the context provided.
- Using a Vector Database to find and answer questions: While LangChain itself might not be directly involved in this step, it can aid in preprocessing and understanding the questions and answers retrieved from the vector database.
- Using the knowledge summaries to answer questions: Again, LangChain can help in processing the questions and answers, as well as in generating responses based on the knowledge summaries.


## CodeLab - 
https://codelabs-preview.appspot.com/?file_id=1QEQsZ2bjpKI9WmCu933177kUhZuZoLK7z7VWFwk-B5U#0

## Demo - 
https://www.youtube.com/watch?v=NIfSHmdudNo

  ## Contribution
WE ATTEST THAT WE HAVEN'T USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK

| Contributor | Contributions            | Percentage |
|-------------|--------------------------|------------|
| Dev Mithunisvar Premraj       | Part 1, Part 4 and Streamlit|33.33%|
| Aneesh Koka        | Part 3 and Part 4 | 33.33% |
| Rishabh Shah         | Part 2 and Streamlit| 33.33% |

