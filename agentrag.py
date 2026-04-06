from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv.ipython import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage ,AIMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

loader = PyPDFDirectoryLoader(path="./pdf")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='o200k_base',
    chunk_size=300,
    chunk_overlap=20
)

chunks = loader.load_and_split(text_splitter)


embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

vectorstore = Chroma.from_documents(
    chunks,
    embedding_model,
    collection_name="rapport_ocp_V2",
    persist_directory="./store"
)


retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 10}
)


prompt_template = """
Answer the following question based only on provided context
The context is delimited by <context> tag
The user question is delimited by <question> tag
If the answer is not found in the context, answer : JE NE SAIS PAS
<context>
{context}
</context>
<question>
{question}
</question>
"""

@tool
def search_docs(query: str):
    """Search information from documents using RAG"""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    final_prompt = prompt_template.format(
        context=context,
        question=query
    )
    
    response = llm.invoke(final_prompt)
    return response.content


@tool
def get_employee_info(name : str):
    """"Get information about a given employee (name,salary,seniority)"""
    print(f"getting employee info for {name}")
    return {"name" : name, "salary" : 100000,"seniority" : 5}

@tool
def send_email(email :str, subject : str, body : str):
    """Send an email to a given email address with a subject and body"""
    print(f"sending email to{email} with subject {subject} and body {body}")
    return f"Email sent to {email} with subject {subject} and body {body}"

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# AGENT
agent = create_agent(
    model=llm,
    tools=[get_employee_info, send_email, search_docs],
    system_prompt="""
You are an intelligent assistant.
Use search_docs for any question about documents.
If the answer is not found, say JE NE SAIS PAS.
"""
)