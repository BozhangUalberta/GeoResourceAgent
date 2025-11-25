from typing import Annotated, Literal, Sequence, List, Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.prebuilt import InjectedState

from src.utils.chroma_utils import add_docs, query_docs
from src.utils.logger_utils import get_logger
from decouple import config 
from openai import OpenAI
import json
import os


TEXT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=1000, chunk_overlap=200
                )

@tool
async def web_url_content_reader(
    userID: Annotated[str, InjectedState("userID")],
    urls: Annotated[List[str], "urls user provided."],
):
    """
    Load, split and store contents into ChromaDB from user provided urls.
    """
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    doc_splits = TEXT_SPLITTER.split_documents(docs_list)

    documents = [doc.page_content for doc in doc_splits]
    metadatas = [doc.metadata for doc in doc_splits]

    succeed = await add_docs(userID, documents, metadatas)

    if succeed:
        return json.dumps("Urls in argument have been stored.")
    else:
        return json.dumps("Encontered error when storing documents (url), check other tools error.")


@tool
async def pdf_reader(
    userID: Annotated[str, InjectedState("userID")],
    filenames: Annotated[List[str], "pdf file names that need to be read. Must in 'filename.pdf' format."],
):
    """
    
    """
    root_path = os.path.join("src","static",userID)
    filepath_list = [os.path.join(root_path, filename) for filename in filenames]

    docs = [PyPDFLoader(filepath).load() for filepath in filepath_list]
    docs_list = [item for sublist in docs for item in sublist]

    doc_splits = TEXT_SPLITTER.split_documents(docs_list)

    documents = [doc.page_content for doc in doc_splits]
    metadatas = [doc.metadata for doc in doc_splits]

    succeed = await add_docs(userID, documents, metadatas)

    if succeed:
        return json.dumps("PDFs in argument have been stored.")
    else:
        return json.dumps("Encontered error when storing documents (pdf), check other tools error.")


@tool
async def retriever(
    userID: Annotated[str, InjectedState("userID")],
    query: Annotated[str, "proper query after enough rounds of rewrite."],
):
    """
    Get the most relevant documents from ChromaDB given proper query.

    return:
        Dict: {{id, distance, metadata, document}, {...}, {...}}
    """
    results = await query_docs(userID, query)

    # Rephrase -> {{id, distance, metadata, document}, {...}, {...}}
    results = [
        {
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
            "document": results["documents"][0][i]
        }
        for i in range(len(results["ids"][0]))
    ]

    return json.dumps(results)


def generate_response(sys_prompt, user_prompt, model="gpt-4o-mini"):
    client = OpenAI(api_key= config("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # Deterministic responses
    )
    return response.choices[0].message.content


# def grade_documents(state) -> Literal["generate", "rewrite"]:
def grade_documents(state) -> Literal["assistant", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True, api_key=config("OPENAI_API_KEY"))

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    # print(messages)
    last_message = messages[-1]
    # print("---last message---")
    # print(last_message)
    question = messages[-3].content
    # print("---question---")
    # print(question)
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        # return "generate"
        return "assistant"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    logger = get_logger(state["userID"])
    logger.info("I'm rephrasing your question to get better result...")
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    # print(messages)
    question = messages[-3].content
    # print("---question---")
    # print(question)


    msg = f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the conversation history:
        {messages[1:-3]}
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """
    msg = [HumanMessage(
        content=msg
    )]

    # Grader
    # response = generate_response("",msg)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True, api_key=config("OPENAI_API_KEY"))
    response = llm.invoke(msg)

    # response = AIMessage(content=response)
    # response.additional_kwargs={"intermediate_node": "rewrite"}
    
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[-3].content
    # print("---question---")
    # print(question)
    last_message = messages[-1]
    # print("---last_message---")
    # print(last_message)

    docs = last_message.content

    # Prompt
    prompt = PromptTemplate(template="""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer: """,
        input_variables=["context", "question"],
    )

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True, api_key=config("OPENAI_API_KEY"))

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})

    return {"messages": [response]}
