import chromadb
import nest_asyncio
import openai
import os
import pandas as pd
import warnings
from datasets import Dataset    # Hugging Face datasets library
from dotenv import load_dotenv
from langchain import hub
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.generator import TestsetGenerator

warnings.filterwarnings("ignore")

# Required for ragas runner thread
nest_asyncio.apply()

# Variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
embedding_function = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
pdf_path = "google-2023-environmental-report.pdf"
collection_name = "google_environmental_report"
str_output_parser = StrOutputParser()
user_query = "What are Google's environmental initiatives?"

# LLMs and embeddings
embedding_ada = "text-embedding-ada-002"
model_gpt35="gpt-3.5-turbo"
model_gpt4="gpt-4o-mini"

embedding_function = OpenAIEmbeddings(model=embedding_ada, openai_api_key=openai.api_key)
llm = ChatOpenAI(model=model_gpt35, openai_api_key=openai.api_key, temperature=0.0)
generator_llm = ChatOpenAI(model=model_gpt35, openai_api_key=openai.api_key, temperature=0.0)
critic_llm = ChatOpenAI(model=model_gpt4, openai_api_key=openai.api_key, temperature=0.0)


# RAG INDEXING (SET UP THE DOCUMENT STORE)
# ========================================

# Load the PDF document and extract text
pdf_reader = PdfReader(pdf_path)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Split the text
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
splits = character_splitter.split_text(text)

dense_documents = [Document(page_content=text, metadata={"id": str(i), "source": "dense"}) for i, text in enumerate(splits)]
sparse_documents = [Document(page_content=text, metadata={"id": str(i), "source": "sparse"}) for i, text in enumerate(splits)]

chroma_client = chromadb.Client()
vectorstore = Chroma.from_documents(
    documents=dense_documents,
    embedding=embedding_function,
    collection_name=collection_name,
    client=chroma_client
)

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
sparse_retriever = BM25Retriever.from_documents(sparse_documents, k=10)
ensemble_retriever = EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever], weights=[0.5, 0.5], c=0)


# RAG RETRIEVAL AND GENERATION
# ============================

# Prompt - ignore LangSmith warning
prompt = hub.pull("jclemens24/rag-prompt")

# Relevance check prompt
relevance_prompt_template = PromptTemplate.from_template(
    """
    Given the following question and retrieved context, determine if the context is relevant to the question.
    Provide a score from 1 to 5, where 1 is not at all relevant and 5 is highly relevant.
    Return ONLY the numeric score, without any additional text or explanation.

    Question: {question}
    Retrieved Context: {retrieved_context}

    Relevance Score:"""
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_score(llm_output):
    try:
        score = float(llm_output.strip())
        return score
    except ValueError:
        return 0

# Chain it all together with LangChain
def conditional_answer(x):
    relevance_score = extract_score(x['relevance_score'])
    if relevance_score < 4:
        return "I don't know."
    else:
        return x['answer']

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | RunnableParallel(
        {"relevance_score": (
            RunnablePassthrough()
            | (lambda x: relevance_prompt_template.format(question=x['question'], retrieved_context=x['context']))
            | llm
            | str_output_parser
        ), "answer": (
            RunnablePassthrough()
            | prompt
            | llm
            | str_output_parser
        )}
    )
    | RunnablePassthrough().assign(final_answer=conditional_answer)
)

rag_chain_similarity = RunnableParallel(
    {"context": dense_retriever,
     "question": RunnablePassthrough()
}).assign(answer=rag_chain_from_docs)

rag_chain_hybrid = RunnableParallel(
    {"context": ensemble_retriever,
     "question": RunnablePassthrough()
}).assign(answer=rag_chain_from_docs)


# OPTIONAL: TEST RUN THE RAG WITH A DENSE (SEMANTIC) SEARCH ONLY AND A HYRBRID SEARCH (BOTH KEYWORD AND SEMANTIC)
# (uncomment this program code section by removing the triple-quoted string literals to run it)
# ===============================================================================================================

'''
    # Question - Submitted to the similarity / dense vector search
    result = rag_chain_similarity.invoke(user_query)
    retrieved_docs = result['context']

    print(f"Original Question to Similarity Search: {user_query}\n")
    print(f"Relevance Score: {result['answer']['relevance_score']}\n")
    print(f"Final Answer:\n{result['answer']['final_answer']}\n\n")
    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"Document {i}: Document ID: {doc.metadata['id']} source: {doc.metadata['source']}")
        print(f"Content:\n{doc.page_content}\n")

    # Question - Submitted to the hybrid / multi-vector search
    result = rag_chain_hybrid.invoke(user_query)
    retrieved_docs = result['context']

    print(f"Original Question to Dense Search: {user_query}\n")
    print(f"Relevance Score: {result['answer']['relevance_score']}\n")
    print(f"Final Answer:\n{result['answer']['final_answer']}\n\n")
    print("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"Document {i}: Document ID: {doc.metadata['id']} source: {doc.metadata['source']}")
        print(f"Content:\n{doc.page_content}\n")
'''

# SYNTHETIC DATA GENERATION
# =========================

# Generator with OpenAI models
generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embedding_function
)

# Create a list of document objects from the chunks
documents = [Document(page_content=chunk) for chunk in splits]

# Simple evolution (50%): Generates straightforward questions based on the provided documents
# Reasoning evolution (25%): Creates questions that require reasoning skills to answer effectively
# Multi_context evolution (25%): Generates questions that necessitate information from multiple 
# related sections or chunks to formulate an answer
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=30,
    distributions={
        simple: 0.5,
        reasoning: 0.25,
        multi_context: 0.25
    }
)

# Comparison dataframe (df)
testset_df = testset.to_pandas()

# Save dataframes to a CSV file in the specified directory
testset_df.to_csv(os.path.join('testset_data.csv'), index=False)

print("testset DataFrame saved successfully in the local directory.")

# Pull data from saved testset
saved_testset_df = pd.read_csv(os.path.join('testset_data.csv'))
print("testset DataFrame loaded successfully from local directory.")


# PREPARE SIMILARITY SEARCH DATASET
# =================================

# Convert the DataFrame to a dictionary
saved_testing_data = saved_testset_df.astype(str).to_dict(orient='list')

# Create the testing_dataset
saved_testing_dataset = Dataset.from_dict(saved_testing_data)

# Update the testing_dataset to include only these columns -
# "question", "ground_truth", "answer", "contexts"
saved_testing_dataset_sm = saved_testing_dataset.remove_columns(["evolution_type", "episode_done"])


# EVALUATION DATASETS FOR EACH CHAIN
# ==================================

# Function to generate answers using the RAG chain
def generate_answer(question, ground_truth, rag_chain):
    result = rag_chain.invoke(question)
    return {
        "question": question,
        "answer": result["answer"]["final_answer"],
        "contexts": [doc.page_content for doc in result["context"]],
        "ground_truth": ground_truth
    }

# Add the "question", "answer", "contexts", and "ground_truth" to the similarity dataset
testing_dataset_similarity = saved_testing_dataset_sm.map(lambda x: generate_answer(x["question"], x["ground_truth"], rag_chain_similarity), remove_columns=saved_testing_dataset_sm.column_names)

# Add the "question", "answer", "contexts", and "ground_truth" to the hybrid dataset
testing_dataset_hybrid = saved_testing_dataset_sm.map(lambda x: generate_answer(x["question"], x["ground_truth"], rag_chain_hybrid), remove_columns=saved_testing_dataset_sm.column_names)


# EVALUATION SCORING
# ==================

# Similarity search score
score_similarity = evaluate(
    testing_dataset_similarity,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    ]
)
similarity_df = score_similarity.to_pandas()

# Hybrid search score
score_hybrid = evaluate(
    testing_dataset_hybrid,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    ]
)
hybrid_df = score_hybrid.to_pandas()


# ANALYSIS
# ========

# Analysis that consolidates everything into easier-to-read scores
# Key columns to compare
key_columns = [
    'faithfulness',
    'answer_relevancy',
    'context_precision',
    'context_recall',
    'answer_correctness',
    'answer_similarity'
]

# Mean scores for each key column in similarity_df
similarity_means = similarity_df[key_columns].mean()

# Mean scores for each key column in hybrid_df
hybrid_means = hybrid_df[key_columns].mean()

# Comparison dataframe
comparison_df = pd.DataFrame({'Similarity Run': similarity_means, 'Hybrid Run': hybrid_means})

# Difference between the means
comparison_df['Difference'] = comparison_df['Similarity Run'] - comparison_df['Hybrid Run']

# Save dataframes to CSV files in the specified directory
similarity_df.to_csv(os.path.join('similarity_run_data.csv'), index=False)
hybrid_df.to_csv(os.path.join('hybrid_run_data.csv'), index=False)
comparison_df.to_csv(os.path.join('comparison_data.csv'), index=True)

print("Dataframes saved successfully in the local directory.")

# Load dataframes from CSV files
sem_df = pd.read_csv(os.path.join('similarity_run_data.csv'))
rec_df = pd.read_csv(os.path.join('hybrid_run_data.csv'))
comparison_df = pd.read_csv(os.path.join('comparison_data.csv'), index_col=0)

print("Dataframes loaded successfully from the local directory.")

# Analysis that consolidates everything into easier-to-read scores
print("\n\n")
print("Performance Comparison:")
print("\n**Retrieval**:")
print(comparison_df.loc[['context_precision', 'context_recall']])
print("\n**Generation**:")
print(comparison_df.loc[['faithfulness', 'answer_relevancy']])
print("\n**End-to-end evaluation**:")
print(comparison_df.loc[['answer_correctness', 'answer_similarity']])
print("\n\n")
