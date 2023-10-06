import openai
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from transcript_loader import aquire_transcript
from text_processing import punctuate, split_text
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
os.environ.get('OPENAI_API_KEY')
db_key = os.environ.get('Pinecone_API_KEY')

class Query_model:

    def __init__(self, url):
        super(Query_model, self).__init__()
        self.url = url
    

    def prepared_text(self):
        text = aquire_transcript(self.url)
        text = punctuate(text)
        docs = split_text(text)
        return docs
    
    def load_model(self):
        docs = self.prepared_text()
        embeddings = OpenAIEmbeddings()
        pinecone.init(
        api_key = db_key,
        environment="asia-southeast1-gcp-free"
        )
        index_name = "video-vectors"
        index = Pinecone.from_texts(docs, embeddings, index_name=index_name)

        model_name = "gpt-3.5-turbo"
        llm = OpenAI(model_name=model_name)
        chain = load_qa_chain(llm, chain_type="stuff")

        return index, chain



    def get_similiar_docs(self, index, query, k=2, score=False):
        if score:
            similar_docs = index.similarity_search_with_score(query, k=k)
        else:
            similar_docs = index.similarity_search(query, k=k)
        return similar_docs

    def get_answer(self, query):
        index, chain = self.load_model()
        similar_docs = self.get_similiar_docs(index, query=query)
        answer = chain.run(input_documents=similar_docs, question=query)
        return answer

    
        
        

