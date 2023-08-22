repo ='tiiuae/falcon-7b-instruct'
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMRequestsChain, LLMChain
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap

load_dotenv()

hub_llm = HuggingFaceHub(repo_id=repo)

prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate into French, {question}"
)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
print(hub_chain.run("It is raining outside"))

video_url = "https://www.youtube.com/watch?v=rFQ5Kmkd4jc"
loader = YoutubeLoader.from_youtube_url(video_url)
transcript = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
docs = text_splitter.split_documents(transcript)

# --------------------------------------------------------------
# Summarization with LangChain
# --------------------------------------------------------------

# Add map_prompt and combine_prompt to the chain for custom summarization
chain = load_summarize_chain(hub_llm, chain_type="map_reduce", verbose=True)
print(chain.llm_chain.prompt.template)
print('----------------------------------------------------------------------')
print(chain.combine_document_chain.llm_chain.prompt.template)

# --------------------------------------------------------------
# Test the Falcon model with text summarization
# --------------------------------------------------------------

output_summary = chain(docs, return_only_outputs=True)
# wrapped_text = textwrap.fill(
#     output_summary, width=100, break_long_words=False, replace_whitespace=False
# )
# print(wrapped_text)

print(output_summary)