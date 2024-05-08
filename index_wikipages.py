import os
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

ENV_AVAILABLE = load_dotenv()  # loads the environment variables from .env file
assert ENV_AVAILABLE, "No .env file found"


class WikiPageList(BaseModel):
    """Data model for list of wiki pages"""

    pages: List[str]


def wikipage_list(query) -> WikiPageList:
    print("Extracting Wikipedia pages...")
    """This function extracts the Wikipedia pages from the query using OpenAI's GPT-3 API and returns them as a list (but stored in a WikiPageList object).
    Args:
        query (str): The query string containing the Wikipedia pages to extract. It must start with `please index:` followed by the Wikipedia pages.
    Returns:
        WikiPageList: A data model object containing the list of Wikipedia pages extracted from the query.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("No OpenAI API key found")

    prompt_template_str = """
    Given the input {query}, extract the Wikipedia pages mentioned after "please index:" and return them as a list.
    If only one page is mentioned, return a single element list.
    """

    program = OpenAIPydanticProgram.from_defaults(
        output_cls=WikiPageList,
        prompt_template_str=prompt_template_str,
        verbose=True,
    )

    wikipage_requests = program(query=query)
    return wikipage_requests


def create_wikidocs(wikipage_requests: WikiPageList):
    print("Creating Wikipedia documents...")
    """This function creates a list of Wikipedia documents from the list of Wikipedia pages.
    Args:
        wikipage_requests (WikiPageList): A data model object containing the list of Wikipedia pages.
    Returns:
        List[str]: A list of Wikipedia documents extracted from the Wikipedia pages.
    """

    reader = WikipediaReader()
    documents = reader.load_data(wikipage_requests.pages)

    return documents


def create_index(query):
    print("Creating index...")
    global index  # why this?
    wikipage_requests = wikipage_list(query)
    docs = create_wikidocs(wikipage_requests)

    index = VectorStoreIndex.from_documents(
        documents=docs, transformations=[Settings.text_splitter]
    )

    return index


if __name__ == "__main__":
    query = "/get wikipages: paris, lagos, lao"
    index = create_index(query)
    print("INDEX CREATED", index)
