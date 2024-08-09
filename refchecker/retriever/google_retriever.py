import os
import json
import math
import requests
import warnings
from typing import List, Dict, Union, Tuple

import rank_bm25
import diskcache
from bs4 import BeautifulSoup

from ..utils import get_model_batch_response, sentencize


SERPER_URL = "https://google.serper.dev/search"

PROMPT_FOR_QUERY_GEN = """Please generate a question on the given text so that when searching on Google with the question, it's possible to get some relevant information on the topics addressed in the text. Note, you just need to give the final question without quotes in one line, and additional illustration should not be included.

For example:
Input text: The Lord of the Rings trilogy consists of The Fellowship of the Ring, The Two Towers, and The Return of the King.
Output: What are the three books in The Lord of the Rings trilogy?

Input text: %s
Output: """


class GoogleRetriever:
    def __init__(self, cache_dir: str = "./.cache"):
        self.bm25 = None
        self._load_key()
        cache_dir = os.path.join(cache_dir, "serper")
        self.cache = diskcache.Cache(cache_dir)
    
    def _load_key(self):
        self.api_key = os.environ.get("SERPER_API_KEY", None)
        assert self.api_key is not None, \
            f"Require environment variable SERPER_API_KEY."

    def _query_google(self, query: str) -> dict:
        """Search Google using Serper API and retrieve abundant information"""
        if query in self.cache:
            return self.cache[query]
        else:
            payload = json.dumps({"q": query})
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            response = requests.request(
                "POST", SERPER_URL, headers=headers, data=payload
            )
            response_dict = json.loads(response.text)
            self.cache[query] = response_dict
            return response_dict
    
    def _get_queries(self, paragraph: str) -> List[str]:
        """Use LLM to generate query to search on the Internet to get relevant
        information. Currently only single query is generated."""
        prompt = PROMPT_FOR_QUERY_GEN % paragraph
        query = get_model_batch_response([prompt], model='gpt-3.5-turbo', temperature=0)[0]
        if query is None:
            raise RuntimeError(
                "Retriever: Empty response from LLM for query generation."
            )
        return [query.strip()]

    @staticmethod
    def _parse_results(results: dict) -> Tuple[List[dict], bool]:
        """Adapted from `FacTool` to utilize retrieved results as answers."""
        snippets = []
        with_answerbox = False
        if results.get("answerBox"):
            # This case indicates that Google has made a good answer to the question, and it's as desired to utilize this information.
            answer_box: dict = results.get("answerBox", {})
            if answer_box.get("answer"):
                element = {
                    "content": answer_box.get("answer"),
                    "source": answer_box.get("link"),
                }
                snippets = [element]
            elif answer_box.get("snippet"):
                element = {
                    "content": answer_box.get("snippet").replace("\n", " "),
                    "source": answer_box.get("link"),
                }
                snippets = [element]
            elif answer_box.get("snippetHighlighted"):
                element = {
                    "content": answer_box.get("snippetHighlighted"),
                    "source": answer_box.get("link"),
                }
                snippets = [element]
        if len(snippets) > 0:
            with_answerbox = True
        if results.get("knowledgeGraph"):
            kg: dict = results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                element = {
                    "content": f"{title}: {entity_type}",
                    "source": kg.get("link"),
                }
                snippets.append(element)
            description = kg.get("description")
            if description:
                element = {"content": description, "source": kg.get("link")}
                snippets.append(element)
            for attribute, value in kg.get("attributes", {}).items():
                element = {"content": f"{attribute}: {value}", "source": kg.get("link")}
                snippets.append(element)
        # TODO: set num of parsing link in parameters
        for result in results["organic"][:3]:
            if "snippet" in result:
                element = {"content": result["snippet"], "source": result["link"]}
                snippets.append(element)
            for attribute, value in result.get("attributes", {}).items():
                element = {"content": f"{attribute}: {value}", "source": result["link"]}
                snippets.append(element)

        if len(snippets) == 0:
            warnings.warn("No usable google search results.")

        return snippets, with_answerbox

    @staticmethod
    def _get_url_text(url) -> str:
        # Read page and return text
        buf = []
        try:
            soup = BeautifulSoup(
                requests.get(url, timeout=10).text, "html.parser"
            )
            for p in soup.find_all("p"):
                pt = p.get_text()
                if len(buf) == 0 or pt not in buf[-1]:
                    buf.append(pt)
            return "\n".join(buf)
        except:
            return ""

    @staticmethod
    def _split_doc(
        text: str,
        max_words_per_paragrpah=384,
        short_paragraph_threshold=96,
        preserve_threshold=8,
    ) -> List[str]:
        """Use spacy to split a document to paragraphs."""
        paras = text.splitlines()
        splitted = []
        sent_to_be_concat = ""
        accumulate_length = 0
        for p in paras:
            p = p.strip()
            if len(p) < 1:
                continue  # empty lines
            sents = sentencize(p)
            for sent in sents:
                if accumulate_length + len(sent) <= max_words_per_paragrpah:
                    sent_to_be_concat += sent.text_with_ws
                    accumulate_length += len(sent)
                else:
                    splitted.append(sent_to_be_concat)
                    sent_to_be_concat = sent.text_with_ws
                    accumulate_length = len(sent)
            if accumulate_length <= short_paragraph_threshold:
                sent_to_be_concat += " "
            else:
                splitted.append(sent_to_be_concat)
                sent_to_be_concat = ""
                accumulate_length = 0
        if accumulate_length >= preserve_threshold:
            splitted.append(sent_to_be_concat)
        return splitted

    def _process_retrieved_docs(
        self,
        docs: List[dict],
        query: str,
        best_k=8,
        max_words_per_paragraph=384,
        skip_repeated_corpus=True,
    ) -> List[Dict[str, Union[str, None]]]:  # {"content": <text>, "url": <url>}
        if len(docs) == 0:
            return None
        if len(docs) == 1:
            return docs
        else:
            links_dict = {}
            corpus, links = [], []  # List of documents
            # retrieve through the links
            for relevance in docs:
                url = relevance["source"]
                if "youtube" in url:
                    continue  # skip youtube due to slow fetching
                if url in links_dict.keys():
                    if skip_repeated_corpus:
                        continue
                    online_text = links_dict[url]
                else:
                    online_text = self._get_url_text(url)
                    links_dict[url] = online_text
                splitted_text = self._split_doc(
                    online_text, max_words_per_paragraph
                )
                corpus.extend(splitted_text)
                links.extend([url] * len(splitted_text))

            meta_doc_dict = dict(zip(corpus, links))
            tokenized_corpus = [doc.split(" ") for doc in corpus]

            bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
            best_docs = bm25.get_top_n(query.split(), corpus, n=best_k)
            return [
                {"content": k, "source": meta_doc_dict[k]}
                for k in best_docs
            ]

    def retrieve(
        self,
        text: str,
        top_k=3,
        max_words_per_paragraph=384
    ) -> List[Dict[str, Union[str, None]]]:
        """
        Search reference documents on the Internet based on LLM generated query.
        Parameters
        ----------
        text : str
            Text to be checked.
        top_k : int
            Number of reference documents to be retrieved.
        max_words_per_paragraph : int
            Maximum number of words in each reference document.
        Returns
        -------
        List[str]
            List of reference documents
        """

        # Step 1. Generate queries for searching using LLM.
        queries = self._get_queries(text)
        # Step 2. Search google with the queries.
        relevant_info_dicts, best_docs_all = [], []
        for q in queries:
            searched_results = self._query_google(q)
            parsed_results, with_answerbox = self._parse_results(
                searched_results
            )
            if with_answerbox:
                answerbox_answer, parsed_results = (
                    parsed_results[0],
                    parsed_results[1:],
                )
            relevant_info_dicts.extend(parsed_results)
            best_docs = self._process_retrieved_docs(
                relevant_info_dicts,
                q,
                best_k=math.ceil((top_k - with_answerbox) / len(queries)),
                max_words_per_paragraph=max_words_per_paragraph,
                skip_repeated_corpus=True,
            )
            if with_answerbox:
                best_docs.insert(0, answerbox_answer)
            best_docs_all.extend(best_docs)
        refs = [
            doc["content"] for doc in best_docs_all
        ]
        return refs
