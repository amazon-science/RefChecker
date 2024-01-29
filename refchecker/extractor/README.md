## Claim-Triplet Extractor

In this work, we adopt LLMs as knowledge extraction models, leveraging their strong language understanding capabilities across diverse textual contexts. We provide [MistralExtractor](mistral_extractor.py) based on Supervised Fine-tuning, [MixtralExtractor](mixtral_extractor.py), [Claude2Extractor](claude2_extractor.py) and [GPT4Extractor](gpt4_extractor.py) as extractor interfaces equipped with different LLMs.


```python
>>> from refchecker.extractor import Claude2Extractor

>>> response = (
        "Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. "
        "It was announced at the company's Artificial Intelligence (AI) Day event on "
        "August 19, 2021"
    )
>>> extractor = Claude2Extractor()
>>> triplets = extractor.extract(response)
>>> print(triplets)
"""
[['Optimus', 'is', 'robotic humanoid'], ['Optimus', 'under development by', 'Tesla, Inc.'], ['Optimus', 'also known as', 'Tesla Bot'], ['Tesla, Inc.', 'announced', 'Optimus'], ['Announcement of Optimus', 'occurred at', 'Artificial Intelligence (AI) Day event'], ['Artificial Intelligence (AI) Day event', 'held on', 'August 19, 2021'], ['Artificial Intelligence (AI) Day event', 'organized by', 'Tesla, Inc.']]
"""
```

We query LLMs with the following prompt to get decomposed triplets. Each triplet is in the format of (head, relation, tail) and serves as a basic checking unit in the next stage. Note that we include the question for LLM response generation in the extraction process, because it may contain useful information in the QA scenario such as heads and relations in some triplets, as shown in the second in-context example.

```
Given a question and a candidate answer to the question, please extract a KG from the candidate answer condition on the question and represent the KG with triples formatted with ("head", "relation", "tail"). When you finished generating the KG, please output a word "<Done>".
Here are some in-context examples:

Question:
Given these paragraphs about the Tesla bot, what is its alias?
Candidate Answer:
Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021
KG:
("Optimus", "is", "robotic humanoid")
("Optimus", "under development by", "Tesla, Inc.")
("Optimus", "also known as", "Tesla Bot")
("Tesla, Inc.", "announced", "Optimus")
("Announcement of Optimus", "occured at", "Artificial Intelligence (AI) Day event")
("Artificial Intelligence (AI) Day event", "held on", "August 19, 2021")
("Artificial Intelligence (AI) Day event", "organized by", "Tesla, Inc.")
<Done>

Question:
here is some text about Andre Weiss, how many years was Andre at University of Dijon in Paris?
Candidate Answer:
11 years
KG:
("Andre Weiss at University of Dijon in Paris", "duration", "11 years")
<Done>

Now geneate the KG for the following candidate answer based on the provided question:

Question:
{q}?
Candidate Answer:
{a}
KG:
```