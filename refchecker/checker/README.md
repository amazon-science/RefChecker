## Checker

Our hallucination checkers take as input a list of reference documents from retrieval or provided by users when querying LLMs, output a label list with each element chosen from `["Entailment", "Neutral", "Contradiction"]`. We provide [LLMChecker](llm_checker.py), and [NLIChecker](nli_checker.py) with the usage demonstrated below.

```python
>>> from refchecker.checker import NLIChecker

>>> checker = NLIChecker()
>>> references = [
        "`` I Dreamed a Dream '' is a song from the musical Les Mis\u00e9rables . "
        "It is a solo that is sung by the character Fantine during the first act . "
        "The music is by Claude - Michel Sch\u00f6nberg , with orchestrations by "
        "John Cameron . The English lyrics are by Neil Diamond And Herbert Kretzmer ,"
        " based on the original French libretto by Alain Boublil and Jean - Marc "
        "Natel from the original French production ."
    ] # each element is the reference or list of references for each input example.
>>> claims = [[["I Dreamed a Dream", "originally from", "the stage musical Les Mis\u00e9rables"],
               ["I Dreamed a Dream", "written by", "Claude-Michel Sch\u00f6nberg and Alain Boublil"],
               ["Anne Hathaway", "sang I Dreamed a Dream in", "the 2012 film adaptation of Les Mis\u00e9rables"]]]
# each element is the claims for each input example.
>>> checker.check(
        claims,
        references
    )  # [['Entailment', 'Contradiction', 'Neutral']]

```

For LLM-based checkers, we query LLMs with the following prompt to get the prediction:

```
I have a claim that made by a language model to a question, please help me for checking whether the claim can be entailed according to the provided reference which is related to the question. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If the claim is contradicted with the reference, answer 'Contradiction'.
If the reference is not relevant to the claim or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Question:
{question}

### Reference:
{reference}

### Claim:
{claim}

Your answer should be only a single word in ['Entailment', 'Neutral', 'Contradiction']
```

NLI-based checkers conduct pair-wise text classification on premise and hypothesis. We concatenate the question q and the reference R as the premise, and concatenate the three elements in a triplet as the hypothesis. We adopt a pre-trained language model (RoBERTa with 355M parameters) as the encoder, and perform ternary-classification as in the usual NLI setting.
