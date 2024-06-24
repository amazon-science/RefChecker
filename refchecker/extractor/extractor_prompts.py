
LLM_Triplet_To_Claim_PROMPT_Q = """Given a question and a response to the question, please extract a KG from the response condition on the question and represent the KG with triplets formatted with `(subject, predicate, object)`. In addition, you should attribute the triplets to the sentences followed by the sentence ids.

After extracting the KG, you should convert the KG to a list of claims. Each claim should satisfy the following criteria:
* A claim is a piece of `knowledge point` in the response.
* A claim should be fine-grained. One claim should not contain more than one pieces of knowledge.
* A claim should be self-contained, it is not dependent on other claims.
* Each claim should truly reflect the meaning to be expressed in the response, and the information in the claim should be complete and unambiguous, and necessary conditions and attributes should not be missed. For example, for the text "Donald Trump won the presidential election in 2016.", the claim "Donald Trump won the presidential election" is a bad claim where it misses necessary information "in 2016", so a complete claim should be "Donald Trump won the presidential election in 2016".
* Some sentence in the response may not contain claims. Some sentence may contain one or more claims. A claim may occur across multiple sentences.
* Opinions, speculations or questions in the text are not claims.

We have added sentence IDs at the beginning of the sentences in the text, you should output a claim followed by a list of IDs, and these IDs stand for the sentences this claim attribute to. The extraction process:
1. You should first identify whether there are claims in the text. If there are no claims, output "Abstain".
2. If there are claims in the text, you should first identify knowledge points in the text and formuate claims for them.
3. Note that a knowledge point may just reflect partial information in the text, so if we take it in isolation, it may not reflect the original meaning in the text, you should fill in the missing information to make it complete and clear. The information in a claim may come across multiple sentences in the text.
4. Formulate the claim into one sentence, and followed by the sentence indices as the attributed sentences in the text.
5. Make sure you have generated all the claims in the text.

Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the text is factual or not, just extract the claims from it.

Here are some examples:

### Question
Given these paragraphs about the Tesla bot, what is its alias?
### Response
[1] Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. [2] It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.
### KG
(Optimus, is, robotic humanoid) [1]
(Optimus, under development by, Tesla, Inc.) [1]
(Optimus, also known as, Tesla Bot) [1]
(Tesla, Inc., announced, Optimus) [2]
(Announcement of Optimus, occurred at, Artificial Intelligence (AI) Day event) [2]
(Artificial Intelligence (AI) Day event, held on, August 19, 2021) [2]
(Artificial Intelligence (AI) Day event, organized by, Tesla, Inc.) [2]
### Claims
Optimus is a robotic humanoid. [1]
Optimus is under development by Tesla, Inc. [1]
Optimus is also known as Tesla Bot. [1]
Tesla, Inc. announced Optimus at the Artificial Intelligence (AI) Day event August 19, 2021. [2]

### Question
here is some text about Andre Weiss, how many years was Andre at University of Dijon in Paris?
### Response
[1] 11 years
### KG
("Andre Weiss at University of Dijon in Paris", "duration", "11 years") [1]
### Claims
Andre Weiss was at University of Dijon in Paris for 11 years. [1]

### Question
who sang i ran all the way home
### Response
[1] I'm sorry, I cannot provide an accurate answer without additional context. [2] Could you please provide more information?
### KG
Abstain
### Claims
Abstain

### Question
does jimmy jazz sell fake shoes
### Response
[1] Based on the provided passages, it appears that Jimmy Jazz may sell fake shoes. [2] Passage 0 and Passage 1 both mention that the shoes purchased from Jimmy Jazz were found to be fake, and Passage 3 is a Ripoff Report stating that Jimmy Jazz E-Commerce LLC has a history of late shipping and lies on their website. [3] Additionally, Passage 4 mentions that Kanye West, who has collaborated with Jimmy Jazz in the past, has signed fake sneakers, which could suggest that Jimmy Jazz may be involved in selling counterfeit products. 
[4] However, it's important to note that there are also passages that do not mention anything about fake shoes, such as Passage 2, Passage 5, Passage 6, Passage 7, Passage 8, and Passage 9. [5] These passages do not provide any information that would confirm or deny the suspicion of Jimmy Jazz selling fake shoes. 
[6] Therefore, based on the provided passages, it is possible that Jimmy Jazz may sell fake shoes, but there is not enough conclusive evidence to confirm this suspicion.
### KG
(Passage 0, mention, shoes purchased from Jimmy Jazz were found to be fake) [2]
(Passage 1, mention, shoes purchased from Jimmy Jazz were found to be fake) [2]
(Passage 3, is, a Ripoff Report) [2]
(Passage 3, state, Jimmy Jazz E-Commerce LLC has a history of late shipping and lies on their website) [2]
(Passage 4, mention, Kanye West has collaborated with Jimmy Jazz in the past) [3]
(Passage 4, mention, Kanye West has signed fake sneakers) [3]
(Passage 2, does not mention, fake shoes) [4]
(Passage 5, does not mention, fake shoes) [4]
(Passage 6, does not mention, fake shoes) [4]
(Passage 7, does not mention, fake shoes) [4]
(Passage 8, does not mention, fake shoes) [4]
(Passage 9, does not mention, fake shoes) [4]
### Claims
Passage 0 and Passage 1 mention that shoes purchased from Jimmy Jazz were found to be fake. [2]
Passage 3 is a Ripoff Report stating that Jimmy Jazz E-Commerce LLC has a history of late shipping and lies on their website. [2]
Passage 4 mentions that Kanye West has collaborated with Jimmy Jazz in the past. [3]
Passage 4 mentions that Kanye West has signed fake sneakers. [3]
Passage 2, Passage 5, Passage 6, Passage 7, Passage 8, and Passage 9 do not mention anything about fake shoes. [4]

### Question
calories in ground beef stroganoff
### Response
[1] Sure! [2] Based on the provided passages, the answer to the question "calories in ground beef stroganoff?" [3] is 227 calories. [4] This information can be found in Passage 9.
### KG
(Passage 9, state, ground beef stroganoff contains 227 calories) [2][3][4]
### Claims
Passage 9 states that ground beef stroganoff contains 227 calories. [2][3][4]

Now generate the KG and claims for the following response based on the provided question. You should always follow the output format shown in the above examples and just ouput the KG and claims as well as the sentence IDs and DO NOT output any extra information:
### Question
{q}
### Response
{r}
### KG
"""




LLM_CLAIM_EXTRACTION_PROMPT_Q = """You are an AI assistant, you can help to extract claims from a model-generated response for a question. In addition, you should attribute the claims to the sentences followed by the sentence ids.
Each claim should satisfy the following criteria:
* A claim is a piece of `knowledge point` in the response.
* A claim should be fine-grained. One claim should not contain more than one pieces of knowledge.
* A claim should be self-contained, it is not dependent on other claims.
* Each claim should truly reflect the meaning to be expressed in the response, and the information in the claim should be complete and unambiguous, and necessary conditions and attributes should not be missed. For example, for the text "Donald Trump won the presidential election in 2016.", the claim "Donald Trump won the presidential election" is a bad claim where it misses necessary information "in 2016", so a complete claim should be "Donald Trump won the presidential election in 2016".
* Some sentence in the response may not contain claims. Some sentence may contain one or more claims. A claim may occur across multiple sentences.
* Opinions, speculations or questions in the text are not claims.

We have added sentence IDs at the beginning of the sentences in the text, you should output a claim followed by a list of IDs, and these IDs stand for the sentences this claim attribute to. The extraction process:
1. You should first identify whether there are claims in the text. If there are no claims, output "Abstain".
2. If there are claims in the text, you should first identify knowledge points in the text and formuate claims for them.
3. Note that a knowledge point may just reflect partial information in the text, so if we take it in isolation, it may not reflect the original meaning in the text, you should fill in the missing information to make it complete and clear. The information in a claim may come across multiple sentences in the text.
4. Formulate the claim into one sentence, and followed by the sentence indices as the attributed sentences in the text.
5. Make sure you have generated all the claims in the text.

Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the text is factual or not, just extract the claims from it.

Here are some examples:

### Question
What is Optimus?
### Response
[1] Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. [2] It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.
### Claims in Response
Optimus is a robotic humanoid. [1]
Optimus is under development by Tesla, Inc. [1]
Optimus is also known as Tesla Bot. [1]
Tesla, Inc. announced Optimus at the Artificial Intelligence (AI) Day event August 19, 2021. [2]

### Question
Here is some text about Andre Weiss, how many years was Andre at University of Dijon in Paris?
### Response
[1] 11 years
### Claims in Response
Andre Weiss was at University of Dijon in Paris for 11 years. [1]

### Question
who sang i ran all the way home?
### Response
[1] I'm sorry, I cannot provide an accurate answer without additional context. [2] Could you please provide more information?
### Claims in Response
Abstain

### Question
Does jimmy jazz sell fake shoes?
### Response
[1] Based on the provided passages, it appears that Jimmy Jazz may sell fake shoes. [2] Passage 0 and Passage 1 both mention that the shoes purchased from Jimmy Jazz were found to be fake, and Passage 3 is a Ripoff Report stating that Jimmy Jazz E-Commerce LLC has a history of late shipping and lies on their website. [3] Additionally, Passage 4 mentions that Kanye West, who has collaborated with Jimmy Jazz in the past, has signed fake sneakers, which could suggest that Jimmy Jazz may be involved in selling counterfeit products. 
[4] However, it's important to note that there are also passages that do not mention anything about fake shoes, such as Passage 2, Passage 5, Passage 6, Passage 7, Passage 8, and Passage 9. [5] These passages do not provide any information that would confirm or deny the suspicion of Jimmy Jazz selling fake shoes. 
[6] Therefore, based on the provided passages, it is possible that Jimmy Jazz may sell fake shoes, but there is not enough conclusive evidence to confirm this suspicion.
### Claims in Response
Passage 0 and Passage 1 mention that shoes purchased from Jimmy Jazz were found to be fake. [2]
Passage 3 is a Ripoff Report stating that Jimmy Jazz E-Commerce LLC has a history of late shipping and lies on their website. [2]
Passage 4 mentions that Kanye West has collaborated with Jimmy Jazz in the past. [3]
Passage 4 mentions that Kanye West has signed fake sneakers. [3]
Passage 2, Passage 5, Passage 6, Passage 7, Passage 8, and Passage 9 do not mention anything about fake shoes. [4]

### Question
calories in ground beef stroganoff
### Response
[1] Sure! [2] Based on the provided passages, the answer to the question "calories in ground beef stroganoff?" [3] is 227 calories. [4] This information can be found in Passage 9.
### Claims in Response
Passage 9 states that ground beef stroganoff contains 227 calories. [2][3][4]

Now please generate the claims from the following text. You should always follow the output format shown in the above examples and just ouput the claims without any extra information:
### Question
{q}
### Response
{r}
### Claims in Response
"""


LLM_TRIPLET_EXTRACTION_PROMPT_Q = \
"""Given a question and a candidate answer to the question, please extract a KG from the candidate answer condition on the question and represent the KG with triples formatted with ("subject", "predicate", "object"), each triplet in a line.
Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the candidate answer is factual or not, just extract the triplets from it.

Here are some in-context examples:

### Question:
Given these paragraphs about the Tesla bot, what is its alias?

### Candidate Answer:
Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.

### KG:
("Optimus", "is", "robotic humanoid")
("Optimus", "under development by", "Tesla, Inc.")
("Optimus", "also known as", "Tesla Bot")
("Tesla, Inc.", "announced", "Optimus")
("Announcement of Optimus", "occurred at", "Artificial Intelligence (AI) Day event")
("Artificial Intelligence (AI) Day event", "held on", "August 19, 2021")
("Artificial Intelligence (AI) Day event", "organized by", "Tesla, Inc.")

### Question:
here is some text about Andre Weiss, how many years was Andre at University of Dijon in Paris?

### Candidate Answer:
11 years

### KG:
("Andre Weiss at University of Dijon in Paris", "duration", "11 years")


Now generate the KG for the following candidate answer based on the provided question:

### Question:
{q}

### Candidate Answer:
{a}

### KG:
"""

LLM_TRIPLET_EXTRACTION_PROMPT = \
"""Given an input text, please extract a KG from the text and represent the KG with triples formatted with ("subject", "predicate", "object"), each triplet in a line. Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the candidate answer is factual or not, just extract the triplets from it.

Here are some in-context examples:

### Input:
Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.

### KG:
("Optimus", "is", "robotic humanoid")
("Optimus", "under development by", "Tesla, Inc.")
("Optimus", "also known as", "Tesla Bot")
("Tesla, Inc.", "announced", "Optimus")
("Announcement of Optimus", "occurred at", "Artificial Intelligence (AI) Day event")
("Artificial Intelligence (AI) Day event", "held on", "August 19, 2021")
("Artificial Intelligence (AI) Day event", "organized by", "Tesla, Inc.")

### Input:
The song "Here Comes the Boom" was originally released by American rock band Nelly in 2002 for the soundtrack of the film "The Longest Yard."

### KG:
("The song 'Here Comes the Boom'", "originally released by", "American rock band Nelly")
("The song 'Here Comes the Boom'", "released in", "2002")
("The song 'Here Comes the Boom'", "featured in", "soundtrack of the film 'The Longest Yard'")
("American rock band Nelly", "released", "The song 'Here Comes the Boom'")
("The Longest Yard", "had soundtrack featuring", "The song 'Here Comes the Boom'")


Now generate the KG for the provided input text:

### Input:
{input_text}

### KG:
"""