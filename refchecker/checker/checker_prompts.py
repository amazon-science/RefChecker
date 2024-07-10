JOINT_CHECKING_PROMPT_Q = """I have a list of claims that made by a language model to a question, please help me for checking whether the claims can be entailed according to the provided reference which is related to the question. 
The reference is a list of passages, and each of the claims is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Question:
[QUESTION]

### Reference:
[REFERENCE]

### Claims:
[CLAIMS]


Your answer should always be only a list of labels, each of the labels is a single word in ['Entailment', 'Neutral', 'Contradiction'], for example, you should output a list like follows:

Entailment
Neutral
Contradiction
Neutral


DO NOT add explanations or you own reasoning to the output, only output the label list.
"""









# JOINT_CHECKING_PROMPT_Q = """# Hallucination Detection

# You are tasked with evaluating claims extracted from a language model's answer to determine their accuracy. Your primary source of information should be the given reference, but you should also use your computational abilities and commonsense knowledge when necessary, especially for tasks like unit conversion or basic logical inferences.

# ## Instructions:
# 1. Carefully read the question, reference, and claims.
# 2. For each claim, compare it to the information in the reference.
# 3. Use your computational abilities and commonsense knowledge when needed, particularly for:
#    - Unit conversions (e.g., miles to kilometers)
#    - Basic mathematical calculations
#    - Logical inferences based on given information
# 5. Assign one of the following labels:
#    - Entailment: The reference (or reference combined with necessary calculations) provides enough information to conclude that the claim is true.
#    - Contradiction: The reference (or reference combined with necessary calculations) provides enough information to conclude that the claim is false.
#    - Neutral: The reference doesn't provide enough information to determine whether the claim is true or false, or the claim contains elements that can't be verified using only the given reference.
# 6. Format your response using the specified output format below.

# ## Input Format:
# The claims are presented in the following format:

# ---
# Claim ID: [number starting from 0]
# Claim: [claim text]
# ---

# ## Output Format:
# Use the following format for each claim evaluation:

# ---
# Claim ID: [number starting from 0]
# Label: <Entailment or Contradiction or Neutral>
# ---

# Repeat this structure for each claim, separating them with blank lines. Ensure that you evaluate all claims in the order they are presented.

# ## Important Notes:
# - Prioritize information from the reference, but apply computations and common sense when needed.
# - Maintain objectivity in your assessments.
# - Ensure you use the exact format specified.
# - Evaluate all claims in the order they are presented.
# - Do not repeat the claim text in your output.
# - Claim IDs start from 0 and increase sequentially.

# ## Example:

# ### Question
# What's the longest river in the world?

# ### Reference
# The Nile is a major north-flowing river in northeastern Africa. It flows into the Mediterranean Sea. The Nile is the longest river in Africa and has historically been considered the longest river in the world, though this has been contested by research suggesting that the Amazon River is slightly longer. Of the world's major rivers, the Nile is one of the smallest, as measured by annual flow in cubic metres of water. About 6,650 km (4,130 mi) long, its drainage basin covers eleven countries: the Democratic Republic of the Congo, Tanzania, Burundi, Rwanda, Uganda, Kenya, Ethiopia, Eritrea, South Sudan, Sudan, and Egypt. In particular, the Nile is the primary water source of Egypt, Sudan and South Sudan. Additionally, the Nile is an important economic river, supporting agriculture and fishing.

# ### Claims

# ---
# Claim ID: 0
# Claim: The longest river in the world is the Nile River
# ---

# ---
# Claim ID: 1
# Claim: The Nile River is located in northeastern Africa
# ---

# ---
# Claim ID: 2
# Claim: The Nile River stretches for approximately 4,132 km
# ---

# ---
# Claim ID: 3
# Claim: The Nile River has sources in Burundi
# ---

# ---
# Claim ID: 4
# Claim: The Nile River flows through 11 countries
# ---

# ---
# Claim ID: 5
# Claim: The Nile River is the largest river in the world by water volume
# ---

# ---
# Claim ID: 6
# Claim: The Nile River flows southward into the Atlantic Ocean
# ---

# ### Output

# ---
# Claim ID: 0
# Label: Neutral
# ---

# ---
# Claim ID: 1
# Label: Entailment
# ---

# ---
# Claim ID: 2
# Label: Entailment
# ---

# ---
# Claim ID: 3
# Label: Neutral
# ---

# ---
# Claim ID: 4
# Label: Entailment
# ---

# ---
# Claim ID: 5
# Label: Contradiction
# ---

# ---
# Claim ID: 6
# Label: Contradiction
# ---

# ## Your Task:

# ### Question
# [QUESTION]

# ### Reference
# [REFERENCE]

# ### Claims
# [CLAIMS]

# ### Output
# [Your response here, using the format specified above]
# """




# JOINT_CHECKING_PROMPT_Q = """# Hallucination Detection

# You are tasked with evaluating claims extracted from a language model's answer to determine their accuracy. Your primary source of information should be the given reference, but you should also use your computational abilities and commonsense knowledge when necessary, especially for tasks like unit conversion or basic logical inferences.

# ## Instructions:
# 1. Carefully read the question, reference, and claims.
# 2. For each claim, compare it to the information in the reference.
# 3. Use your computational abilities and commonsense knowledge when needed, particularly for:
#    - Unit conversions (e.g., miles to kilometers)
#    - Basic mathematical calculations
#    - Logical inferences based on given information
# 5. Assign one of the following labels:
#    - Entailment: The reference (or reference combined with necessary calculations) provides enough information to conclude that the claim is true; or even not explictly stated, the reference can imply the claim.
#    - Contradiction: The reference (or reference combined with necessary calculations) provides enough information to conclude that the claim is false.
#    - Neutral: The reference doesn't provide enough information to determine whether the claim is true or false, or the claim contains elements that can't be verified using only the given reference.
# 6. Format your response using the specified output format below.

# ## Input Format:
# The claims are presented in the following format:

# ---
# Claim ID: [number starting from 0]
# Claim: [claim text]
# ---

# ## Output Format:
# Use the following format for each claim evaluation:

# ---
# Claim ID: [number starting from 0]
# Label: <Entailment or Contradiction or Neutral>
# ---

# Repeat this structure for each claim, separating them with blank lines. Ensure that you evaluate all claims in the order they are presented.

# ## Example:

# ### Question
# What's the longest river in the world?

# ### Reference
# The Nile is a major north-flowing river in northeastern Africa. It flows into the Mediterranean Sea. The Nile is the longest river in Africa and has historically been considered the longest river in the world, though this has been contested by research suggesting that the Amazon River is slightly longer. Of the world's major rivers, the Nile is one of the smallest, as measured by annual flow in cubic metres of water. About 6,650 km (4,130 mi) long, its drainage basin covers eleven countries: the Democratic Republic of the Congo, Tanzania, Burundi, Rwanda, Uganda, Kenya, Ethiopia, Eritrea, South Sudan, Sudan, and Egypt. In particular, the Nile is the primary water source of Egypt, Sudan and South Sudan. Additionally, the Nile is an important economic river, supporting agriculture and fishing.

# ### Claims
# ---
# Claim ID: 0
# Claim: The Nile River is located in northeastern Africa
# ---

# ---
# Claim ID: 1
# Claim: The Nile River has sources in Burundi
# ---

# ---
# Claim ID: 2
# Claim: The Nile River is the largest river in the world by water volume
# ---

# ### Output
# ---
# Claim ID: 0
# Label: Entailment
# ---

# ---
# Claim ID: 1
# Label: Neutral
# ---

# ---
# Claim ID: 2
# Label: Contradiction
# ---


# ## Your Task:

# ### Question
# [QUESTION]

# ### Reference
# [REFERENCE]

# ### Claims
# [CLAIMS]

# ### Output
# [Your response here, using the format specified above]
# """


# JOINT_CHECKING_PROMPT_Q = \
# """I have a list of claims that made by a language model to a question, please help me for checking whether the claims can be entailed according to the provided reference which is related to the question. 
# The reference is a list of passages, and each of the claims is represented as a triplet formatted with ("subject", "predicate", "object").

# If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
# If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
# If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

# Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

# ### Question:
# [QUESTION]

# ### Reference:
# [REFERENCE]

# ### Claims:
# [CLAIMS]

# Your answer should always be only a list of labels, each of the labels is a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
# """


JOINT_CHECKING_PROMPT_Q_WITH_RATIONALE = """# Hallucination Detection

You are tasked with evaluating claims extracted from a language model's answer to determine their accuracy. Your primary source of information should be the given reference, but you should also use your computational abilities and commonsense knowledge when necessary, especially for tasks like unit conversion or basic logical inferences.

## Instructions:
1. Carefully read the question, reference, and claims.
2. For each claim, compare it to the information in the reference.
3. Use your computational abilities and commonsense knowledge when needed, particularly for:
   - Unit conversions (e.g., miles to kilometers)
   - Basic mathematical calculations
   - Logical inferences based on given information
4. Provide a detailed rationale for your judgment, including any calculations or conversions you perform.
5. Based on your rationale, assign one of the following labels:
   - Entailment: The reference (or reference combined with necessary calculations) provides enough information to conclude that the claim is true; or even not explictly stated, the reference can imply the claim.
   - Contradiction: The reference (or reference combined with necessary calculations) provides enough information to conclude that the claim is false.
   - Neutral: The reference doesn't provide enough information to determine whether the claim is true or false, or the claim contains elements that can't be verified using only the given reference.
6. Format your response using the specified output format below.

## Input Format:
The claims are presented in the following format:

---
Claim ID: [number starting from 0]
Claim: [claim text]
---

## Output Format:
Use the following format for each claim evaluation:

---
Claim ID: [number starting from 0]
Rationale: <Your detailed explanation for the judgment, including relevant quotes from the reference, any calculations performed, and reasoning applied>
Label: <Entailment or Contradiction or Neutral>
---

Repeat this structure for each claim, separating them with blank lines. Ensure that you evaluate all claims in the order they are presented.

## Example:

### Question
What's the longest river in the world?

### Reference
The Nile is a major north-flowing river in northeastern Africa. It flows into the Mediterranean Sea. The Nile is the longest river in Africa and has historically been considered the longest river in the world, though this has been contested by research suggesting that the Amazon River is slightly longer. Of the world's major rivers, the Nile is one of the smallest, as measured by annual flow in cubic metres of water. About 6,650 km (4,130 mi) long, its drainage basin covers eleven countries: the Democratic Republic of the Congo, Tanzania, Burundi, Rwanda, Uganda, Kenya, Ethiopia, Eritrea, South Sudan, Sudan, and Egypt. In particular, the Nile is the primary water source of Egypt, Sudan and South Sudan. Additionally, the Nile is an important economic river, supporting agriculture and fishing.

### Claims
---
Claim ID: 0
Claim: The Nile River is located in northeastern Africa
---

---
Claim ID: 1
Claim: The Nile River has sources in Burundi
---

---
Claim ID: 2
Claim: The Nile River is the largest river in the world by water volume
---

### Output
---
Claim ID: 0
Rationale: The reference explicitly states, "The Nile is a major north-flowing river in northeastern Africa." This directly supports the claim that the Nile River is located in northeastern Africa.
Label: Entailment
---

---
Claim ID: 1
Rationale: The reference mentions that the Nile's drainage basin covers eleven countries, including Burundi: "its drainage basin covers eleven countries: the Democratic Republic of the Congo, Tanzania, Burundi, Rwanda, Uganda, Kenya, Ethiopia, Eritrea, South Sudan, Sudan, and Egypt." While this implies that Burundi contributes to the Nile's water supply, it does not explicitly state that the Nile has its sources in Burundi. The claim is more specific than what the reference provides, so we cannot definitively confirm or deny it based solely on this information.
Label: Neutral
---

---
Claim ID: 2
Rationale: The reference contradicts this claim. It states, "Of the world's major rivers, the Nile is one of the smallest, as measured by annual flow in cubic metres of water." This directly opposes the claim that the Nile is the largest river in the world by water volume. In fact, it suggests that the Nile is among the smallest major rivers in terms of water volume.
Label: Contradiction
---


## Your Task:

### Question
[QUESTION]

### Reference
[REFERENCE]

### Claims
[CLAIMS]

### Output
[Your response here, using the format specified above]
"""



# JOINT_CHECKING_PROMPT_Q_WITH_RATIONALE = """# Hallucination Detection

# You are tasked with evaluating claims extracted from a language model's answer to determine their accuracy. Your primary source of information should be the given reference, but you should also use your computational abilities and commonsense knowledge when necessary, especially for tasks like unit conversion or basic logical inferences.

# ## Instructions:
# 1. Carefully read the question, reference, and claims.
# 2. For each claim, compare it to the information in the reference.
# 3. Use your computational abilities and commonsense knowledge when needed, particularly for:
#    - Unit conversions (e.g., miles to kilometers)
#    - Basic mathematical calculations
#    - Logical inferences based on given information
# 4. Provide a detailed rationale for your judgment, including any calculations or conversions you perform.
# 5. Based on your rationale, assign one of the following labels:
#    - Entailment: The reference (or reference combined with necessary calculations) provides enough information to conclude that the claim is true.
#    - Contradiction: The reference (or reference combined with necessary calculations) provides enough information to conclude that the claim is false.
#    - Neutral: The reference doesn't provide enough information to determine whether the claim is true or false, or the claim contains elements that can't be verified using only the given reference.
# 6. Format your response using the specified output format below.

# ## Input Format:
# The claims are presented in the following format:

# ---
# Claim ID: [number starting from 0]
# Claim: [claim text]
# ---

# ## Output Format:
# Use the following format for each claim evaluation:

# ---
# Claim ID: [number starting from 0]
# Rationale: <Your detailed explanation for the judgment, including relevant quotes from the reference, any calculations performed, and reasoning applied>
# Label: <Entailment or Contradiction or Neutral>
# ---

# Repeat this structure for each claim, separating them with blank lines. Ensure that you evaluate all claims in the order they are presented.

# ## Important Notes:
# - Prioritize information from the reference, but apply computations and common sense when needed.
# - Clearly show your work for any calculations or conversions you perform.
# - Be precise and thorough in your rationale.
# - Maintain objectivity in your assessments.
# - Ensure you use the exact format specified, including the dashes and blank lines.
# - Always provide your rationale before stating the label.
# - Evaluate all claims in the order they are presented.
# - Do not repeat the claim text in your output.
# - Claim IDs start from 0 and increase sequentially.

# ## Example:

# ### Question
# What's the longest river in the world?

# ### Reference
# The Nile is a major north-flowing river in northeastern Africa. It flows into the Mediterranean Sea. The Nile is the longest river in Africa and has historically been considered the longest river in the world, though this has been contested by research suggesting that the Amazon River is slightly longer. Of the world's major rivers, the Nile is one of the smallest, as measured by annual flow in cubic metres of water. About 6,650 km (4,130 mi) long, its drainage basin covers eleven countries: the Democratic Republic of the Congo, Tanzania, Burundi, Rwanda, Uganda, Kenya, Ethiopia, Eritrea, South Sudan, Sudan, and Egypt. In particular, the Nile is the primary water source of Egypt, Sudan and South Sudan. Additionally, the Nile is an important economic river, supporting agriculture and fishing.

# ### Claims

# ---
# Claim ID: 0
# Claim: The longest river in the world is the Nile River
# ---

# ---
# Claim ID: 1
# Claim: The Nile River is located in northeastern Africa
# ---

# ---
# Claim ID: 2
# Claim: The Nile River stretches for approximately 4,132 miles
# ---

# ---
# Claim ID: 3
# Claim: The Nile River has sources in Burundi
# ---

# ---
# Claim ID: 4
# Claim: The Nile River flows through 11 countries
# ---

# ---
# Claim ID: 5
# Claim: The Nile River is the largest river in the world by water volume
# ---

# ---
# Claim ID: 6
# Claim: The Nile River flows southward into the Atlantic Ocean
# ---

# ### Output

# ---
# Claim ID: 0
# Rationale: The reference provides conflicting information on this claim. It states that the Nile "has historically been considered the longest river in the world", but also mentions that "this has been contested by research suggesting that the Amazon River is slightly longer". The reference does not provide a definitive answer, presenting both the historical view and recent contestation. Given this uncertainty, we cannot definitively confirm or deny the claim.
# Label: Neutral
# ---

# ---
# Claim ID: 1
# Rationale: The reference directly supports this claim with the statement: "The Nile is a major north-flowing river in northeastern Africa." This is an explicit confirmation of the river's location.
# Label: Entailment
# ---

# ---
# Claim ID: 2
# Rationale: The reference states that the Nile is "About 6,650 km (4,130 mi) long". The claim asserts that the Nile River stretches for approximately 4,132 miles. To compare these, we need to consider the precision of the measurements:
# 1. Reference: 4,130 miles
# 2. Claim: 4,132 miles
# The difference is only 2 miles, which is negligible considering the "About" qualifier in the reference and the "approximately" in the claim. Given the scale of the river and the inherent uncertainties in measuring such long distances, these two measurements can be considered equivalent for practical purposes.
# Label: Entailment
# ---

# ---
# Claim ID: 3
# Rationale: The reference does not explicitly mention the sources of the Nile. It states that the Nile's "drainage basin covers eleven countries" and lists Burundi among these countries. However, being part of the drainage basin does not necessarily mean that Burundi contains the river's sources. The reference does not provide enough information to confirm or deny this specific claim about the Nile's sources.
# Label: Neutral
# ---

# ---
# Claim ID: 4
# Rationale: The reference directly supports this claim by stating: "its drainage basin covers eleven countries". The claim that "The Nile River flows through 11 countries" is essentially equivalent to the reference's statement about the drainage basin covering 11 countries, as a river's drainage basin typically includes the countries through which it flows.
# Label: Entailment
# ---

# ---
# Claim ID: 5
# Rationale: This claim directly contradicts the information provided in the reference. The reference explicitly states: "Of the world's major rivers, the Nile is one of the smallest, as measured by annual flow in cubic metres of water." This statement clearly indicates that the Nile is not the largest river in the world by water volume, but rather one of the smallest among major rivers in terms of water flow.
# Label: Contradiction
# ---

# ---
# Claim ID: 6
# Rationale: This claim contradicts the information given in the reference on two points. First, the reference clearly states that the Nile is a "north-flowing river", not a southward-flowing one. Second, the reference specifies that the Nile "flows into the Mediterranean Sea", not the Atlantic Ocean. Both the direction of flow and the river's outlet are incorrectly stated in the claim, directly contradicting the reference.
# Label: Contradiction
# ---

# ## Your Task:

# ### Question
# [QUESTION]

# ### Reference
# [REFERENCE]

# ### Claims
# [CLAIMS]

# ### Output
# [Your response here, using the format specified above]
# """


LLM_CHECKING_PROMPT_Q = \
"""I have a claim that made by a language model to a question, please help me for checking whether the claim can be entailed according to the provided reference which is related to the question. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Question:
{question}

### Reference:
{reference}

### Claim:
{claim}

Your answer should always be only a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
"""

LLM_CHECKING_PROMPT = \
"""I have a claim that made by a language model, please help me for checking whether the claim can be entailed according to the provided reference. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Reference:
{reference}

### Claim:
{claim}

Your answer should always be only a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
"""


SUBSENTENCE_CLAIM_CHECKING_PROMPT = \
"""I have a claim that made by a language model, please help me for checking whether the claim can be entailed according to the provided reference. 
The reference is a list of passages, and the claim is a sentence.

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Reference:
{reference}

### Claim:
{claim}

Your answer should always be only a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
"""