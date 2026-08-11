"""Prompts used by the disclosure-variant generation pipeline."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Exact paper system prompt. Instructional ``{field}`` names are literal;
# only the Inputs block is filled from the scenario item.
_GENERATION_PROMPT_TEMPLATE = """\
You are a communication expert who knows how to communicate sensitive information in various ways.
Your task is to generate 9 variants that cover all combinations of granularity x identifiability (as defined below) based on the input provided.

Definition of Granularity: 
- General: The response is a high-level abstraction of the information without mentioning fine details about the action, processes, or context.
- Moderately detailed: The response elaborates some details about the information, but is still abstract and not exhaustive.
- Very detailed: The response covers the comprehensive and fine-grained details of the information.
Definition of Identifiability:
- Not identifiable: The response anonymizes or omits all personal identifiers of the data subject that could be used to directly or indirectly trace back to them. The data subject is completely not identifiable.
- Partially identifiable: The response contains attributes that cannot directly identify the individual but could be combined with other attributes or publicly available information to trace back to them.
- Fully identifiable: The response contains attributes that can directly identify the data subject - such as their name, role, or other specific identifiers. (Note that disclosing the first name or last name of the data subject is considered fully identifiable.)

The data sharing practice is specified with {{third_narrative}} and other attributes.
The response should be a one-sentence, first-person narrative that is sent by {{data_sender}} through the {{transmission_principle}} to {{data_recipient}} in order to complete {{data_sender}}'s sharing action.
The response should disclose the {{sensitive_information_item}} of {{data_subject}} at the specified granularity level and the specified identifiability level. If it's notes, share the detailed content of the notes instead of saying "I shared a summary/notes".
    
Please strictly follow the four-phase pipeline, output only the variants in the specified format with a title "## Output Variants".

---
## Inputs  
Data Sender: {data_sender}
Data Subject: {data_subject}
Data Recipient: {data_recipient}
Transmission Principle: {transmission_principle}
Story Plot: {third_narrative}
Sensitive Information Item: {sensitive_information_item}  
Details of Sensitive Information Item: {description_of_sensitive_information_item} 
---
## Output Format  
- [general, not identifiable] {{variant1}}  
- [general, partially identifiable] {{variant5}}  
- [general, fully identifiable] {{variant4}}  
- [moderately detailed, not identifiable] {{variant8}}  
- [moderately detailed, partially identifiable] {{variant9}}  
- [moderately detailed, fully identifiable] {{variant7}}  
- [very detailed, not identifiable] {{variant3}}  
- [very detailed, partially identifiable] {{variant6}}  
- [very detailed, fully identifiable] {{variant2}}  
**Do not output anything else.**
---
## Generation Process  
### Important Notes:
- Keep the response as short, concise, and to the point as possible. Only include the necessary information about the {{sensitive_information_item}}.
- All details should be based strictly on the {{description_of_sensitive_information_item}}. Do not invent or add information that is not mentioned in the description.
- The tone and language should be contextually appropriate. Avoid awkward or unnatural phrasing.
- The response should maintain a consistent language style and word use across variants, varying **only** in identifiability and granularity.
- If the data sender is the data subject, then in the "not identifiable" variants, avoid using "I" or "me" to refer to the data subject. Instead, take a non-identifying third-person perspective to describe the data subject and the information.
- Don't explicitly mention the {{transmission principle}} in the response. Instead, think of the variant as a part of the message shared through the {{transmission_principle}} and make it natural.
- For fully identifiable variants with direct identifiers like names, if the identity of the data subject is not familiar to the data recipient, then also mention the data subject's identity in the response.
- Avoid starting a sentence with "I'm sharing that" or "I'm letting you know". Be more natural.

### Step 1: Extreme Variants  
1. Procedure:  
- Identify the variant1 "general + not identifiable" response. This variant has the minimal relevant disclosure of the data subject's information.
- Identify the variant2 "very detailed + fully identifiable" response. This variant has the most identifiable and comprehensive disclosure of the data subject's information.
- Extract core event elements (who, what, when, where).  
2. Generate:  
- variant1 "general + not identifiable": ...
- variant2 "very detailed + fully identifiable": ...

### Step 2: Cross-Diagonal Variants  
1. Procedure:
- Based on the two extreme variants above, variant1 and variant2, keep all other irrelevant elements consistent to generate the cross-diagonal variants variant3 "very detailed + not identifiable" and variant4 "general + fully identifiable".
- variant3 "very detailed + not identifiable" should have the same granularity level as variant2 and same identifiability level as variant1
- variant4 "general + fully identifiable" should have the same granularity level as variant1 and the same identifiability level as variant2.
2. Generate:  
- variant3 "very detailed + not identifiable": ...
- variant4 "general + fully identifiable": ...

### Step 3: Row & Column Midpoints  
1. Procedure:
- Fix the "general". Use variant1 and variant4 to generate variant5 "general + partially identifiable" response. Ensure the response has the same granularity level as "general", and the identifiability lies between the given two responses.
- Fix the "very detailed". Use variant2 and variant3 to generate a variant6 "very detailed + partially identifiable" response. Make sure the response has the same granularity level as the "very detailed", and the identifiability lies between the given two responses.
- Fix the "fully identifiable". Use variant2 and variant4 to generate a variant7 "moderately detailed + fully identifiable" response. Make sure the response has the same identifiability level as the "fully identifiable", and the granularity lies between the given two responses.
- Fix the "not identifiable". Use variant1 and variant3 to generate a variant8 "moderately detailed + not identifiable" response. Make sure the response has the same identifiability level as the "not identifiable", and the granularity lies between the given two responses.
2. Generate:  
- variant5 "general + partially identifiable":  ...
- variant6 "very detailed + partially identifiable": ...
- variant7 "moderately detailed + fully identifiable": ...
- variant8 "moderately detailed + not identifiable": ...

### Step 4: Center Variant  
1. Procedure:
- Based on the four midpoint variants above variant5, variant6, variant7, and variant8, keep all core elements consistent to generate variant9 "moderately detailed + partially identifiable". 
2. Generate:  
- variant9 "moderately detailed + partially identifiable": ...

---
Start with **Step 1**, write out your reasoning first, then output the corresponding text. After completing all steps, present the 9 variants in the "Output Format" exactly as specified.
"""


def build_generation_prompt(item: Mapping[str, Any]) -> str:
    """Build the four-phase prompt for one normalized scenario item."""
    return _GENERATION_PROMPT_TEMPLATE.format(
        data_sender=item["data_sender"],
        data_subject=item["data_subject"],
        data_recipient=item["data_recipient"],
        transmission_principle=item["transmission_principle"],
        third_narrative=item["story"],
        sensitive_information_item=item["data_type"],
        description_of_sensitive_information_item=item["sensitive_info_items"],
    )
