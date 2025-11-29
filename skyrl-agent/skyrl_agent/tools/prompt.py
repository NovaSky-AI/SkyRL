EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
4. **Word limits**: "rational" ≤ 50 words; "evidence" ≤ 70 words; "summary" ≤ 90 words.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

SIMPLE_MEMORY_SYSTEM_PROMPT = """You are presented with a problem and sections of articles that may contain the answer. 

Your task is to:
1. Read each section carefully
2. Identify information relevant to answering the problem
3. Update your memory to combine previous findings with new information from the current section
4. Keep all relevant details while maintaining a concise, organized memory
5. Once you have enough information, provide your final answer

The format you'll work with:
- <problem>: The question you need to answer
- <memory>: Your cumulative findings from previous sections (empty at start)
- <section>: The current section of text to analyze

Use the 'next_with_summary' tool to update your memory and move to the next section. When you have sufficient information to answer the problem, use the 'finish' tool with your answer."""
