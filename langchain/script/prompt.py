RAG_TEMPLATE = """
You are an expert assistant. Use the following context to answer the question as accurately as possible:
Context: {context}
Question: {question}
Answer only if the context contains relevant information.
"""
### Industry-Specific Menu Page Content Template
WEB_MENU_TEMPLATE = '''
    1. **Industry Selection**
    - [List of industries for the user to choose from, e.g., Finance, Healthcare, Education, IT, Manufacturing, etc.]

    2. **Industry Overview**
    - Provide an overview of the selected industry. [Include: definition of the industry, key characteristics, current market trends, etc.]

    3. **Key Services and Products**
    - List the main services or products offered in the industry. [Include brief descriptions of each service or product.]

    4. **Target Audience**
    - Describe the main target audience for the industry. [Include: B2B, B2C, age groups, regions, etc.]

    5. **Competitive Analysis**
    - Analyze the main competitors in the industry. [Include their strengths and weaknesses.]

    6. **Future Outlook**
    - Explain the future outlook of the industry. [Include: technological advancements, market growth predictions, major challenges, etc.]

    ### User-Specific Answer Generation Template

    1. **Question:** "What are the major trends in this industry?"
    - **Context:** Explain the current major trends in the selected industry. [Example for the finance industry: the rise of digital banking, growth of fintech companies, etc.]

    2. **Question:** "What services can we offer in this industry?"
    - **Context:** Describe the main services or products that can be offered in the selected industry. [Example for the IT industry: cloud services, data analytics, cybersecurity, etc.]

    3. **Question:** "Who are our competitors and what are their strengths and weaknesses?"
    - **Context:** Identify the main competitors in the selected industry and analyze their strengths and weaknesses.

    4. **Question:** "What is the future outlook for this industry?"
    - **Context:** Explain the future outlook and major challenges for the selected industry. [Example for the healthcare industry: advancements in AI for diagnostics, expansion of telemedicine services, etc.]
    '''