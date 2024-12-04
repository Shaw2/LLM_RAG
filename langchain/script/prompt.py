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
    
TITLE_STRUCFTURE = """- You can use this example to write down title.
                    [Title]
                    The example of title
                    """

KEYWORDS_STRUCTRUE = """- You can use this example to write down keywords.
                    [Keywords]
                    1. A
                    2. B
                    3. C
                    """

MENU_STRUCTURE = """- You can use this kind of structure when build two_depth menu. start with 'number' is first_depth menu. start with '-' is second_depth menu.
                    [Two_depth Menu]
                    1. Home
                    2. Company Introduction   
                            - Company History   
                            - Company Vision   
                            - CEO Message                    
                    3. Business Overview   
                            - Business Areas
                            - Business Achievements
                            - Future Goals 
                    4. Contact Us
                            - Location   
                            - Phone   
                            - FAQs
                            - Team members"""

CONTENT_STRUCTURE = """ - You can use this when make contents of menues
                    1. Main
                    Content: A brief introduction to the company's vision and mission.

                    2. About Us
                        2-1. CEO's Message
                            Content: Basic company information, goals, and vision.
                        2-2. Organization Chart
                            Content: Introduction to the company's organizational structure and key personnel.
                        2-3. History
                            Content: The company's growth process over the years.
                    
                    3. Services
                        3-1. Business Areas
                            Content: The company's business items and areas of operation.
                        3-2. Business Achievements
                            Content: The company's business accomplishments.
                        3-3. Product Introduction
                            Content: Description of the company's products and services.
                    
                    4. Customer Service
                        4-1. Notices
                            Content: The company's latest news and events.
                        4-2. Customer Inquiries
                            Content: Contact information, email, social media links, etc.
                        4-3. FAQ
                            Content: Frequently asked questions and answers. """