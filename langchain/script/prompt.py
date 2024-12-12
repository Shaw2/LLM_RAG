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
    
# TITLE_STRUCTURE = """- You can use this example to write down title.
#                     [Title]
#                     The example of title
#                     """

# KEYWORDS_STRUCTURE = """- You can use this example to write down keywords.
#                     [Keywords]
#                     1. A
#                     2. B
#                     3. C
#                     """

# MENU_STRUCTURE = """- You can use this kind of structure when build two_depth menu. start with 'number' is first_depth menu. start with '-' is second_depth menu.
#                     [Two_depth Menu]
#                     1. Home, 
#                     2. Company Introduction, 
#                             - Company History,    
#                             - Company Vision,  
#                             - CEO Message,                     
#                     3. Business Overview,    
#                             - Business Areas, 
#                             - Business Achievements, 
#                             - Future Goals,  
#                     4. Contact Us, 
#                             - Location,    
#                             - Phone,    
#                             - FAQs, 
#                             - Team members"""
# CONTENT_STRUCTURE = """ - You can use this when make contents of menues
#                     1. Main
#                     Content: A brief introduction to the company's vision and mission.

#                     2. About Us
#                         2-1. CEO's Message
#                             Content: Basic company information, goals, and vision.
#                         2-2. Organization Chart
#                             Content: Introduction to the company's organizational structure and key personnel.
#                         2-3. History
#                             Content: The company's growth process over the years.
                    
#                     3. Services
#                         3-1. Business Areas
#                             Content: The company's business items and areas of operation.
#                         3-2. Business Achievements
#                             Content: The company's business accomplishments.
#                         3-3. Product Introduction
#                             Content: Description of the company's products and services.
                    
#                     4. Customer Service
#                         4-1. Notices
#                             Content: The company's latest news and events.
#                         4-2. Customer Inquiries
#                             Content: Contact information, email, social media links, etc.
#                         4-3. FAQ
#                             Content: Frequently asked questions and answers. """
                            
TITLE_STRUCTURE = """- 제목 작성에 이 예제를 사용할 수 있습니다.
                    [제목]
                    제목의 예시
                    """

KEYWORDS_STRUCTURE = """- 키워드 작성에 이 예제를 사용할 수 있습니다.
                    [키워드]
                    1. A
                    2. B
                    3. C
                    """

MENU_STRUCTURE = """- 이중 메뉴를 작성할 때 이 구조를 사용할 수 있습니다. '숫자'로 시작하는 것은 first_depth 메뉴, '-'로 시작하는 것은 second_depth 메뉴입니다.
                    [이중 메뉴]
                    1. 홈, 
                    2. 회사 소개, 
                            - 회사 역사,    
                            - 회사 비전,  
                            - CEO 메시지,                     
                    3. 사업 개요,    
                            - 사업 영역, 
                            - 사업 실적, 
                            - 미래 목표,  
                    4. 문의하기, 
                            - 위치,    
                            - 전화,    
                            - 자주 묻는 질문, 
                            - 팀 멤버"""

CONTENT_STRUCTURE = """ 메뉴 구조의 데이터를 key로 하여, 각 메뉴에 적합한 내용들을 value로 생성해야합니다. 
                    {"홈" : "홈에 적합한 내용 },
                    {"회사소개" : "회사소개에 적합한 내용 },
                    {"회사역사" : "회사역사에 적합한 내용 },
                    {"회사 비전" : "회사 비전에 적합한 내용 },
                    {"CEO 메시지" : "CEO 메시지에 적합한 내용 },
                    {"사업 개요" : "사업 개요에 적합한 내용 },
                    {"사업 영역" : "사업 영역에 적합한 내용 },
                    {"사업 실적" : "사업 실적에 적합한 내용 },
                    {"미래 목표" : "미래 목표에 적합한 내용 },
                    {"문의하기" : "문의하기에 적합한 내용 },
                    {"위치" : "위치에 적합한 내용 },
                    {"전화" : "전화에 적합한 내용 },
                    {"자주 묻는 질문" : "자주 묻는 질문에 적합한 내용 },
                    {"팀 멤버" : "팀 멤버에 적합한 내용 }"""