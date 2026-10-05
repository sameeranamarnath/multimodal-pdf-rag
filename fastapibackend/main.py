import os  
import requests  
from fastapi import FastAPI, HTTPException  
from fastapi.middleware.cors import CORSMiddleware  
from langchain.chat_models import AzureChatOpenAI  
from langchain.prompts import ChatPromptTemplate  
from pydantic import BaseModel  
from typing import List  
import smtplib  
from email.mime.multipart import MIMEMultipart  
from email.mime.text import MIMEText  
  
# Initialize FastAPI  
app = FastAPI() 

app = FastAPI()  
  
# Configure CORS middleware  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins=["http://localhost:3000"],  # React frontend  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
) 

# Manually parsed the .env file and populate os.environ to get the Lever API key
if os.path.exists(".env"):
    with open(".env", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", maxsplit=1)
                os.environ[key] = value.strip('"')


# Environment variables (set these securely in your environment)  
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  
LEVER_API_KEY = os.getenv("LEVER_API_KEY")  
GMAIL_SMTP_SERVER = os.getenv("GMAIL_SMTP_SERVER")  
GMAIL_SMTP_PORT = int(os.getenv("GMAIL_SMTP_PORT", "587"))  
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")  
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")  
print(AZURE_OPENAI_DEPLOYMENT_NAME)

auth = (LEVER_API_KEY, '')



# Initialize Azure OpenAI LLM via LangChain  
llm = AzureChatOpenAI(  
    azure_endpoint=AZURE_OPENAI_ENDPOINT,  
    openai_api_version="2024-12-01-preview",  
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,  
    openai_api_key=AZURE_OPENAI_API_KEY,  
    temperature=0.2,  
)  
  
# Structured prompt template as previously discussed  
prompt_template = ChatPromptTemplate.from_template("""  
You are a professional Talent Acquisition AI assistant tasked with matching candidate profiles to relevant job postings.  
  
### Job Posting:  
Title: {job_title}  
Location: {job_location}  
Description: {job_description}  
Required Skills and Competencies: {job_skills}  
  
### Candidate Profile:  
Name: {candidate_name}  
Current Role: {candidate_role}  
Location: {candidate_location}  
Summary: {candidate_summary}  
Experience: {candidate_experience}  
Skills: {candidate_skills}  
  
### Instructions:  
- Identify alignment and mismatches clearly.  
- Assign suitability score (0-100).  
- Justify clearly.  
- Response needs to be only the JSON string, no additional data like ``` or json needed, just the JSON as it is                                                   
                                                   
  
### Response Format (JSON):  
{{  
  "suitability_score": int,  
  "alignment_areas": [str],  
  "gaps_or_mismatches": [str],  
  "reasoning": str,  
  "recommendation": str  
}}  
""")  
  
# Data models for email endpoint  
class EmailRequest(BaseModel):  
    subject: str  
    addresses: List[str]  
    body: str  
  
# Endpoint 1: Candidate-Posting Matching  
@app.get("/match_candidates")  
def match_candidates():  
    try:  
        # Fetch job postings from Lever API  
        postings_response = requests.get("https://api.sandbox.lever.co/v1/postings", auth=auth)  
        #print(postings_response.json())
        postings = postings_response.json()["data"]  
  
        # Fetch candidates from Lever API  
        candidates_response = requests.get("https://api.sandbox.lever.co/v1/candidates", auth=auth)  
        candidates = candidates_response.json()["data"]  
  
        matched_results = []  
  
        # Simplified matching: Iterate over limited candidates and postings (for demo)  
        for candidate in candidates[:5]:  
            for posting in postings[:5]:  
                # Prepare prompt input  
                # Safely handle posting categories  
                posting_categories = posting.get("categories") or {}  
                posting_skills = ", ".join([str(v) for v in posting_categories.values() if v])  
                
                # Safely handle candidate tags and experiences  
                candidate_tags = candidate.get("tags") or []  
                candidate_skills = ", ".join([str(tag) for tag in candidate_tags if tag])  
                
                candidate_experience_list = candidate.get("experience") or []  
                candidate_experience = "; ".join(  
                    [str(exp.get("title")) for exp in candidate_experience_list if exp.get("title")]  
                )  
                
                prompt = prompt_template.format_messages(  
                    job_title=posting.get("text", "Not specified"),  
                    job_location=posting.get("location", "Not specified"),  
                    job_description=posting.get("descriptionPlain", "Not specified"),  
                    job_skills=posting_skills if posting_skills else "Not specified",  
                    candidate_name=candidate.get("name", "Not specified"),  
                    candidate_role=candidate.get("headline", "Not specified"),  
                    candidate_location=candidate.get("location", "Not specified"),  
                    candidate_summary=candidate.get("summary", "Not specified"),  
                    candidate_experience=candidate_experience if candidate_experience else "Not specified",  
                    candidate_skills=candidate_skills if candidate_skills else "Not specified",  
                )  
                # Invoke Azure OpenAI via LangChain  
                #print(prompt)
                response = llm(prompt).content  
  
                matched_results.append({  
                    "candidate": candidate.get("name"),  
                    "posting": posting.get("text"),  
                    "match_details": response  
                })  
  
        #print(f"returning results:  {matched_results}")
        import json
        with open("response.json","w") as f:
            json.dump(matched_results,f,indent =4)
        
        return {"matched_results": matched_results}  
  
    except Exception as e:  
        raise HTTPException(status_code=500, detail=f"Error during matching: {str(e)}")  
  
# Endpoint 2: SMTP Email Sending  
@app.post("/send_email")  
def send_email(email_request: EmailRequest):  
    try:  
        # Setup SMTP connection  
        server = smtplib.SMTP(GMAIL_SMTP_SERVER, GMAIL_SMTP_PORT)  
        server.starttls()  
        server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)  
        print(email_request)
  
        # Compose email  
        msg = MIMEMultipart()  
        msg['From'] = GMAIL_EMAIL  
        msg['To'] = ", ".join(email_request["addresses"])  
        msg['Subject'] = email_request["subject"]  
        msg.attach(MIMEText(email_request["body"], 'html'))  
  
        # Send email  
        server.sendmail(GMAIL_EMAIL, email_request["addresses"], msg.as_string())  
        server.quit()  
  
        return {"status": "success", "message": "Emails sent successfully."}  
  
    except Exception as e:  
        raise HTTPException(status_code=500, detail=f"Error sending email or invalid emails: {str(e)}")  


#print(match_candidates())
#email_request={"subject":"checking email","addresses":["focusedpeacock@gmail.com","rando@dadsddasd.com"],"body":"new email from server"}
#send_email(email_request)
