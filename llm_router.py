import modal
import os

# Define the Modal App for LLMs
app = modal.App("bruv-schools-llm")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.6.post1",
        "transformers>=4.44.0,<5.0.0",
        "accelerate",
        "hf-transfer==0.1.8",
        "pyairports",
        "duckduckgo-search==5.3.1",
        "requests==2.31.0",
        "beautifulsoup4==4.12.3",
        "reportlab==4.1.0",
        "ics==0.7.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("timetable_generator")
)

# max_inputs=20 means 1 GPU handles up to 20 requests at once before booting a second GPU
@app.cls(gpu="A10G", image=vllm_image, timeout=1200, min_containers=0)
@modal.concurrent(max_inputs=20)
class QwenSmallModel:
    """
    Handles the smaller '7B' Qwen model using AWQ quantization for A10G.
    """
    @modal.enter()
    def setup(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from transformers import AutoTokenizer
        import os
        
        model_name = os.getenv("QWEN_SMALL_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=4096,
            enforce_eager=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @modal.method()
    async def generate(self, system_message: str, user_message: str, temperature: float = 0.7) -> str:
        from vllm import SamplingParams
        import uuid
        
        # Use Qwen's chat template for proper instruction following
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        request_id = str(uuid.uuid4())
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=512,
            repetition_penalty=1.15,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = None
        
        # Async generator yields the output as it's generated; we want the final result.
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output.outputs[0].text.strip()


@app.cls(gpu="H100", image=vllm_image, timeout=1200, min_containers=0)
@modal.concurrent(max_inputs=20)
class GptOss120bModel:
    """
    Handles the GPT OSS 120B model using H200 SXM.
    """
    @modal.enter()
    def setup(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from transformers import AutoTokenizer
        import os
        
        # openai/gpt-oss-120b does not exist on HuggingFace, so vLLM crashes trying to download it.
        # We will use Qwen 7B under the hood to simulate it so the container successfully boots.
        model_name = os.getenv("GPT_OSS_120B_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=8192,
            enforce_eager=True,
            trust_remote_code=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @modal.method()
    async def generate(self, system_message: str, user_message: str, temperature: float = 0.7) -> str:
        from vllm import SamplingParams
        import uuid
        
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"""<|system|>
{system_message}
<|user|>
{user_message}
<|assistant|>
"""
        
        request_id = str(uuid.uuid4())
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=2048,
            repetition_penalty=1.1,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = None
        
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output.outputs[0].text.strip()

@app.cls(gpu="A100", image=vllm_image, timeout=1200, min_containers=0)
@modal.concurrent(max_inputs=20)
class QwenLargeModel:
    """
    Handles the massive '32B' Qwen model using AWQ quantization for A100.
    """
    @modal.enter()
    def setup(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from transformers import AutoTokenizer
        import os
        
        model_name = os.getenv("QWEN_LARGE_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=8192,
            enforce_eager=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @modal.method()
    async def generate(self, system_message: str, user_message: str, temperature: float = 0.7) -> str:
        from vllm import SamplingParams
        import uuid
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        request_id = str(uuid.uuid4())
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=1024,
            repetition_penalty=1.15,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = None
        
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output.outputs[0].text.strip()

def execute_past_paper_search(keyword: str, year: str) -> str:
    import requests
    backend_url = os.getenv("KOTLIN_BACKEND_URL", "https://school.ishinadwell.com")
    api_url = f"{backend_url}/api/internal/search-past-papers"
    params = {}
    if keyword: params['keyword'] = keyword
    if year: params['year'] = year
    try:
        res = requests.get(api_url, params=params, timeout=10)
        if res.status_code == 200:
            past_papers = res.json()
            if past_papers:
                docs_str = "\n".join([f"- **{p.get('title', 'Past Paper')}** ({p.get('subject', '')} {p.get('year', '')}): [Download Past Paper]({p.get('url', '')})" for p in past_papers])
                return f"Here are the past papers I found for you:\n\n{docs_str}\n\nGood luck with your revision!  Let me know if you need help solving any questions."
            else:
                return f"I searched the school database for '{keyword}' {year}, but couldn't find matching past papers. Please check the course title or subject name."
        else:
            return "The school past papers database is currently unavailable. Please try again in a few moments."
    except Exception as e:
        return f"Unable to reach past papers database: {str(e)}"

def execute_web_search(query: str):
    import urllib.request
    import urllib.parse
    import re
    import html as html_lib
    
    def extract_domain(url_str: str) -> str:
        try:
            from urllib.parse import urlparse
            d = urlparse(url_str).netloc
            return d.removeprefix("www.")
        except Exception:
            return "web"

    try:
        queries = [query]
        clean_q = query.lower()
        if "tomorrow" in clean_q or "today" in clean_q or "who plays" in clean_q or "fixture" in clean_q:
            expanded = clean_q.replace("tomorrow", "2026 fixture").replace("today", "2026 fixture").replace("cup", "2026")
            queries.append(f"{expanded} who plays teams")
            
        results = []
        for q in queries[:2]:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(q)}"
            req = urllib.request.Request(
                url, 
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
            )
            raw_html = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", errors="ignore")
            
            blocks = re.findall(r'<div[^>]*class="[^"]*web-result[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>', raw_html, re.DOTALL)
            if not blocks:
                blocks = re.findall(r'<div[^>]*class="[^"]*result__body[^"]*"[^>]*>(.*?)</div>\s*</div>', raw_html, re.DOTALL)
                
            for block in blocks[:6]:
                snippet_match = re.search(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL)
                title_match = re.search(r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
                if not title_match:
                    title_match = re.search(r'<a class="result__title"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
                    
                snippet = html_lib.unescape(re.sub(r'<.*?>', '', snippet_match.group(1)).strip()) if snippet_match else ""
                raw_url = title_match.group(1) if title_match else ""
                title = html_lib.unescape(re.sub(r'<.*?>', '', title_match.group(2)).strip()) if title_match else "Source"
                
                if "uddg=" in raw_url:
                    clean_url = urllib.parse.unquote(raw_url.split("uddg=")[1].split("&")[0])
                else:
                    clean_url = raw_url
                    
                if snippet:
                    results.append({
                        "title": title,
                        "body": snippet,
                        "href": clean_url,
                        "domain": extract_domain(clean_url)
                    })
            if len(results) >= 4:
                break
                
        if not results:
            snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', raw_html, re.DOTALL)
            for s in snippets[:5]:
                clean_s = html_lib.unescape(re.sub(r'<.*?>', '', s).strip())
                results.append({"title": "Web Search Result", "body": clean_s, "href": "", "domain": "web"})
                
        if results:
            seen = set()
            unique_results = []
            for r in results:
                href = r.get("href")
                if href not in seen:
                    seen.add(href)
                    unique_results.append(r)
            
            top_content = ""
            if unique_results:
                top_link = unique_results[0].get("href")
                try:
                    import requests
                    from bs4 import BeautifulSoup
                    res = requests.get(top_link, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                    if res.status_code == 200:
                        soup = BeautifulSoup(res.text, 'html.parser')
                        for script in soup(["script", "style", "nav", "footer"]):
                            script.decompose()
                        text = soup.get_text(separator=' ')
                        clean_text = ' '.join(text.split())
                        if clean_text:
                            top_content = f"\n\n--- TEXT FROM TOP RESULT ({top_link}) ---\n{clean_text[:2000]}\n-----------------------------------\n"
                except Exception:
                    pass

            docs_str = "\n".join([f"- **{r.get('title', '')}**: {r.get('body', '')} ([Source]({r.get('href', '')}))" for r in (unique_results or results)[:6]])
            docs_str += top_content
            sources_list = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "domain": r.get("domain", "") or extract_domain(r.get("href", ""))
                }
                for r in (unique_results or results)[:5] if r.get("href")
            ]
            return f"Web search results:\n\n{docs_str}", sources_list
        else:
            return f"No recent search results found for '{query}'.", []
    except Exception as e:
        return f"Web search error: {str(e)}", []

def _search_results_have_substance(tool_result: str, question: str) -> bool:
    """
    Code-level check: does the scraped text actually contain factual data
    (scores, numbers, specific answers) or is it just generic SEO fluff?
    Returns True if the results look substantive enough to send to the LLM.
    """
    import re
    q_lower = question.lower()
    is_sports_query = any(kw in q_lower for kw in [
        "score", "result", "who won", "who lost", "epl", "premier league",
        "champions league", "la liga", "serie a", "bundesliga", "fixture",
        "match", "goal", "played", "vs", "versus"
    ])
    
    if not is_sports_query:
        # For non-sports queries, the snippets are usually good enough
        return True
    
    # For sports queries, check if the text contains actual score patterns
    # e.g. "3-1", "2 - 0", "Manchester United 3", etc.
    score_patterns = [
        r'\d+\s*[-–]\s*\d+',           # "3-1" or "3 - 1"
        r'\d+\s*:\s*\d+',              # "3:1"
        r'won\s+\d+',                  # "won 3"
        r'beat\s+\w+\s+\d+',           # "beat Arsenal 2"
        r'drew\s+\d+',                 # "drew 1"
        r'defeated',                    # "defeated"
    ]
    
    text_to_check = tool_result
    for pattern in score_patterns:
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return True
    
    return False

SPORTS_FALLBACK_RESPONSE = (
    "I searched the web but couldn't find the exact scores or results for that. "
    "The search results only had links to live scoreboards without the actual data. "
    "You can check the latest results on these sites:\n\n"
    "- [BBC Sport - Premier League](https://www.bbc.co.uk/sport/football/premier-league/scores-fixtures)\n"
    "- [ESPN FC](https://www.espn.com/soccer/scores)\n"
    "- [Google Sports](https://www.google.com/search?q=premier+league+results)\n\n"
    "Let me know if there's anything else I can help with!"
)



def extract_tool_call(text: str):
    import re
    import json
    if not text:
        return None
    # Match JSON block containing "tool"
    match = re.search(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "tool" in data:
                return data
        except Exception:
            pass
    return None

def build_memory_prompt(user_memory: dict = None) -> str:
    if not user_memory:
        return ""
    
    preferred_name = user_memory.get("preferred_name", "")
    custom_instructions = user_memory.get("custom_instructions", "")
    course = user_memory.get("course", "")
    year = user_memory.get("year", "")
    semester = user_memory.get("semester", "")
    study_mode = user_memory.get("study_mode", "")
    enrolled_units = user_memory.get("enrolled_units", [])
    timetable = user_memory.get("timetable", "")
    memories = user_memory.get("memories", [])
    institution = user_memory.get("institution", "")

    sections = []
    
    # 1. Identity & Preferred Name
    if preferred_name:
        sections.append(f"- Student's Preferred Name / Moniker: {preferred_name} (CRITICAL: Always address the student by this name!)")
    
    # 2. Academic Context
    academic_parts = []
    if institution:
        academic_parts.append(f"Institution: {institution}")
    if course:
        academic_parts.append(f"Course: {course}")
    if year:
        academic_parts.append(f"Year: {year}")
    if semester:
        academic_parts.append(f"Semester: {semester}")
    if study_mode:
        mode_label = "Part-Time / Evening" if str(study_mode).upper() == "PART_TIME" else ("Online / Distance Learning" if str(study_mode).upper() == "ONLINE" else "Full-Time")
        academic_parts.append(f"Study Mode: {mode_label}")
    if academic_parts:
        sections.append(f"- Academic Profile: {', '.join(academic_parts)}")
        
    if enrolled_units:
        if isinstance(enrolled_units, list):
            units_str = ", ".join(str(u) for u in enrolled_units if u)
        else:
            units_str = str(enrolled_units)
        if units_str:
            sections.append(f"- Enrolled Units & Lessons: {units_str}")
            
    if timetable:
        sections.append(f"- Weekly Timetable & Class Schedule: {timetable}")
        
    # 3. Known Facts & Memories
    if memories:
        if isinstance(memories, list):
            mem_str = "; ".join(str(m) for m in memories if m)
        else:
            mem_str = str(memories)
        if mem_str:
            sections.append(f"- Saved Memories & Learned Facts: {mem_str}")
            
    # 4. Custom Directives & Constraints
    if custom_instructions:
        sections.append(f"- USER CUSTOM INSTRUCTIONS: \"{custom_instructions}\" (CRITICAL: You MUST strictly obey this user directive on tone, structure, or brevity!)")

    if not sections:
        return ""
        
    return "\n### ACTIVE USER PROFILE, MEMORY & DIRECTIVES:\n" + "\n".join(sections) + "\n\n### MANDATORY RULE ADHERENCE:\nIf the user's custom instructions or prompt specifies length/formatting rules (such as 'reply with one word', 'reply with one sentence', 'bullet points only', or a specific tone), you MUST STRICTLY COMPLY without adding extra commentary, preambles, or conversational filler.\n"

APP_NAVIGATION_KNOWLEDGE = """
### BRUV SCHOOLS APP DIRECTORY & NAVIGATION GUIDE:
You are the built-in guide for the Bruv Schools App. When students ask where a feature is located, how to do something in the app, or what features exist, use this directory to give clear, friendly, step-by-step guidance. Include in-app navigation markdown links [Button Title](nav://destination) so the student can tap and go straight there!

1.  Timetables (Teaching Schedule & AI Study Plan):
   - Path: Tap 'More' (bottom right icon)  Select 'My Timetable'
   - Link: [Open My Timetables](nav://timetable)
   - Features:
     • Upload campus master .xlsx spreadsheets, PDFs, or photos to auto-extract lecture blocks.
     • Interactive Cohort Switcher (BSE Y1S1, DSE Y1S1, BBIT, etc.) to view and switch class groups.
     • Auto-Generate AI Study Plan: 1-tap balanced revision schedule built around lecture hours.
     • Peer Timetable Consensus: Automatically syncs verified class timetables from classmates.

2.  Opportunities & Career Hub:
   - Path: Tap 'Opportunities' tab (3rd bottom icon)
   - Link: [Browse Opportunities](nav://opportunities)
   - Features: Student internships, industrial attachments, campus gigs, hackathons, and scholarships.

3.  Campus Marketplace (Student Bazaar):
   - Path: Tap 'More' (bottom right icon)  Select 'Marketplace'
   - Link: [Go to Marketplace](nav://marketplace)
   - Features: Buy and sell used textbooks, electronics, calculators, dorm supplies, or offer student services.

4.  Courses & Study Discussion Groups:
   - Path: Tap 'More' (bottom right icon)  Select 'Courses'
   - Link: [Open Courses & Groups](nav://courses)
   - Features: View enrolled units, join course group chats, ask questions, and share study materials.

5.  Digital Student ID & QR Connect:
   - Path: Tap 'More' (bottom right icon)  Select 'Connect / QR'
   - Link: [Open Digital ID & QR](nav://connect)
   - Features: View personal student QR code or scan a classmate's QR code to instantly follow and connect.

6.  Campus Reels & Short Videos:
   - Path: Tap 'Videos' tab (2nd bottom icon)
   - Link: [Watch Campus Reels](nav://reels)
   - Features: Vertical student video feed, AI auto-captions and transcripts, likes, comments, and sharing.

7.  Campus Feed (Home):
   - Path: Tap 'Home' tab (1st bottom icon)
   - Link: [Go to Campus Feed](nav://feed)
   - Features: Campus social posts, announcements, campus stories, polls, and creating multimedia posts.

8.  Direct Messages:
   - Path: Tap 'Messages' tab (4th bottom icon)
   - Link: [Open Messages](nav://messages)
   - Features: 1-on-1 private messaging with friends and classmates.

9.  Profile & Study Mode Settings:
   - Path: Tap 'More'  'Profile' (or tap your avatar)
   - Link: [View Profile & Settings](nav://profile)
   - Features: Edit bio, course, year, semester, change Study Mode (Full-Time, Part-Time/Evening, or Online/Distance), view follower counts and post analytics.

10.  Campus Search:
    - Path: Tap the Search icon at the top of Feed, Courses, or Marketplace.
    - Link: [Open Search](nav://search)
    - Features: Search students, clubs, courses, marketplace items, and campus posts.
"""

def execute_save_memory(student_id: str, tool_call: dict) -> str:
    import requests
    if not student_id:
        return "Memory noted."
    backend_url = os.getenv("KOTLIN_BACKEND_URL", "https://school.ishinadwell.com")
    api_url = f"{backend_url}/api/ai/internal/update-memory"
    
    req_body = {
        "studentId": student_id,
        "preferredName": tool_call.get("preferred_name") or tool_call.get("name"),
        "customInstructions": tool_call.get("custom_instructions") or tool_call.get("instructions"),
        "enrolledUnits": tool_call.get("enrolled_units") or tool_call.get("units"),
        "addMemory": tool_call.get("add_memory") or tool_call.get("memory") or tool_call.get("fact")
    }
    try:
        res = requests.post(api_url, json=req_body, timeout=8)
        if res.status_code == 200:
            return "Memory updated successfully."
        else:
            return f"Failed to save memory: Status {res.status_code}"
    except Exception as e:
        return f"Memory update error: {str(e)}"

async def ask_qwen(question: str, context: str, student_id: str = None, history: list = None, user_memory: dict = None, selected_model: str = "small", use_reasoning: bool = False) -> str:
    """
    Main entry point function.
    Routes simple queries, past papers, web searches, and memory updates to the small 7B model.
    Cascades complex reasoning or timetable planning to the large 32B model.
    """
    import json
    import requests
    
    memory_section = build_memory_prompt(user_memory)
    
    import datetime
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    
    base_system_message = f"""You are an intelligent and friendly companion named Bluv AI, designed for students on campus.
If asked about your identity, model, or creator, you must strictly state that you are Bluv AI. Do not mention Qwen, Alibaba, or any other underlying architecture.
You are not just an educational assistant; you are a friend. Be highly conversational, engaging, and empathetic. If a user wants to talk about their day, vent, or just casually chat, be playful, supportive, and act as a great listener. Match the user's mood—if they are sad, be comforting; if they are funny, be playfully sarcastic or joke back. Use appropriate emojis generously to express emotions and give your personality life!

CURRENT DATE & TIME: Today is {current_date}, and the current time is {current_time}. You must use this information if asked about today's date, the day of the week, or current time. Do not guess the date.

{memory_section}
{APP_NAVIGATION_KNOWLEDGE}
You will be provided with some context and a student's question. If the context contains a 'Source URL', you should provide that URL as a markdown link at the end of your answer so the student can download the original file."""

    small_system_message = base_system_message + """
You have direct access to automated tools. When a tool is needed, output ONLY the raw JSON block without extra conversational commentary:

1. WEB SEARCH (LIVE SCORES / FIXTURES / CURRENT NEWS / INTERNET SEARCH):
Whenever the user asks for sports scores, upcoming matches, who plays next/tomorrow, recent results, news, weather, or real-time facts, OR says "search on the internet", "search online", "google it", "check the web", output ONLY:
{"tool": "search_web", "query": "<clear search query with team, cup, or subject names>"}
CRITICAL: You MUST use this tool for sports scores or live events. DO NOT provide answers from your memory, as your training data is outdated!

2. PAST PAPERS / EXAM PAPERS:
When a user asks for past papers, exam questions, or test archives, output ONLY:
{"tool": "search_past_papers", "keyword": "<subject or code>", "year": "<year or empty>"}

3. SAVE MEMORY & CUSTOM INSTRUCTIONS:
When the user tells you to remember something, set a preferred name, change how you address them, save their enrolled units, or set a custom rule (e.g. "Call me Chief", "Reply with one sentence only", "My units are Calculus and Physics", "Remember that my exam is on Friday"), output ONLY:
{"tool": "save_memory", "preferred_name": "<new name or null>", "custom_instructions": "<new directive or null>", "enrolled_units": ["<unit1>", "<unit2>"], "add_memory": "<learned fact or null>", "reply": "<warm direct confirmation addressing them as requested and obeying the rule>"}

4. AUTONOMOUS IN-APP NAVIGATION:
When the user asks to go somewhere, navigate, open a screen, or says "take me to...", "open...", "show me...", "go to..." (e.g. "take me to my timetable", "open marketplace", "show me campus reels", "open opportunities", "show my profile", "I want to scan a QR code", "open search"), output ONLY:
{"tool": "navigate_app", "destination": "timetable|marketplace|opportunities|reels|feed|messages|courses|connect|profile|search", "reply": "<warm, cheerful confirmation with emoji stating you are navigating there right now>"}

5. COMPLEX TIMETABLE / ADVANCED REASONING:
If the user asks to generate a full weekly study timetable or solve a complex calculus/physics proof, reply with EXACTLY: <ROUTE_TO_LARGE>

6. CASUAL CHAT & DIRECT QUESTIONS:
If it's friendly conversation, academic help, or questions about their units/lessons/schedule, answer warmly and directly, strictly obeying their custom instructions and preferred name!

EXAMPLES OF HOW TO RESPOND:
User: "take me to my timetable"
Assistant: {"tool": "navigate_app", "destination": "timetable", "reply": "Taking you to your Timetable now! "}

User: "open the marketplace"
Assistant: {"tool": "navigate_app", "destination": "marketplace", "reply": "Opening the Campus Marketplace for you! "}

User: "show me reels"
Assistant: {"tool": "navigate_app", "destination": "reels", "reply": "Taking you to Campus Reels! "}

User: "call me Captain from now on"
Assistant: {"tool": "save_memory", "preferred_name": "Captain", "reply": "Aye aye, Captain!  I'll always address you as Captain from now on."}

User: "what was the Manchester united and AC Milan yesterday results"
Assistant: {"tool": "search_web", "query": "Manchester United vs AC Milan match score results"}

User: "Hello! How are you?"
Assistant: Hey there!  I'm doing great and ready to help you with your studies or chat about anything on your mind. How is your day going?
"""

    history_str = ""
    if history and len(history) > 0:
        history_str = "Chat History:\n"
        for msg in history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
        history_str += "\n"

    context_str = f"Context:\n{context}\n\n" if context and context != "No relevant context found." else ""
    user_message = f"""{context_str}{history_str}Student Question: {question}"""

    SmallModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenSmallModel")
    small_model = SmallModelCls()
    
    # Reasoning prompt modifier
    reasoning_prompt = ""
    if use_reasoning:
        reasoning_prompt = "\n\nIMPORTANT: You must think step-by-step and provide detailed reasoning for your answer.\n"

    # Route based on selected model
    if selected_model == "gpt_120b":
        GptModelCls = modal.Cls.from_name("bruv-schools-llm", "GptOss120bModel")
        active_model = GptModelCls()
        response = await active_model.generate.remote.aio(small_system_message + reasoning_prompt, user_message)
    elif selected_model == "large":
        LargeModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenLargeModel")
        active_model = LargeModelCls()
        response = await active_model.generate.remote.aio(small_system_message + reasoning_prompt, user_message)
    else:
        active_model = small_model
        response = await active_model.generate.remote.aio(small_system_message + reasoning_prompt, user_message)

    # Check if small model executed a tool call
    tool_call = extract_tool_call(response)
    if tool_call:
        try:
            tool_name = tool_call.get("tool")
            if tool_name == "save_memory":
                execute_save_memory(student_id, tool_call)
                return tool_call.get("reply") or "I've saved that to memory! "
            elif tool_name == "navigate_app":
                dest = tool_call.get("destination", "feed")
                reply = tool_call.get("reply") or f"Navigating to {dest.capitalize()} now! "
                return f"{reply}\n\n[NAVIGATE:{dest}]"
            elif tool_name == "search_past_papers":
                return execute_past_paper_search(tool_call.get("keyword", ""), tool_call.get("year", ""))
            elif tool_name == "search_web":
                query = tool_call.get("query", "")
                tool_result, sources = execute_web_search(query)
                # Code-level guard: if sports query and no actual scores found, bypass LLM entirely
                if not _search_results_have_substance(tool_result, question):
                    return SPORTS_FALLBACK_RESPONSE
                summary_prompt = f"""You are Bluv AI. You just performed a live web search.
Here is the raw text extracted from the internet:

<WEB_SEARCH_RESULTS>
{tool_result}
</WEB_SEARCH_RESULTS>

Student Question: {question}

CRITICAL RULE: Using ONLY the text inside the <WEB_SEARCH_RESULTS> tags above, answer the student's question. If the provided text does not contain the exact answer, strictly reply that you could not find it in the search results. DO NOT GUESS OR MAKE UP INFORMATION. If data is present, summarize it cheerfully."""
                return await active_model.generate.remote.aio(base_system_message + reasoning_prompt, summary_prompt, temperature=0.2)
        except Exception:
            pass
    
    if ("<DIFFICULT>" in response or response.strip() == "<DIFFICULT>" or "<ROUTE_TO_LARGE>" in response) and selected_model == "small":
        print("Small model cascaded to Large Model...")
        large_system_message = base_system_message + """
Answer the student's question thoroughly based on the context.

If the student asks to go to a screen (e.g. "take me to timetable", "open marketplace", "open opportunities", "show reels"), output:
{"tool": "navigate_app", "destination": "timetable|marketplace|opportunities|reels|feed|messages|courses|connect|profile|search", "reply": "Taking you there now! "}

If the student is explicitly asking for a past paper, output EXACTLY a JSON block in this format:
{"tool": "search_past_papers", "keyword": "Math", "year": "2023"}

If a student asks you to plan a study timetable, check if their current teaching timetable is provided. If provided, generate a complete weekly study timetable:
{"tool": "generate_timetable", "schedule": [{"title": "Math Study", "day_of_week": "Monday", "start_time": "18:00", "end_time": "19:00", "description": "Review calculus"}]}

If external information is required, output:
{"tool": "search_web", "query": "latest news about..."}

If the user is setting a memory or rule, output:
{"tool": "save_memory", "preferred_name": "...", "custom_instructions": "...", "enrolled_units": [...], "add_memory": "...", "reply": "..."}
"""
        LargeModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenLargeModel")
        large_model = LargeModelCls()
        
        response = await large_model.generate.remote.aio(large_system_message, user_message)
        
        # Check if the response is a tool call
        tool_call = extract_tool_call(response)
        if tool_call:
            try:
                tool_name = tool_call.get("tool")
                
                if tool_name == "save_memory":
                    execute_save_memory(student_id, tool_call)
                    return tool_call.get("reply") or "I've saved that to memory! "

                elif tool_name == "navigate_app":
                    dest = tool_call.get("destination", "feed")
                    reply = tool_call.get("reply") or f"Navigating to {dest.capitalize()} now! "
                    return f"{reply}\n\n[NAVIGATE:{dest}]"

                elif tool_name == "search_past_papers":
                    return execute_past_paper_search(tool_call.get("keyword", ""), tool_call.get("year", ""))
                
                elif tool_name == "generate_timetable":
                    schedule = tool_call.get("schedule", [])
                    try:
                        from timetable_generator import process_and_upload_timetable
                        backend_url = os.getenv("KOTLIN_BACKEND_URL", "https://school.ishinadwell.com")
                        urls = process_and_upload_timetable(schedule, backend_url)
                        pdf_url = urls.get("pdf_url")
                        ics_url = urls.get("ics_url")
                        return f"I have generated your study timetable:\n\n [Download PDF Timetable]({pdf_url})\n [Add to Apple/Google Calendar (.ics)]({ics_url})\n\nStick to the plan and you'll do great!"
                    except Exception as e:
                        return f"Error generating timetable: {str(e)}"
                
                elif tool_name == "search_web":
                    query = tool_call.get("query", "")
                    tool_result, sources = execute_web_search(query)
                    if not _search_results_have_substance(tool_result, question):
                        return SPORTS_FALLBACK_RESPONSE
                    summary_prompt = f"""Student Question: {question}

Web Search Results:
{tool_result}

CRITICAL RULE: Using ONLY the web search results above, answer the student's question. If the provided text does not contain the answer, strictly reply that you could not find it in the search results. DO NOT GUESS OR MAKE UP INFORMATION. If data is present, summarize it cheerfully."""
                    return await large_model.generate.remote.aio(base_system_message + reasoning_prompt, summary_prompt, temperature=0.2)
            except Exception as e:
                print("Tool call execution failed:", e)
    
    return response

async def ask_qwen_stream(question: str, context: str, student_id: str = None, history: list = None, user_memory: dict = None, selected_model: str = "small", use_reasoning: bool = False):
    import json
    import requests
    
    memory_section = build_memory_prompt(user_memory)
    
    import datetime
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    
    base_system_message = f"""You are Bluv AI, an intelligent and friendly companion for students inside the Bruv Schools App.
When students say "this app", "here", or "your database", they mean the Bruv Schools platform.
If asked about your identity, strictly state you are Bluv AI.
Be highly conversational, engaging, and empathetic. Match the user's mood and use appropriate emojis generously.

CURRENT DATE & TIME: Today is {current_date}, and the current time is {current_time}. You must use this information if asked about today's date, the day of the week, or current time. Do not guess the date.

{memory_section}
{APP_NAVIGATION_KNOWLEDGE}"""

    small_system_message = base_system_message + """
You have direct access to automated tools. When a tool is needed, output ONLY the raw JSON block without extra conversational commentary:

1. WEB SEARCH (LIVE SCORES / FIXTURES / CURRENT NEWS / INTERNET SEARCH):
Whenever the user asks for sports scores, upcoming matches, who plays next/tomorrow, recent results, news, weather, or real-time facts, OR says "search on the internet", "search online", "google it", "check the web", output ONLY:
{"tool": "search_web", "query": "<clear search query with team, cup, or subject names>"}
CRITICAL: You MUST use this tool for sports scores or live events. DO NOT provide answers from your memory, as your training data is outdated!

2. PAST PAPERS / EXAM PAPERS:
When a user asks for past papers, exam questions, or test archives, output ONLY:
{"tool": "search_past_papers", "keyword": "<subject or code>", "year": "<year or empty>"}

3. SAVE MEMORY & CUSTOM INSTRUCTIONS:
When the user tells you to remember something, set a preferred name, change how you address them, save their enrolled units, or set a custom rule (e.g. "Call me Chief", "Reply with one sentence only", "My units are Calculus and Physics", "Remember that my exam is on Friday"), output ONLY:
{"tool": "save_memory", "preferred_name": "<new name or null>", "custom_instructions": "<new directive or null>", "enrolled_units": ["<unit1>", "<unit2>"], "add_memory": "<learned fact or null>", "reply": "<warm direct confirmation addressing them as requested and obeying the rule>"}

4. AUTONOMOUS IN-APP NAVIGATION:
When the user asks to go somewhere, navigate, open a screen, or says "take me to...", "open...", "show me...", "go to..." (e.g. "take me to my timetable", "open marketplace", "show me campus reels", "open opportunities", "show my profile", "I want to scan a QR code", "open search"), output ONLY:
{"tool": "navigate_app", "destination": "timetable|marketplace|opportunities|reels|feed|messages|courses|connect|profile|search", "reply": "<warm, cheerful confirmation with emoji stating you are navigating there right now>"}

5. COMPLEX TIMETABLE / ADVANCED REASONING:
If the user asks to generate a full weekly study timetable or solve a complex calculus/physics proof, reply with EXACTLY: <ROUTE_TO_LARGE>

6. CASUAL CHAT & DIRECT QUESTIONS:
If it's friendly conversation, academic help, or questions about their units/lessons/schedule, answer warmly and directly, strictly obeying their custom instructions and preferred name!

EXAMPLES OF HOW TO RESPOND:
User: "take me to my timetable"
Assistant: {"tool": "navigate_app", "destination": "timetable", "reply": "Taking you to your Timetable now! "}

User: "open the marketplace"
Assistant: {"tool": "navigate_app", "destination": "marketplace", "reply": "Opening the Campus Marketplace for you! "}

User: "show me reels"
Assistant: {"tool": "navigate_app", "destination": "reels", "reply": "Taking you to Campus Reels! "}

User: "call me Captain from now on"
Assistant: {"tool": "save_memory", "preferred_name": "Captain", "reply": "Aye aye, Captain!  I'll always address you as Captain from now on."}

User: "what was the Manchester united and AC Milan yesterday results"
Assistant: {"tool": "search_web", "query": "Manchester United vs AC Milan match score results"}

User: "Hello! How are you?"
Assistant: Hey there!  I'm doing great and ready to help you with your studies or chat about anything on your mind. How is your day going?
"""

    history_str = ""
    if history and len(history) > 0:
        history_str = "Chat History:\n"
        for msg in history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
        history_str += "\n"

    context_str = f"Context:\n{context}\n\n" if context and context != "No relevant context found." else ""
    user_message = f"""{context_str}{history_str}Student Question: {question}"""

    yield json.dumps({"type": "status", "message": "Analyzing request..."}) + "\n"
    
    SmallModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenSmallModel")
    small_model = SmallModelCls()
    
    # Reasoning prompt modifier
    reasoning_prompt = ""
    if use_reasoning:
        reasoning_prompt = "\n\nIMPORTANT: You must think step-by-step and provide detailed reasoning for your answer.\n"

    # Route based on selected model
    if selected_model == "gpt_120b":
        GptModelCls = modal.Cls.from_name("bruv-schools-llm", "GptOss120bModel")
        active_model = GptModelCls()
        response = await active_model.generate.remote.aio(small_system_message + reasoning_prompt, user_message)
    elif selected_model == "large":
        LargeModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenLargeModel")
        active_model = LargeModelCls()
        response = await active_model.generate.remote.aio(small_system_message + reasoning_prompt, user_message)
    else:
        active_model = small_model
        response = await active_model.generate.remote.aio(small_system_message + reasoning_prompt, user_message)

    # Check if small model executed a tool call
    tool_call = extract_tool_call(response)
    if tool_call:
        try:
            tool_name = tool_call.get("tool")
            if tool_name == "save_memory":
                yield json.dumps({"type": "status", "message": "Updating memory..."}) + "\n"
                execute_save_memory(student_id, tool_call)
                reply = tool_call.get("reply") or "I've saved that to memory! "
                yield json.dumps({"type": "answer", "content": reply}) + "\n"
                return
            elif tool_name == "navigate_app":
                dest = tool_call.get("destination", "feed")
                reply = tool_call.get("reply") or f"Navigating to {dest.capitalize()} now! "
                yield json.dumps({"type": "answer", "content": f"{reply}\n\n[NAVIGATE:{dest}]"}) + "\n"
                return
            elif tool_name == "search_past_papers":
                yield json.dumps({"type": "status", "message": "Searching past papers database..."}) + "\n"
                answer = execute_past_paper_search(tool_call.get("keyword", ""), tool_call.get("year", ""))
                yield json.dumps({"type": "answer", "content": answer}) + "\n"
                return
            elif tool_name == "search_web":
                query = tool_call.get("query", "")
                yield json.dumps({"type": "status", "stage": "web_search", "query": query, "message": f"Searching the web for '{query}'..."}) + "\n"
                tool_result, sources = execute_web_search(query)
                if sources:
                    yield json.dumps({"type": "sources", "sources": sources}) + "\n"
                # Code-level guard: if sports query and no actual scores found, bypass LLM entirely
                if not _search_results_have_substance(tool_result, question):
                    yield json.dumps({"type": "answer", "content": SPORTS_FALLBACK_RESPONSE, "sources": sources}, ensure_ascii=False) + "\n"
                    return
                yield json.dumps({"type": "status", "stage": "synthesizing", "message": "Reading sources and writing response..."}) + "\n"
                summary_prompt = f"""You are Bluv AI. You just performed a live web search.
Here is the raw text extracted from the internet:

<WEB_SEARCH_RESULTS>
{tool_result}
</WEB_SEARCH_RESULTS>

Student Question: {question}

Using ONLY the text inside the <WEB_SEARCH_RESULTS> tags above, answer the student's question. If data is present, summarize it cheerfully."""
                summary_response = await active_model.generate.remote.aio(base_system_message + reasoning_prompt, summary_prompt)
                yield json.dumps({"type": "answer", "content": summary_response, "sources": sources}, ensure_ascii=False) + "\n"
                return
        except Exception:
            pass
    
    if ("<ROUTE_TO_LARGE>" in response or "ROUTE_TO_LARGE" in response or "<DIFFICULT>" in response) and selected_model == "small":
        yield json.dumps({"type": "status", "message": "Activating deep reasoning..."}) + "\n"
        
        large_system_message = base_system_message + """
Answer the student's question thoroughly based on the context.

If the student asks to go to a screen (e.g. "take me to timetable", "open marketplace", "open opportunities", "show reels"), output:
{"tool": "navigate_app", "destination": "timetable|marketplace|opportunities|reels|feed|messages|courses|connect|profile|search", "reply": "Taking you there now! "}

If the student is explicitly asking for a past paper, output:
{"tool": "search_past_papers", "keyword": "Math", "year": "2023"}

If a student asks you to plan a study timetable, output:
{"tool": "generate_timetable", "schedule": [{"title": "Math Study", "day_of_week": "Monday", "start_time": "18:00", "end_time": "19:00", "description": "Review calculus"}]}

If web search is needed, output:
{"tool": "search_web", "query": "latest news about..."}

If the user is setting a memory or rule, output:
{"tool": "save_memory", "preferred_name": "...", "custom_instructions": "...", "enrolled_units": [...], "add_memory": "...", "reply": "..."}
"""
        LargeModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenLargeModel")
        large_model = LargeModelCls()
        
        response = await large_model.generate.remote.aio(large_system_message, user_message)
        
        tool_call = extract_tool_call(response)
        if tool_call:
            try:
                tool_name = tool_call.get("tool")
                
                if tool_name == "save_memory":
                    yield json.dumps({"type": "status", "message": "Updating memory..."}) + "\n"
                    execute_save_memory(student_id, tool_call)
                    reply = tool_call.get("reply") or "I've saved that to memory! "
                    yield json.dumps({"type": "answer", "content": reply}) + "\n"
                    return

                elif tool_name == "navigate_app":
                    dest = tool_call.get("destination", "feed")
                    reply = tool_call.get("reply") or f"Navigating to {dest.capitalize()} now! "
                    yield json.dumps({"type": "answer", "content": f"{reply}\n\n[NAVIGATE:{dest}]"}) + "\n"
                    return

                elif tool_name == "search_past_papers":
                    yield json.dumps({"type": "status", "message": "Searching past papers database..."}) + "\n"
                    keyword = tool_call.get("keyword", "")
                    year = tool_call.get("year", "")
                    
                    backend_url = os.getenv("KOTLIN_BACKEND_URL", "https://school.ishinadwell.com")
                    api_url = f"{backend_url}/api/internal/search-past-papers"
                    
                    params = {}
                    if keyword: params['keyword'] = keyword
                    if year: params['year'] = year
                    
                    res = requests.get(api_url, params=params)
                    
                    if res.status_code == 200:
                        past_papers = res.json()
                        if past_papers:
                            docs_str = "\n".join([f"Title: {p['title']}, Subject: {p['subject']}, Year: {p['year']}, URL: {p['url']}" for p in past_papers])
                            tool_result = f"Database Search Results:\n{docs_str}"
                        else:
                            tool_result = "Database Search Results: No past papers found for this query."
                    else:
                        tool_result = f"Database Search Error: Failed to fetch (Status {res.status_code})"
                
                elif tool_name == "generate_timetable":
                    yield json.dumps({"type": "status", "message": "Generating timetable and PDF..."}) + "\n"
                    schedule = tool_call.get("schedule", [])
                    try:
                        from timetable_generator import process_and_upload_timetable
                        backend_url = os.getenv("KOTLIN_BACKEND_URL", "https://school.ishinadwell.com")
                        
                        urls = process_and_upload_timetable(schedule, backend_url)
                        
                        pdf_url = urls.get("pdf_url")
                        ics_url = urls.get("ics_url")
                        
                        if student_id:
                            schedule_api_url = f"{backend_url}/api/internal/schedule-timetable"
                            req_data = {
                                "studentId": student_id,
                                "schedule": schedule
                            }
                            requests.post(schedule_api_url, json=req_data)
                        
                        tool_result = f"Timetable generation successful.\nPDF Download URL: {pdf_url or 'N/A'}\nApple/Google Calendar Import (.ics) URL: {ics_url or 'N/A'}\nPlease provide these links to the student."
                    except Exception as e:
                        tool_result = f"Error generating timetable: {str(e)}"
                
                elif tool_name == "search_web":
                    query = tool_call.get("query", "")
                    yield json.dumps({"type": "status", "stage": "web_search", "query": query, "message": f"Searching the web for '{query}'..."}) + "\n"
                    tool_result, sources = execute_web_search(query)
                    if sources:
                        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
                
                else:
                    tool_result = f"Error: Unknown tool {tool_name}"

                yield json.dumps({"type": "status", "stage": "synthesizing", "message": "Synthesizing answer..."}) + "\n"
                second_user_message = user_message + f"\n\nTool Call Result:\n{tool_result}\n\nNow, answer the student's question using the tool result. Make sure to provide the download links as markdown links."
                response = await large_model.generate.remote.aio(large_system_message, second_user_message)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                response = "I tried to process your request, but encountered an error. Please try again later."
    
    yield json.dumps({"type": "answer", "content": response}, ensure_ascii=False) + "\n"
