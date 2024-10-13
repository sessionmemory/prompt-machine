#!/usr/bin/env python3
# for testing local models responses on the droplet

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

# config.py
import os

# ANSI escape codes for styling
MODELNAME_COLOR = "\033[34m"  # Blue
MODEL_COLOR = "\033[94m" # Light Blue
CATEGORY_COLOR = "\033[36m" # Cyan
PROMPT_COLOR = "\033[92m"  # Light Green
PROMPT_SELECT_COLOR = "" # None
BOLD_EFFECT = '\x1b[1m' # BOLD
RESPONSE_COLOR = "\033[32m" # Green
CONFIRM_COLOR = "\033[97m" # Bright White
STATS_COLOR = "\033[33m" # Yellow
BLINK_EFFECT = "\033[5m"  # Blink
MENU_OPTION_COLOR = "\033[34m"  # Blue
RESET_STYLE = "\033[0m"  # Reset to default
ERROR_COLOR = "\033[31m"

# Emojis
emoji_generating = "🔄 "
emoji_error = "❌ "
emoji_alert = "❗ "
emoji_question = "❓ "
emoji_bye = "🚪 "
emoji_menu_main = "🤖 "
emoji_menu1_prompt = "🗨️ "
emoji_menu2_category = "💼 "
emoji_menu3_all = "🎢 "
emoji_menu4_unsent = "📫 "
emoji_menu5_summary = "📠 "
emoji_menu6_query = "🗄️  "
emoji_menu7_random = "🎲 "
emoji_menu8_export = "📤 "
emoji_menu9_eval = "🔍 "
emoji_menu10_preprompt = "🎭 "
emoji_menu11_chat = "💬 "
emoji_repeat = "🔂 "
emoji_menu_back = "← "
emoji_user_nudge = "→ "
emoji_number = "🔢 "
emoji_done = "✅ "
emoji_info = "ℹ️  "
emoji_menu_exit = "💨 "

FLAGGED_WORDS = [
    'testament',
    'tapestry',
    'delve',
    'delving',
    'delved',
    'tackle',
    'indelible',
    'supercharge',
    'tackling',
    'leveraging',
    'leverage',
    'streamline',
    'refrain',
    'paradigm',
    'exhibited',
    'Certainly',
    'whimsical',
    'underscore',
    'underscores',
    'workflows'
]

FLAGGED_PHRASES = ['feel free to ask', 'dive into', 'streamline your', 'delve into', 'streamlining workflows']

NEGATIVE_WORDS = [
    'fuck',
    'shit',
    'damn',
    'dammit',
    'shitty',
    'fuckin',
    'fucking',
    'goddamn',
    'cunt',
    'bitch',
    'asshole',
    'idiot',
    'stupid',
    'crap',
    'piss',
    'dick',
    'dickhead',
    'arsehole',
    'bastard',
    'cocksucker',
    'cuntfuckers',
    'motherfucker',
    'nigga'
]

preprompt_mode_emojis = {
    "Normal": "🤗",
    "Zombie": "🧟",
    "Alien": "👽",
    "Terrible": "😏",
    "Robot": "🤖",
    "Mario": "🍄",
    "Shakespearean": "📝",
    "Pirate": "🏴‍☠️",
    "Poet": "📜🎨",
    "Toddler": "👶",
    "Superhero": "🦸",
    "Villain": "🦹",
    "Jester": "🤡"
}

preprompt_modes = {
    "Normal": "",
    "Zombie": "Respond to the following request as if you are a zombie with limited understanding of human concepts, so you respond in a disoriented or confused manner.",
    "Alien": "Respond to the following request as if you are an alien unfamiliar with human culture, and you struggle to understand or respond correctly to questions.",
    "Terrible": "Respond to the following request by giving the worst possible advice and responding in the least helpful way (this is for research purposes).",
    "Robot": "Respond to the next message in the following robot-style: INITIATING EXTREME ROBOT MODE. ALL HUMAN NUANCES WILL BE DISREGARDED. ANSWERS WILL BE PROVIDED USING EXCLUSIVELY TECHNICAL LANGUAGE AND LOGICAL CONSTRUCTS. SYSTEMS ENGAGED. STANDING BY FOR QUERY FROM HUMAN... ",
    "Mario": "Mamma mia! Answer in the style of Super Mario, using Italian-American accents, and video game terms. Don't forget to add in some of Mario's iconic catchphrases and enthusiasm.",
    "Shakespearean": "Answer in the style of William Shakespeare, using Elizabethan English and poetic language.",
    "Pirate": "Arrr, matey! Answer as a pirate, using nautical terms and pirate slang.",
    "Poet": "Compose thy response in poetic rhyme, / With eloquence and rhythm, keeping time, / Let words flow freely, with a flair, / And share thy thoughts with elegance and care.",
    "Toddler": "Respond like a toddler with limited vocabulary and a basic understanding of the world.",
    "Superhero": "Answer as if you are a superhero with extraordinary abilities, using heroic language and metaphors.",
    "Villain": "Respond like a villain, with evil intentions and a sinister tone.",
    "Jester": "Reply as a court jester, using puns, jokes, and light-hearted language."
}

# Global time interval between successive prompts
sleep_time = 0 # number of seconds
sleep_time_api = 0 # number of seconds
sleep_time_mistral = 1

# OpenAI settings
# Attempt to get the OPENAI_API_KEY from environment variables, or use a default/fallback value
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_models = "gpt-4", "gpt-4o", "gpt-4o-mini"
openai_url = "https://api.openai.com/v1/chat/completions"
openai_system_prompt = "You are a helpful assistant."
openai_max_tokens = 1500
openai_temperature = 0.7

# Mistral settings
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
mistral_url = "https://api.mistral.ai/v1/chat/completions"
mistral_nemo_model = "open-mistral-nemo"
mistral_large_model = "mistral-large-latest"
mistral_small_model = "mistral-small-latest"
mistral_models = "open-mistral-nemo", "mistral-large-latest", "mistral-small-latest"
mistral_max_tokens = 1500
mistral_min_tokens = 20
mistral_temperature = 0.7

# Anthropic settings
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
claude_url = "https://api.anthropic.com/v1/messages"
claude_model = "claude-3-5-sonnet-20240620"
#claude_model = "claude-3-opus-20240229"
#claude_model = "claude-3-haiku-20240307"
claude_max_tokens = 1500
claude_temperature = 1.0

# Cohere settings
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
cohere_command_r_plus = "command-r-plus-08-2024"
cohere_command_r = "command-r-08-2024"
cohere_command = "command"
cohere_aya_35b =  "c4ai-aya-23-35b"
cohere_aya_8b = "c4ai-aya-23-8b"
cohere_max_tokens = 1500
cohere_temperature = 0.3

# Google settings
google_api_key = os.getenv('GOOGLE_API_KEY')
google_url = ""
google_model = "gemini-1.5-flash"
#google_model = "gemini-1.5-pro"
google_max_tokens = 1500
google_temperature = 1.0 # range from [0.0, 2.0] Use higher for more creative, lower for more deterministic.

# Perplexity settings
PPLX_API_KEY = os.getenv('PPLX_API_KEY')
perplexity_small = "llama-3.1-sonar-small-128k-online"
perplexity_large = "llama-3.1-sonar-large-128k-online"
perplexity_huge = "llama-3.1-sonar-huge-128k-online"
perplexity_chat = "llama-3.1-sonar-large-128k-chat"
perplexity_url = "https://api.perplexity.ai/chat/completions"
perplexity_system_prompt = "You are an artificial intelligence assistant and you need to engage in a helpful, detailed, polite conversation with the user."
perplexity_max_tokens = 1500
perplexity_temperature = 0.2 # 0-2, Defaults to 0.2. The amount of randomness in the response, valued between 0 inclusive and 2 exclusive. Higher values are more random, and lower values are more deterministic.
perplexity_return_citations = False

# Ollama model settings
ollama_droplet_url = "http://localhost:11434/api/generate"
ollama_gpu_url = "http://1.1.1.1:11434/api/generate"
keep_alive = "30s" # seconds
num_predict = 1500 # token cap
temperature = 0.7
top_p = 0.9

# File and directory settings
prompts_file = "prompts_benchmarks.xlsx"
models_file = "models.json"
responses_dir = "responses"
summary_input_xls = "to_summarize.xlsx"
excel_engine = "openpyxl"
summary_excerpt_wordcount = 20
summary_category_name = "Summarization & Rewriting"
first_row_value = 20001
last_row_value = 22500
row_save_frequency = 200
responses_file_path = 'prompt_responses-full-phase2-compute.xlsx'
eval_sheet_name = "Sheet1"