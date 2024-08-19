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
PROMPT_COLOR = "\033[95m"  # Magenta
PROMPT_SELECT_COLOR = "" # Bright Magenta
BOLD_EFFECT = '\x1b[1m' # BOLD
RESPONSE_COLOR = "\033[32m" # Green
CONFIRM_COLOR = "\033[97m" # Bright White
STATS_COLOR = "\033[33m" # Yellow
BLINK_EFFECT = "\033[5m"  # Blink
RESET_STYLE = "\033[0m"  # Reset to default
ERROR_COLOR = {BOLD_EFFECT}

# Creating a styled, blinged-out message
welcome_message = (
    f"{BLINK_EFFECT}{BOLD_EFFECT}{MODEL_COLOR}✨🌟 Welcome ✨ "
    f"{CATEGORY_COLOR}🎈✨ to the ✨🎈 "
    f"{PROMPT_COLOR}🚀✨ Prompt ✨🚀 "
    f"{RESPONSE_COLOR}🎉✨ Machine! ✨🎉"
    f"{RESET_STYLE}"
)

# Emojis
emoji_generating = "🔄 "
emoji_error = "❌ "
emoji_alert = "❗ "
emoji_question = "❓ "
emoji_bye = "🚪 "
emoji_menu_main = "🤖 "
emoji_menu1_single = "1️⃣  "
emoji_menu2_prompt = "💬 "
emoji_menu3_category = "💼 "
emoji_menu4_all = "🎢 "
emoji_menu5_unsent = "📫 "
emoji_menu6_summary = "📠 "
emoji_menu7_query = "🗄️  "
emoji_menu8_exit = "💨 "
emoji_menu_back = "← "
emoji_user_nudge = "→ "
emoji_number = "🔢 "
emoji_done = "✅ "
emoji_info = "ℹ️ "

# OpenAI settings
# Attempt to get the OPENAI_API_KEY from environment variables, or use a default/fallback value
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_url = "https://api.openai.com/v1/chat/completions"
openai_system_prompt = "You are a helpful assistant."
openai_max_tokens = 1500
openai_temperature = 0.7

# Mistral settings
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
mistral_url = "https://api.mistral.ai/v1/chat/completions"
mistral_model = "open-mistral-nemo" 
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

# Google settings
google_api_key = os.getenv('GOOGLE_API_KEY')
google_url = ""
google_model = "gemini-1.5-flash"
#google_model = "gemini-1.5-pro"
google_max_tokens = 1500
google_temperature = 1.0 # range from [0.0, 2.0] Use higher for more creative, lower for more deterministic.

# Perplexity settings
PPLX_API_KEY = os.getenv('PPLX_API_KEY')
#perplexity_model = "llama-3.1-sonar-small-128k-online" # small, large, or huge
perplexity_model = "llama-3.1-sonar-large-128k-online"
#perplexity_model = "llama-3.1-sonar-huge-128k-online"
perplexity_url = "https://api.perplexity.ai/chat/completions"
perplexity_system_prompt = "You are an artificial intelligence assistant and you need to engage in a helpful, detailed, polite conversation with the user."
perplexity_max_tokens = 1500
perplexity_temperature = 0.2 # 0-2, Defaults to 0.2. The amount of randomness in the response, valued between 0 inclusive and 2 exclusive. Higher values are more random, and lower values are more deterministic.
perplexity_return_citations = False

# Ollama model settings
ollama_url = "http://localhost:11434/api/generate"
keep_alive = "30s" # seconds
sleep_time = 2 # number of seconds
num_predict = 3000 # token cap

# File and directory settings
prompts_file = "prompts.json"
models_file = "models.json"
responses_dir = "responses"
summary_input_xls = "to_summarize.xlsx"
excel_engine = "openpyxl"
summary_excerpt_wordcount = 15
summary_category_name = "Comprehension and Summarization"