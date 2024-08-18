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
MODEL_COLOR = "\033[34m"  # Blue
CATEGORY_COLOR = "\033[36m" # Cyan
PROMPT_COLOR = "\033[95m"  # Magenta
BOLD_EFFECT = "\033[1m" # BOLD
RESPONSE_COLOR = "\033[32m" # Green
CONFIRM_COLOR = "\033[97m" # Bright White
STATS_COLOR = "\033[33m" # Yellow
BLINK_EFFECT = "\033[5m"  # Blink
RESET_STYLE = "\033[0m"  # Reset to default
ERROR_COLOR = {BOLD_EFFECT}

# Emojis
emoji_generating = "🔄 "
emoji_error = "❌ "
emoji_alert = "❗ "
emoji_question = "❓ "
emoji_bye = "✌🏻 "
emoji_menu_main = "🤖 "
emoji_menu1_single = "1️⃣ "
emoji_menu2_prompt = "💬 "
emoji_menu3_category = "💼 "
emoji_menu4_all = "🎢 "
emoji_menu5_unsent = "📫 "
emoji_menu6_summary = "📠 "
emoji_menu7_query = "🗄️ "
emoji_menu8_exit = "💨 "
emoji_menu_back = "🔙 "
emoji_user_nudge = "→ "
emoji_number = "🔢 "
emoji_done = "✅ "

# Basic Commands and Words
msg_word_enter = "{BOLD_EFFECT}{STATS_COLOR}Enter{RESET_STYLE}"
msg_word_select = "{BOLD_EFFECT}{STATS_COLOR}Select{RESET_STYLE}"
msg_word_error = "{BOLD_EFFECT}{ERROR_COLOR}Error{RESET_STYLE}"
msg_word_invalid = "{BOLD_EFFECT}{ERROR_COLOR}Invalid{RESET_STYLE}"
msg_word_model = "{BOLD_EFFECT}{MODEL_COLOR}model{RESET_STYLE}"
msg_word_category = "{BOLD_EFFECT}{CATEGORY_COLOR}category{RESET_STYLE}"
msg_word_prompt = "{BOLD_EFFECT}{PROMPT_COLOR}prompt{RESET_STYLE}"
msg_word_number = "{BOLD_EFFECT}number{RESET_STYLE}"
yes_or_no = "{BOLD_EFFECT}{STATS_COLOR}(y/n){RESET_STYLE}"

# User Interaction and Prompts
msg_user_nudge = "{RESPONSE_COLOR}{BOLD_EFFECT}{emoji_user_nudge} {RESET_STYLE}"
msg_select_confirm = "\n{msg_user_nudge}{STATS_COLOR}You have selected:{RESET_STYLE}\n- {PROMPT_COLOR}{prompt}{RESET_STYLE}"
msg_enter_selection = "\n{msg_user_nudge}{STATS_COLOR}Please {msg_word_enter} your selection: {RESET_STYLE}"
msg_select_summary_prompt = "\n{msg_user_nudge}{STATS_COLOR}{msg_word_select} a summarization {RESET_STYLE}{msg_word_prompt}:"
msg_use_same_model = "\n{msg_user_nudge}Do you want to continue with{RESET_STYLE} {BOLD_EFFECT}{MODEL_COLOR}{selected_model}{RESET_STYLE} {CONFIRM_COLOR}or select a different model? {BOLD_EFFECT}{STATS_COLOR}(y/n){RESET_STYLE}: {RESET_STYLE}"

# Model and Processing Related Messages
msg_generating_msg = "\n{emoji_generating} Generating response for {msg_word_model} {BOLD_EFFECT}{MODEL_COLOR}{model_name}{RESET_STYLE} with {msg_word_prompt}: {PROMPT_COLOR}{prompt}{RESET_STYLE}"
msg_content = "{RESPONSE_COLOR}{first_choice_content}{RESET_STYLE}"
msg_continue_model = "\n{msg_user_nudge}{CONFIRM_COLOR}Do you want to continue with{RESET_STYLE} {BOLD_EFFECT}{MODEL_COLOR}{selected_model}{RESET_STYLE} {CONFIRM_COLOR}or select a different {msg_word_model}? {yes_or_no}:"
msg_no_resp_processing = "\n{emoji_alert}Response processing was not implemented for this {msg_word_model}."

# Error and Invalid Inputs
msg_error_response = "{emoji_error}{msg_word_error} generating response for {msg_word_prompt} '{prompt}': {e}"
msg_invalid_response = "{emoji_error}No valid response generated."
msg_invalid_returning = "{emoji_error}{msg_word_invalid} option {emoji_error} Returning to the Main Menu!  {emoji_menu_main}"
msg_invalid_number = "{emoji_alert}{msg_word_invalid} input {emoji_error} Please enter a {msg_word_number} {emoji_number}."
msg_invalid_retry = "{emoji_alert}{msg_word_invalid} selection {emoji_error} Please enter a valid selection or type {STATS_COLOR}'q'{RESET_STYLE}. {msg_user_nudge}"
msg_no_matching = "{emoji_alert}No matching responses found. 🙁"

# No Selections or Errors Related to Selections
msg_no_prompts = "{emoji_alert}No {msg_word_prompt}s selected."
msg_no_category = "{emoji_alert}No {msg_word_category} selected."
msg_no_models = "{emoji_alert}No {msg_word_model}s selected."
msg_no_missing_prompts = "{emoji_alert}No missing {msg_word_prompt}s for this {msg_word_model}."
msg_no_summary_prompts = "{emoji_alert}No {msg_word_prompt}s found in {CATEGORY_COLOR}'Comprehension and Summarization'{RESET_STYLE}. Please check your {BOLD_EFFECT}{prompts_file}{RESET_STYLE} file."

# Farewell and Miscellaneous
msg_prompt_quantity = "{msg_user_nudge} {emoji_number} {msg_word_enter} the {msg_word_number} of times to send each {msg_word_prompt} (1-10): "
msg_number_1_10 = "{msg_user_nudge} {emoji_number} Please enter a {msg_word_number} between 1 and 10."
msg_farewell = "Bye now! {emoji_bye} {RESET_STYLE}"

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

# File and directory settings
prompts_file = "prompts.json"
models_file = "models.json"
responses_dir = "responses"
summary_input_xls = "to_summarize.xlsx"