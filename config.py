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
enter = "{BOLD_EFFECT}Enter{RESET_STYLE}"
msg_user_nudge = "{RESPONSE_COLOR}{BOLD_EFFECT}→ {RESET_STYLE}"
msg_farewell = "Bye now! 👋🏻 {RESET_STYLE}"
msg_invalid_returning = "Invalid option ❌ Returning to the main menu. 🤖"
msg_invalid_number = "Invalid input ❌ Please enter a number."
msg_invalid_retry = "Invalid selection ❌ Please enter a valid option {RESPONSE_COLOR}{BOLD_EFFECT}→{RESET_STYLE}"
msg_error_response_prompt = "❗ Error generating response for prompt '{prompt}': {e}"
msg_error_response = "Error generating response: {e}"
msg_select_prompt = "\n{RESPONSE_COLOR}{BOLD_EFFECT}→{RESET_STYLE} Select a prompt:"
msg_select_category = "\n{RESPONSE_COLOR}{BOLD_EFFECT}→{RESET_STYLE} Select a prompt category:"
msg_select_prompts_cat = "\nSelect prompts from the category '{CATEGORY_COLOR}{selected_category}{RESET_STYLE}':"
msg_select_confirm = "You have selected:\n- {PROMPT_COLOR}{prompt}{RESET_STYLE}"
msg_enter_prompt_number = "{RESPONSE_COLOR}{BOLD_EFFECT}→{RESET_STYLE} Enter the number of the prompt you want to use: "
msg_generatingmsg_ = "\n🔄 Generating response for model {BOLD_EFFECT}{MODEL_COLOR}{model_name}{RESET_STYLE} with prompt: {PROMPT_COLOR}{prompt}{RESET_STYLE}"
msg_invalid_response = "No valid response generated."
msg_continue_model = "\n{CONFIRM_COLOR}{RESPONSE_COLOR}{BOLD_EFFECT}→{RESET_STYLE} Do you want to continue with{RESET_STYLE} {BOLD_EFFECT}{MODEL_COLOR}{selected_model}{RESET_STYLE} {CONFIRM_COLOR}or select a different model? {BOLD_EFFECT}{STATS_COLOR}(y/n){RESET_STYLE}: {RESET_STYLE}"
msg_enter_selection = "{RESPONSE_COLOR}{BOLD_EFFECT}→{RESET_STYLE} Enter your selection: "
msg_no_prompts = "No prompts selected."
msg_no_category = "No category selected."
msg_prompt_quantity = "Enter the number of times to send each prompt (1-10): "
msg_number_1_10 = "{RESPONSE_COLOR}{BOLD_EFFECT}→{RESET_STYLE} Please enter a number between 1 and 10."
msg_no_missing_prompts = "No missing prompts for this model."
msg_no_summary_prompts = "No prompts found for 'Comprehension and Summarization'. Please check your {BOLD_EFFECT}prompts.json{RESET_STYLE} file."
msg_select_summary_prompt = "\n{RESPONSE_COLOR}{BOLD_EFFECT}→{RESET_STYLE} Select a summarization prompt:"
msg_no_matching = "No matching responses found."
msg_no_resp_processing = "Response processing not implemented for this model."

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