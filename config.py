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
RESET_STYLE = "\033[0m"  # Reset to default

# OpenAI settings
# Attempt to get the OPENAI_API_KEY from environment variables, or use a default/fallback value
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_url = "https://api.openai.com/v1/chat/completions"
openai_system_prompt = "You are a helpful assistant."
openai_max_tokens = 1024
openai_temperature = 0.7

# Mistral settings
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
mistral_url = "https://api.mistral.ai/v1/chat/completions"
mistral_model = "mistral-small-latest"
mistral_max_tokens = 1024
mistral_min_tokens = 20
mistral_temperature = 0.7

# Anthropic settings
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
claude_url = "https://api.anthropic.com/v1/messages"
claude_model = "claude-3-5-sonnet-20240620"
claude_max_tokens = 1024
claude_temperature = 1.0

# Google settings
google_api_key = os.getenv('GOOGLE_API_KEY')
google_url = ""
google_model = "gemini-1.5-flash"
google_max_tokens = 1024
google_temperature = 1.0

# Perplexity settings
PPLX_API_KEY = os.getenv('PPLX_API_KEY')
perplexity_model = "llama-3.1-sonar-small-128k-online" # small, large, or huge
perplexity_url = "https://api.perplexity.ai"
perplexity_system_prompt = "You are an artificial intelligence assistant and you need to engage in a helpful, detailed, polite conversation with the user."
perplexity_max_tokens = 1024
perplexity_temperature = 0.2
perplexity_return_citations = False

# Ollama model settings
ollama_url = "http://localhost:11434/api/generate"
keep_alive = "30s" # seconds
sleep_time = 1  # number of seconds

# File and directory settings
prompts_file = "prompts.json"
models_file = "models.json"
responses_dir = "responses"
summary_input_xls = "to_summarize.xlsx"