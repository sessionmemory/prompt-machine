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
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_api_key_here')
openai_system_prompt = "You are a helpful assistant."
openai_max_tokens = 1024
openai_temperature = 0.7

# Mistral settings
mistral_model = "mistral-large-latest"
mistral_max_tokens = 1024
mistral_temperature = 0.7

# Anthropic settings
claude_model = "claude-3-5-sonnet-20240620"
claude_max_tokens = 1024

# Google settings
google_model = "gemini-1.5-flash"
google_max_tokens = 1024
google_temperature = 1.0

# Ollama model settings
keep_alive = "30s" # seconds
sleep_time = 1  # number of seconds

# File and directory settings
prompts_file = "prompts.json"
models_file = "models.json"
responses_dir = "responses"
summary_input_xls = "to_summarize.xlsx"