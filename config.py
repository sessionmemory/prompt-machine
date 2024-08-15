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

# OpenAI settings
# Attempt to get the OPENAI_API_KEY from environment variables, or use a default/fallback value
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_api_key_here')
openai_system_prompt = "You are a helpful assistant."
openai_max_tokens = 1000
openai_temperature = 0.7

# Ollama model settings
keep_alive = "30s" # seconds
sleep_time = 1  # number of seconds

# File and directory settings
prompts_file = "prompts.json"
models_file = "models.json"
responses_dir = "responses"
summary_input_xls = "to_summarize.xlsx"