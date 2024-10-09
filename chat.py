#!/usr/bin/env python3
# chat.py - handles chat mode via the Ollama CLI

"""
Module Docstring
"""

__author__ = "Alex Bishop"
__version__ = "0.3.0"
__license__ = "MIT"

import os
import subprocess
from models import load_models

def start_ollama_chat(model_name):
    """Starts an Ollama chat session via terminal command."""
    try:
        # Build the command for the Ollama chat session
        chat_command = f"docker exec -it ollama ollama run {model_name}"
        print(f"Starting Ollama chat with model: {model_name}")
        
        # Open the Ollama chat via a subprocess (in terminal)
        process = subprocess.Popen(chat_command, shell=True)

        # Wait for the Ollama chat process to end
        process.communicate()

        # Detect if the process ends (i.e., '/bye' command ends the chat)
        if process.returncode == 0:
            print("Ollama chat session ended.")
            return True
        else:
            print(f"Ollama chat encountered an issue with model: {model_name}")
            return False
    except Exception as e:
        print(f"Error starting Ollama chat: {str(e)}")
        return False

def ollama_chat_mode():
    """Initiates the chat mode using an Ollama model."""
    print("\n💬 Welcome to the Ollama Chat Mode 💬\n")

    # Load available models from models.json
    models = load_models()

    # Filter only models hosted on the droplet (location = "droplet")
    droplet_models = [model['name'] for model in models if model['location'] == 'droplet']

    if not droplet_models:
        print("No Ollama models found on the droplet.")
        return

    # Display the models for selection
    print("Select an Ollama model to chat with:")
    for idx, model_name in enumerate(droplet_models, 1):
        print(f"{idx}. {model_name}")

    # Get the user's selection
    try:
        selection = int(input("\nEnter the number of the model to chat with: ")) - 1
        if 0 <= selection < len(droplet_models):
            selected_model = droplet_models[selection]
            # Start the Ollama chat session with the selected model
            chat_ended = start_ollama_chat(selected_model)

            # After chat session ends, return to main menu
            if chat_ended:
                print("Returning to the main menu.")
            else:
                print("An issue occurred with the chat session. Returning to the main menu.")
        else:
            print("Invalid selection. Returning to the main menu.")
    except ValueError:
        print("Invalid input. Returning to the main menu.")