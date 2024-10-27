import sys
from process_text import get_term_match

def main():
    print("Welcome to Getty Vocabulary Tester")
    while True:
        term = input("\nEnter a term to search (or type 'exit' to quit): ").strip()
        if term.lower() == "exit":
            break
        
        print("\nSelect Vocabulary:")
        print("1. AAT")
        print("2. ULAN")
        print("3. TGN")
        vocab_choice = input("\nEnter choice (1-3): ").strip()
        
        vocab_map = {"1": "AAT", "2": "ULAN", "3": "TGN"}
        vocabulary = vocab_map.get(vocab_choice, "AAT")
        
        print(f"\nSearching '{term}' in {vocabulary}...\n")
        results = get_term_match(vocabulary, term, notes="object")

        # Display results in a structured way
        if results:
            print("\nResults:")
            for idx, result in enumerate(results, start=1):
                print(f"\nResult {idx}:")
                print(f" - Term: {result['term']}")
                print(f" - Subject ID: {result['subject_id']}")
                print(f" - Hierarchy: {result['hierarchy']}")
        else:
            print(f"\nNo results found for '{term}' in {vocabulary}.")

if __name__ == "__main__":
    main()