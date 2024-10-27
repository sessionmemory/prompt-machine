import sys
from process_text import get_term_match

def parse_vocabulary_choice(choice):
    """
    Parse the user's vocabulary choice input, allowing for formats like '1,3' or '1-3'.
    Returns a list of selected vocabulary strings (AAT, ULAN, TGN).
    """
    vocab_map = {"1": "AAT", "2": "ULAN", "3": "TGN"}
    selected_vocab = set()

    # Split by commas and handle ranges like 1-3
    parts = choice.split(',')
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            selected_vocab.update(vocab_map[str(i)] for i in range(start, end + 1))
        else:
            selected_vocab.add(vocab_map.get(part.strip(), "AAT"))

    return list(selected_vocab)

def main():
    print("Welcome to Getty Vocabulary Tester")
    while True:
        term = input("\nEnter a term to search (or type 'exit' to quit): ").strip()
        if term.lower() == "exit":
            break

        print("\nSelect Vocabulary (You can enter multiple selections, e.g., 1,3 or 1-3):")
        print("1. Art & Architecture Thesaurus (AAT)")
        print("2. Union List of Artist Names (ULAN)")
        print("3. Getty Thesaurus of Geographic Names (TGN)")
        vocab_choice = input("\nEnter choice(s) (e.g., 1-3, 1, 2,3): ").strip()

        vocabularies = parse_vocabulary_choice(vocab_choice)
        print(f"\nSearching '{term}' in {', '.join(vocabularies)}...\n")

        for vocabulary in vocabularies:
            print(f"\n--- Searching {vocabulary} ---")
            results = get_term_match(vocabulary, term)

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