def remove_duplicates(comma_separated_list):
    # Split the input string into a list
    items = comma_separated_list.split(', ')
    # Create a list to hold unique items
    unique_items = []
    # Loop through the items and append unique ones to unique_items
    for item in items:
        if item not in unique_items:
            unique_items.append(item)
    # Join the unique items back into a comma-separated string
    return ', '.join(unique_items)

input_string = 'long string, long string, longstring longstring, long long string'
output_string = remove_duplicates(input_string)
print(output_string)