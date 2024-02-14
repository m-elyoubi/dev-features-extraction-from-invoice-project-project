import re

text = """Account summary
Amount due on Bill 7015
$400.00"""

# Define a regular expression pattern to match the amount due on Bill 7015
pattern = r"Amount due on Bill 7015[\s\S]*?\$([\d,]+\.\d{2})"

# Use re.search to find the match
match = re.search(pattern, text)

if match:
    amount_due = match.group(1)
    print("Amount due on Bill 7015:", amount_due)
else:
    print("Amount due on Bill 7015 not found in the text.")
