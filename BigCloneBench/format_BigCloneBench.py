import json

# open a format_data.txt file to write
with open("format_data.txt", "w", encoding="utf-8") as out:
    with open("data.jsonl", encoding="utf-8") as f:
        for line in f:
            line_data = json.loads(line)
            # write the func to the format_data.txt with a <s> and Class temp { } </s> tag
            out.write("<s>" + "class temp {" + line_data["func"] + "} </s>\n")
