import json
import os

input_file = "output/output.json"
base_name = os.path.splitext(input_file)[0]
output_file = base_name + ".jsonl"

# Load the JSON file
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as out_f:
    for conv_id, conv_data in data.items():
        conversation = []

        # Each entry has "query" and "response" — turn them into "role" + "content" pairs
        for turn in conv_data["conversation"]:
            if "query" in turn:
                conversation.append({"role": "human", "content": turn["query"]})
            if "response" in turn:
                conversation.append({"role": "bot", "content": turn["response"]})

        record = {
            "conversation_id": conv_id,
            "conversation": conversation
        }

        # Write as one line per conversation
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Successfully converted {input_file} to {output_file}")
