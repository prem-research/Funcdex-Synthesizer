import json
from config import config

input_path = config.paths.output.scored_conversations
output_path = config.paths.output.accepted_conversations
min_quality = config.scoring.min_tool_use_quality

conversations = []

print(f"Loading scored conversations from {input_path}...")
with open(input_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        tool_use_quality = int(data["scores"]["tool_use_quality"])
        if tool_use_quality >= min_quality:
            conversations.append(line)

print(f"Found {len(conversations)} conversations with tool_use_quality >= {min_quality}")

print(f"Writing accepted conversations to {output_path}...")
with open(output_path, 'w') as f:
    for conversation in conversations:
        f.write(conversation)

print(f"Done! Wrote {len(conversations)} high-quality conversations.")
