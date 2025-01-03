import json
import os
from typing import Dict
import jsonlines

from generate_usernames import generate_username
def compile_all(assistant_author_name: str):
    source_dir = "tmp/conversations/"
    output_dir = "train/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, "dataset.jsonl")
    total_conversations, total_messages, max_messages, min_messages = 0, 0, 0, 1000000
    with jsonlines.open(output_file, "w") as writer:
        for filename in os.listdir(source_dir):
            if filename.endswith(".json"):
                input_file = os.path.join(source_dir, filename)
                with open(input_file, "rb") as f:
                    raw = json.load(f)
                conversations: list[list[dict]] = raw["conversations"]
                for conversation in conversations:
                    total_conversations += 1
                    formatted_conversation = format_conversation(conversation, assistant_author_name)
                    total_messages += len(formatted_conversation["conversations"])
                    max_messages = max(max_messages, len(formatted_conversation["conversations"]))
                    min_messages = min(min_messages, len(formatted_conversation["conversations"]))
                    writer.write(formatted_conversation)
    
    print(f"Compiled {total_conversations} conversations with {total_messages} messages. Max messages in a conversation: {max_messages}, Min messages in a conversation: {min_messages}. Average messages in a conversation: {total_messages / total_conversations}")

def format_conversation(conversation: list[dict], assistant_author: str):
    # For each conversation, we want to replace the usernames for the non-assistant turns with randomly generated names
    author_map: Dict[str, str] = {}
    for message in conversation:
        author = message["author"]
        if author != assistant_author and author not in author_map:
            author_map[author] = generate_username()
    formatted_conversation = [{
        "role":"system",
        "content": f"This is a conversation between multiple users in an online chat. You are {assistant_author}. Reply to the conversation roleplaying as {assistant_author}. Never write messages for other users, only for {assistant_author}. Write a single chat message at a time. Always stay in character.",
        "training": False
    },
    { "role": "user", "content": "<Chat History>", "training": False },
    ]
    for message in conversation:
        assert isinstance(message["content"], str), "Content must be a string"
        # Replace every author name in the content
        for author, replacement in author_map.items():
            message["content"] = message["content"].replace(author, replacement)
        
        formatted_message = {
            "role": message["role"],
            "content": message["content"],
        }
        if "training_detail" in message:
            formatted_message["training_detail"] = message["training_detail"]
        if "training" in message:
            formatted_message["training"] = message["training"]
        formatted_conversation.append(formatted_message)

    return {"conversations": formatted_conversation}