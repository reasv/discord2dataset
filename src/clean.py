import json
import os
import re
from types import NoneType
from typing import Any, Dict, List
from dateutil import parser
from generate_usernames import generate_username
from datetime import timedelta

def clean(input_file: str, output_file: str, target_user_id: str, target_names: list[str], target_name_replacement: str):
    with open(input_file, "rb") as f:
        raw = json.load(f)
    
    raw_messages: list[dict] = raw["messages"]
    cleaned_messages = []
    # Contains all the various names found for the target user
    target_names_set = set(target_names)
    # Contains all the various names found for each user
    other_user_names: Dict[str, set] = dict()
    # Mapping of user id to substituted name
    other_user_names_sub: Dict[str, str] = dict()
    channel_id = raw["channel"]["id"]
    id_to_idx = {message["id"]: idx for idx, message in enumerate(raw_messages)}

    def substitute_names(content: str | None):
        if not content:
            return content
        for name in sorted(list(target_names_set), key=len, reverse=True):
            content = re.sub(re.escape(name), target_name_replacement, content, flags=re.IGNORECASE)
        for author_id, author_names in other_user_names.items():
            for name in sorted(list(author_names), key=len, reverse=True):
                content = re.sub(re.escape(name), other_user_names_sub[author_id], content, flags=re.IGNORECASE)
        return content

    for message in raw_messages:
        reference = None
        if message.get("reference") and message["reference"]["channelId"] == channel_id:
            reference = message["reference"]["messageId"]

        if message["author"]["id"] == target_user_id:
            author = target_name_replacement
            target_names_set.add(message["author"]["name"])
            target_names_set.add(message["author"]["nickname"])
            
        else:
            author_id = message["author"]["id"]
            if author_id not in other_user_names:
                other_user_names[author_id] = set()
                other_user_names_sub[author_id] = generate_username()
            author = other_user_names_sub[author_id]
            other_user_names[author_id].add(message["author"]["name"])
            other_user_names[author_id].add(message["author"]["nickname"])

        reference_message = None
        if reference and reference in id_to_idx:
            reference_message = cleaned_messages[id_to_idx[reference]]
        
        cleaned_messages.append({
            "id": message["id"],
            "author": author,
            "content": substitute_names(message["content"]),
            "timestamp": message["timestamp"],
            "embeds": [{"url": substitute_names(embed["url"]), "title": substitute_names(embed["title"]), "description": substitute_names(embed["description"])} for embed in message["embeds"]],
            "attachments": [{"filename": substitute_names(a["fileName"])} for a in message["attachments"]],
            "reference": reference_message
        })

    with open(output_file,'w') as f:
        json.dump({"messages": cleaned_messages}, f, indent=2)

    print(f"Names of target: {target_names_set} names of other: {other_user_names}")

def clean_all(target_user_id: str, target_names: list[str], target_name_replacement: str):
    source_dir = "source/"
    output_dir = "tmp/cleaned/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            input_file = os.path.join(source_dir, filename)
            output_file = os.path.join(output_dir, filename)
            clean(input_file, output_file, target_user_id, target_names, target_name_replacement)