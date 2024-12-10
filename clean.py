import json
import os
import re
from types import NoneType
from typing import Any, Dict, List
from dateutil import parser
from dotenv import load_dotenv
from generate_usernames import generate_username
from datetime import timedelta

load_dotenv()

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
        if reference:
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

def timedelta_from_iso(t1: str, t2: str):
    if t1 == None or t2 == None:
        return None
    return parser.parse(t1) - parser.parse(t2)

def condense_messages_by_author(input_file: str, output_file: str, assistant_author: str):
    with open(input_file, "rb") as f:
        raw = json.load(f)
    
    raw_messages: list[dict] = raw["messages"]
    condensed_messages = []
    condensed_message = []
    for message in raw_messages:
        if len(condensed_message) > 0 and message["author"] == condensed_message[0]["author"]:
            time_from_last_message = timedelta_from_iso(message["timestamp"], condensed_message[-1]["timestamp"])
            time_from_first_message = timedelta_from_iso(message["timestamp"], condensed_message[0]["timestamp"])
            if time_from_last_message < timedelta(minutes=1) or time_from_first_message < timedelta(minutes=10):
                length = len(message["content"])
                if length < 4096:
                    condensed_message.append(message)
                    continue
        
        if len(condensed_message) > 0:
            condensed_messages.append(condense_format_messages(condensed_message, is_assistant=message["author"] == assistant_author))
        condensed_message = [message]

    with open(output_file,'w') as f:
        json.dump(condensed_messages, f, indent=2)

quote = re.compile(r"\[\d\d:\d\d\] ?([\w\d \.]+):")
quote_broken1 = re.compile(r"\d\d:\d\d\] ?([\w\d \.]+):")
quote_ampm = re.compile(r"\[\d\d:\d\d (AM|PM)\] ?([\w\d \.]+):")

def clean_content(content: str) -> str:
    content = quote.sub(r"> ", content)
    content = quote_broken1.sub(r"> ", content)
    content = quote_ampm.sub(r"> ", content)
    return content

import re
from typing import List, Dict, Union

def mask_content(content: str, train_detail: List[Dict[str, Union[int, bool]]]) -> List[Dict[str, Union[int, bool]]]:
    # A regex pattern for URLs:
    # This pattern matches URLs with optional protocols, domain, and typical path/query structures.
    # It is a fairly robust URL regex that catches HTTP/HTTPS and many other common forms.
    url_pattern = re.compile(
        r'(https?://[^\s]+)'
    )

    result = []

    for segment in train_detail:
        begin = segment['begin_offset']
        end = segment['end_offset']
        is_train = segment['train']

        if not is_train:
            # If this segment is already train: false, just keep it as is.
            result.append(segment)
            continue

        # For train: true segments, find URLs within this segment
        segment_text = content[begin:end+1]
        urls = list(url_pattern.finditer(segment_text))

        if not urls:
            # No URLs in this segment, keep it as is.
            result.append(segment)
            continue

        # If there are URLs, we need to split the segment around them.
        current_start = begin

        for match in urls:
            url_start_in_segment, url_end_in_segment = match.span()
            # Convert these local (segment-based) offsets to absolute offsets:
            url_start = begin + url_start_in_segment
            url_end = begin + url_end_in_segment - 1  # inclusive index

            # Text before the URL (if any)
            if url_start > current_start:
                result.append({
                    'begin_offset': current_start,
                    'end_offset': url_start - 1,
                    'train': True
                })

            # The URL itself is train: false
            result.append({
                'begin_offset': url_start,
                'end_offset': url_end,
                'train': False
            })

            current_start = url_end + 1

        # After the last URL, if there's remaining text, keep it train: true
        if current_start <= end:
            result.append({
                'begin_offset': current_start,
                'end_offset': end,
                'train': True
            })

    # The result now contains segments with URLs masked out (train: false)
    # and all other text (train: true) as required.
    return result

def test_mask_content(content: str, train_detail: List[Dict[str, Union[int, bool]]]) -> None:
    """
    Tests the mask_content function by asserting that:
    - The output segments cover the entire content without gaps.
    - There are no zero-length ranges.
    - There are no overlapping ranges.

    Args:
        content (str): The input string representing a chat message.
        train_detail (List[Dict[str, Union[int, bool]]]): The list of segment dictionaries.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    output = mask_content(content, train_detail)

    # Ensure that output is not empty
    assert output, "The output from mask_content is empty."

    # Check that the first segment starts at 0
    assert output[0]['begin_offset'] == 0, f"First segment does not start at 0. Starts at {output[0]['begin_offset']}."

    # Check that the last segment ends at the last character of the content
    expected_last_end = len(content) - 1
    assert output[-1]['end_offset'] == expected_last_end, (
        f"Last segment does not end at {expected_last_end}. Ends at {output[-1]['end_offset']}."
    )

    previous_end = -1  # Initialize to -1 to start checking from 0

    for idx, segment in enumerate(output, start=1):
        begin = segment['begin_offset']
        end = segment['end_offset']
        is_train = segment['train']

        # Check for zero-length ranges
        assert begin <= end, (
            f"Segment {idx} has invalid range: begin_offset ({begin}) > end_offset ({end})."
        )

        # Check for coverage without gaps
        expected_begin = previous_end + 1
        assert begin == expected_begin, (
            f"Segment {idx} begins at {begin}, expected {expected_begin}, indicating a gap or overlap."
        )

        # Update previous_end for the next iteration
        previous_end = end

def condense_format_messages(messages: list[Dict[str, Any]], is_assistant: bool) -> Dict[str, str | bool | List[Dict[str, int | bool]]]:
    assert isinstance(messages[-1]["timestamp"], str), "Timestamp must be a string"
    assert isinstance(messages[0]["author"], str), "Author must be a string"
    formatted_message: Dict[str, str | bool | List[Dict[str, int | bool]]] = {
        "role": "assistant" if is_assistant else "human",
        "content": "" if is_assistant else f"{messages[0]['author']}: ",
        "timestamp": messages[-1]["timestamp"],
        "author": messages[0]["author"]
    }
    train_detail = []

    for message in messages:
        assert isinstance(message["content"], str), "Content must be a string"
        assert isinstance(formatted_message["content"], str), "Content must be a string"
        if message["reference"]:
            assert isinstance(message["reference"], dict), "Reference must be a dictionary"
            reference_content = message["reference"]["content"]
            assert isinstance(reference_content, str), "Reference content must be a string"
            formatted_reference = clean_content(reference_content).replace("\n", "\n> ") + "\n"
            reference_start_idx = len(formatted_message["content"])
            formatted_message["content"] += formatted_reference
            reference_end_idx = len(formatted_message["content"]) - 1
            train_detail.append({"begin_offset": reference_start_idx, "end_offset": reference_end_idx, "train": True})
        
        content_start_idx = len(formatted_message["content"])
        formatted_message["content"] += clean_content(message["content"]) + "\n"
        content_end_idx = len(formatted_message["content"]) - 1
        train_detail.append({"begin_offset": content_start_idx, "end_offset": content_end_idx, "train": True})
        if len(message["attachments"]) > 0:
            attachments = ("\n" + " ".join([a["filename"] for a in message["attachments"]])) if len(message["attachments"]) > 0 else ""
            attachments_start_idx = len(formatted_message["content"])
            formatted_message["content"] += attachments
            attachments_end_idx = len(formatted_message["content"]) - 1
            train_detail.append({"begin_offset": attachments_start_idx, "end_offset": attachments_end_idx, "train": False})
        
        assert isinstance(message["embeds"], list), "Embeds must be a list"
        if len(message["embeds"]) > 0:
            formatted_embeds = ""
            for embed in message["embeds"]:
                assert isinstance(embed, dict), "Embed must be a dictionary"
                assert isinstance(embed["title"], str), "Embed title must be a string"
                assert isinstance(embed["description"], str), "Embed description must be a string"
                if len(embed["title"]) == 0 and len(embed["description"]) == 0:
                    continue
                formatted_embeds += f"\n{embed['url']}\n{embed['title']}\n{embed['description']}"
            if len(formatted_embeds) > 0:
                embeds_start_idx = len(formatted_message["content"])
                formatted_message["content"] += formatted_embeds
                embeds_end_idx = len(formatted_message["content"]) - 1
                train_detail.append({"begin_offset": embeds_start_idx, "end_offset": embeds_end_idx, "train": False})

    if not is_assistant:
        formatted_message["training"] = False
    else:
        assert isinstance(formatted_message["content"], str), "Content must be a string"
        test_mask_content(formatted_message["content"], train_detail)
        train_detail = mask_content(formatted_message["content"], train_detail)
        test_mask_content(formatted_message["content"], train_detail)
        formatted_message["training_detail"] = train_detail

    return formatted_message
    

if __name__ == "__main__":
    target_user_id = os.getenv("TARGET_USER_ID")
    assert target_user_id, "TARGET_USER_ID is not set"
    target_names = os.getenv("TARGET_NAMES", "").split(",")
    target_name_replacement = os.getenv("TARGET_NAME_REPLACEMENT", "assistant")

    clean_all(target_user_id, target_names, target_name_replacement)
    condense_messages_by_author("tmp/cleaned/source.json", "tmp/condensed/source.json", target_name_replacement)