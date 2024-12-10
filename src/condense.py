import json
import os
import re
from typing import Any, Dict, List, Union
from dateutil import parser
from datetime import timedelta

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
            condensed_messages.append(condense_format_messages(condensed_message, is_assistant=condensed_message[0]["author"] == assistant_author))
        condensed_message = [message]

    with open(output_file,'w') as f:
        json.dump({"messages": condensed_messages}, f, indent=2)

quote = re.compile(r"\[\d\d:\d\d\] ?([\w\d \.]+):")
quote_broken1 = re.compile(r"\d\d:\d\d\] ?([\w\d \.]+):")
quote_ampm = re.compile(r"\[\d\d:\d\d (AM|PM)\] ?([\w\d \.]+):")

def clean_content(content: str) -> str:
    content = quote.sub(r"> ", content)
    content = quote_broken1.sub(r"> ", content)
    content = quote_ampm.sub(r"> ", content)
    return content



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

def ensure_substantial_content( 
    content: str,
    train_detail: List[Dict[str, Union[int, bool]]]
) -> List[Dict[str, Union[int, bool]]]:
    """
    Ensures that the remaining content after masking is substantial.
    Additionally removes instances of @username before assessing substantiality.
    If the concatenated train: true segments (after removing @username and whitespace) 
    are fewer than 3 characters, mask the entire content as train: false.

    Args:
        content (str): The input chat message.
        train_detail (List[Dict[str, Union[int, bool]]]): 
            List of segment dictionaries with 'begin_offset', 'end_offset', and 'train' keys.

    Returns:
        List[Dict[str, Union[int, bool]]]: 
            Modified list of segments. Either unchanged if substantial, 
            or a single segment masking the entire content.
    """
    # Compile a regex pattern to identify @username instances
    at_username_pattern = re.compile(r'@[^@\s]+')
    
    # Collect all train: true segments' text
    train_true_texts = []
    for segment in train_detail:
        if segment.get('train', False):
            begin = segment['begin_offset']
            end = segment['end_offset']
            # Extract substring; since end_offset is inclusive, add 1
            substring = content[begin:end + 1]
            # Remove all @username instances from the substring
            cleaned_substring = at_username_pattern.sub('', substring)
            train_true_texts.append(cleaned_substring)
    
    # Concatenate all cleaned train: true segments
    concatenated_text = ''.join(train_true_texts)
    
    # Remove all whitespace characters using regex for thoroughness
    stripped_text = re.sub(r'\s+', '', concatenated_text)
    
    # Check if the remaining string has at least 3 characters
    if len(stripped_text) >= 3:
        # Content is substantial; return train_detail unchanged
        return train_detail
    else:
        # Content is unsubstantial; mask the entire message
        if not content:
            # Handle empty content by returning an empty list
            return []
        return [{
            'begin_offset': 0,
            'end_offset': len(content) - 1,
            'train': False
        }]

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
        train_detail = ensure_substantial_content(formatted_message["content"], train_detail)
        test_mask_content(formatted_message["content"], train_detail)
        formatted_message["training_detail"] = train_detail

    return formatted_message

def condense_all(assistant_author_name: str):
    source_dir = "tmp/cleaned/"
    output_dir = "tmp/condensed/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            input_file = os.path.join(source_dir, filename)
            output_file = os.path.join(output_dir, filename)
            condense_messages_by_author(input_file, output_file, assistant_author_name)