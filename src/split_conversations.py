import json
import os
from datetime import timedelta

from condense import timedelta_from_iso

def split_conversations(input_file: str, output_file: str, assistant_author: str):
    with open(input_file, "rb") as f:
        raw = json.load(f)
    
    raw_messages: list[dict] = raw["messages"]
    conversations = []
    next_index = 0
    while next_index < len(raw_messages):
        start, end = find_conversation_idxes(next_index, assistant_author, raw_messages)
        if start is None or end is None:
            print(f"Could not find any more conversations after index {next_index}")
            break
        conversation = raw_messages[start:end + 1]
        if conversation_has_training(conversation):
            convo_no_tail = remove_conversation_tail(conversation)
            if len(convo_no_tail) < len(conversation):
                print(f"Removed tail of conversation at [{start}:{end}] (length: {end-start} new length: {len(convo_no_tail)}) to remove non-training messages (removed: {len(conversation) - len(convo_no_tail)})")
            assert len(convo_no_tail) > 0, f"Conversation at [{start}:{end}] (length: {end-start}) has no training messages"
            conversations.append(convo_no_tail)
        else:
            print(f"Skipping conversation at [{start}:{end}] (length: {end-start}) because it does not contain any training messages")
        next_index = end + 1

    with open(output_file, "w") as f:
        json.dump({"conversations": conversations}, f, indent=2)

def split_all(assistant_author_name: str):
    source_dir = "tmp/condensed/"
    output_dir = "tmp/conversations/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            input_file = os.path.join(source_dir, filename)
            output_file = os.path.join(output_dir, filename)
            split_conversations(input_file, output_file, assistant_author_name)

def find_conversation_idxes(first_index: int, assistant_author: str, messages: list[dict]):
    # Constants
    min_context = 8
    max_context = 16
    max_context_lookback_time = timedelta(hours=2)
    max_conversation_gap = timedelta(hours=1)
    max_conversation_length = 50

    # Find the first message that is from the assistant
    start = first_index
    for message in messages[first_index:]:
        if message["author"] == assistant_author:
            break
        start += 1
    if start == len(messages):
        return None, None

    first_assistant_idx = start
    first_assistant_time = messages[first_assistant_idx]["timestamp"]
    
    # Include max_context messages before the assistant message
    start = max(0, start - max_context)
    # Remove messages that are too far back in time, but keep at least min_context messages
    # Use timedelta_from_iso(message_time, first_assistant_time) to get the time difference
    for i in range(start, first_assistant_idx - min_context):
        if timedelta_from_iso(first_assistant_time, messages[i]["timestamp"]) > max_context_lookback_time:
            start = i + 1
    
    # Find the last assistant message before a gap of max_conversation_gap since the last assistant message
    end = first_assistant_idx
    last_assistant_time = first_assistant_time
    for message in messages[first_assistant_idx:]:
        if timedelta_from_iso(message["timestamp"], last_assistant_time) > max_conversation_gap:
            print(f"Conversation at [{start}:{end}] (length: {end-start}) terminated after time gap: {timedelta_from_iso(message['timestamp'], last_assistant_time)}")
            break
        else:
            if message["author"] == assistant_author:
                last_assistant_time = message["timestamp"]
                end += 1
        # End conversation after max_conversation_length messages
        if end - start >= max_conversation_length:
            print(f"Conversation at [{start}:{end}] (length: {end-start}) terminated after reaching max length")
            break

    return start, end

def conversation_has_training(conversation: list[dict]):
    """
    Check if a conversation has any training messages.
    A conversation has training messages if:
    - Any message has the "training" key set to True
    - Any message has a "training_detail" key with a list of training details, where at least one detail has the "train" key set to True
    """
    for message in conversation:
        if message.get("training") == True:
            return True
        else:
            for train_detail in message.get("training_detail", []):
                if train_detail["train"] == True:
                    return True
    return False

def remove_conversation_tail(conversation: list[dict]):
    """
    Remove the tail of a conversation that is not relevant for training.
    The tail is defined as the last messages in the conversation that
    do not have the "training" key set to True or a "training_detail" key
    with a list of training details, where at least one detail has the "train" key set to True.
    """
    for idx, message in enumerate(reversed(conversation)):
        if message.get("training") == True:
            # If idx is 0, return the full conversation
            return conversation[:-idx] if idx != 0 else conversation
        else:
            for train_detail in message.get("training_detail", []):
                if train_detail.get("train") == True:
                    return conversation[:-idx] if idx != 0 else conversation
    return conversation
