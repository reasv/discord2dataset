import os
from dotenv import load_dotenv
from clean import clean_all
from condense import condense_all
from prepare_datasets import compile_all
from split_conversations import split_all
load_dotenv()

if __name__ == "__main__":
    target_user_id = os.getenv("TARGET_USER_ID")
    assert target_user_id, "TARGET_USER_ID is not set"
    target_names = os.getenv("TARGET_NAMES", "").split(",")
    target_name_replacement = os.getenv("TARGET_NAME_REPLACEMENT", "assistant")

    clean_all(target_user_id, target_names, target_name_replacement)
    condense_all(target_name_replacement)
    split_all(target_name_replacement)
    compile_all(target_name_replacement)