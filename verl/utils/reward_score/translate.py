import re
import json
import requests
from typing import Dict, Optional
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


# Replace with your actual COMET scoring endpoint
COMET_API_URL = "http://<internal_ip>:<port>/score"
# model_path = "/xxx/wmt20-comet-da/checkpoints/model.ckpt"

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return None

def evaluate_translation(text, target_sentence):
    target_language = detect_language(target_sentence)
    if not target_language:
        return False, "Could not detect the target language"

    detected_languages = set()
    for sentence in text.split('.'):
        language = detect_language(sentence)
        if language:
            detected_languages.add(language)

    if len(detected_languages) > 1:
        return False, "Detected multiple languages"

    if len(detected_languages) == 1 and target_language == detected_languages.pop():
        return True, "All text is in the target language"

    return False, f"Text is not in the target language. Detected: {detected_languages}"


def extract_solution(solution_str: str) -> (Optional[str], str):
    """Extract final answer from the model's response string."""
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "助手<think>" in solution_str:
        processed_str = solution_str.split("助手<think>", 1)[1]
        processed_str = "<think>" + processed_str
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parse predicted roles from answer string."""
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b.*?\b(knight|knave)\b',
            re.IGNORECASE
        )
        match = pattern.search(answer_text)

        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None

    return status_dict


def validate_response_structure(processed_str: str) -> bool:
    """Check whether the model response has valid <think> and <answer> tags in the right order."""
    print("\n[Structure Validation]")
    validation_passed = True

    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def compute_score(source_str: str,
                  solution_str: str,
                  ground_truth: Dict[str, str],
                  format_reward: int = 1) -> float:
    """Compute score for model output using format and external COMET scoring."""
    print("\n" + "=" * 80)
    print(" Processing New Sample ".center(80, '='))

    solution_text = ground_truth.get('target_text', '')
    print(f"[Ground Truth] {solution_text}")

    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    format_correct = validate_response_structure(processed_str)
    format_score = 0.2 if format_correct else 0

    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    answer_score = 0
    if format_correct and answer_text:
        data = [{
            "src": source_str,
            "mt": answer_text,
            "ref": solution_text
        }]

        headers = {"Content-Type": "application/json"}

        while True:
            try:
                score_dict = requests.post(
                    url=COMET_API_URL,
                    data=json.dumps(data),
                    headers=headers
                ).json()
                break
            except:
                print("Error sending data:\n", data)
                continue

        comet_score = float(score_dict.get("system_score", 0))
        comet_score = max(0, round(comet_score, 2))
        print(f"  COMET score: {comet_score}")
        answer_score = comet_score
    else:
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-" * 80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("=" * 80 + "\n")

    return total_score
