import json
import time
from litellm import completion
from tqdm import tqdm
from joblib import Parallel, delayed
from pydantic import BaseModel
from json_repair import repair_json
import os
from config import config

api_key = config.llm_api.scorer.api_key
model = config.llm_api.scorer.model
base_url = config.llm_api.scorer.base_url

def load_prompt(prompt_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", f"{prompt_name}.txt")
    with open(prompt_path, 'r') as f:
        return f.read()

print("Loading prompts...")
scorer_prompt = load_prompt("conversation_scorer")
print("Prompts loaded successfully!")


class ConversationScore(BaseModel):
    reasoning: str
    user_query_clarity: int
    tool_use_quality: int


def completion_with_retry(**kwargs):
    max_retries = config.generation.correction.max_retries
    retry_delay = config.generation.correction.retry_delay
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = completion(timeout=config.scoring.timeout, **kwargs)
            return response
        except Exception as e:
            error_str = str(e)
            retry_count += 1
            
            if retry_count >= max_retries:
                print(f"Max retries reached. Last error: {error_str[:100]}")
                raise
            
            print(f"LLM API call failed (attempt {retry_count}/{max_retries}): {error_str[:100]}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)


def score_single_conversation(conversation_data: dict) -> dict:
    try:
        metadata = conversation_data.get("metadata", {})
        scenario = metadata.get("scenario", "N/A")
        user_persona = metadata.get("user_persona", "N/A")
        disposition = metadata.get("disposition", "N/A")
        conversation_dml = conversation_data.get("conversation_dml", "")
        
        prompt = scorer_prompt.format(
            scenario=scenario,
            user_persona=user_persona,
            disposition=disposition,
            conversation_dml=conversation_dml
        )
        
        response = completion_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            base_url=base_url,
            temperature=config.scoring.temperature,
            response_format=ConversationScore,
            reasoning={
                "effort": config.scoring.reasoning_effort,
            }
        )
        
        score_content = response.choices[0].message.content.strip()
        scores = json.loads(repair_json(score_content))
        
        result = conversation_data.copy()
        result["scores"] = {
            "reasoning": scores.get("reasoning", ""),
            "user_query_clarity": scores.get("user_query_clarity", -1),
            "tool_use_quality": scores.get("tool_use_quality", -1)
        }
        
        print(f"✓ Scored conversation {conversation_data.get('id', 'unknown')} - "
              f"Clarity: {scores['user_query_clarity']}, Tool Use: {scores['tool_use_quality']}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error scoring conversation {conversation_data.get('id', 'unknown')}: {str(e)[:100]}")
        result = conversation_data.copy()
        result["scores"] = {
            "error": str(e),
            "user_query_clarity": 0,
            "tool_use_quality": 0
        }
        return result


def score_conversations(
    input_file: str = None,
    output_file: str = None,
    n_jobs: int = None,
    limit: int = None
):
    if input_file is None:
        input_file = config.paths.output.conversations_with_system_prompts
    if output_file is None:
        output_file = config.paths.output.scored_conversations
    if n_jobs is None:
        n_jobs = config.processing.scoring.n_jobs
    print(f"Loading conversations from {input_file}...")
    
    conversations = []
    with open(input_file, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
            if limit and len(conversations) >= limit:
                break
    
    print(f"Loaded {len(conversations)} conversations")
    print(f"Scoring with {n_jobs} parallel workers using {model}...")
    print(f"Output will be saved to {output_file}")
    print("=" * 80)
    
    scored_conversations = Parallel(n_jobs=n_jobs)(
        delayed(score_single_conversation)(conv)
        for conv in tqdm(conversations, desc="Scoring conversations")
    )
    
    print(f"\nSaving scored conversations to {output_file}...")
    with open(output_file, 'w') as f:
        for scored_conv in scored_conversations:
            f.write(json.dumps(scored_conv) + "\n")
    
    print("=" * 80)
    print("SCORING SUMMARY")
    print("=" * 80)
    
    successful_scores = [
        conv for conv in scored_conversations 
        if "error" not in conv.get("scores", {})
    ]
    
    if successful_scores:
        clarity_scores = [conv["scores"]["user_query_clarity"] for conv in successful_scores]
        tool_scores = [conv["scores"]["tool_use_quality"] for conv in successful_scores]
        
        print(f"Successfully scored: {len(successful_scores)}/{len(scored_conversations)}")
        print(f"Failed to score: {len(scored_conversations) - len(successful_scores)}")
        print()
        print(f"User Query Clarity - Avg: {sum(clarity_scores)/len(clarity_scores):.2f}, "
              f"Min: {min(clarity_scores)}, Max: {max(clarity_scores)}")
        print(f"Tool Use Quality   - Avg: {sum(tool_scores)/len(tool_scores):.2f}, "
              f"Min: {min(tool_scores)}, Max: {max(tool_scores)}")
        
        print("\nUser Query Clarity Distribution:")
        for score in range(1, 6):
            count = clarity_scores.count(score)
            pct = (count / len(clarity_scores)) * 100
            print(f"  Score {score}: {count:4d} ({pct:5.1f}%)")
        
        print("\nTool Use Quality Distribution:")
        for score in range(1, 6):
            count = tool_scores.count(score)
            pct = (count / len(tool_scores)) * 100
            print(f"  Score {score}: {count:4d} ({pct:5.1f}%)")
    else:
        print("No conversations were successfully scored!")
    
    print("=" * 80)
    print(f"Complete! Scored conversations saved to {output_file}")


if __name__ == "__main__":
    score_conversations()
