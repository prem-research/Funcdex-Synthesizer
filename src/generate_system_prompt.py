import json
import re
from litellm import completion
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import time
from config import config

def load_prompt(prompt_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", f"{prompt_name}.txt")
    with open(prompt_path, 'r') as f:
        return f.read()

print("Loading prompts...")
SYSTEM_PROMPT_TEMPLATE = load_prompt("system_prompt_generation")
print("Prompts loaded successfully!")

tool_defs = json.load(open(config.paths.data.tool_definitions))

tool_id_to_info = {}
for tool in tool_defs:
    tool_id_to_info[tool["tool_id"]] = {
        "description": tool["tool_description"],
        "toolkit_name": tool["toolkit_name"]
    }

api_key = config.llm_api.system_prompt.api_key
base_url = config.llm_api.system_prompt.base_url
model = config.llm_api.system_prompt.model


def extract_tool_names_from_dml(dml_text):
    tool_names = set()
    
    tool_pattern = r'<tool>\s*(\{.*?\})\s*</tool>'
    matches = re.findall(tool_pattern, dml_text, re.DOTALL)
    
    for match in matches:
        try:
            tool_call = json.loads(match)
            if "name" in tool_call:
                tool_names.add(tool_call["name"])
        except json.JSONDecodeError:
            continue
    
    return list(tool_names)

def get_tool_info_for_names(tool_names):
    tool_info = []
    for tool_name in tool_names:
        if tool_name in tool_id_to_info:
            tool_info.append({
                "name": tool_name,
                "description": tool_id_to_info[tool_name]["description"]
            })
        else:
            tool_info.append({
                "name": tool_name,
                "description": "No description available"
            })
    
    return tool_info

def generate_system_prompt(scenario, disposition, tool_info_list):
    tools_text = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in tool_info_list])
    
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        scenario=scenario,
        disposition=disposition,
        tools_text=tools_text
    )

    max_retries = config.generation.correction.max_retries
    retry_delay = config.generation.correction.retry_delay
    
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                base_url=base_url,
                temperature=0.7,
                timeout=config.generation.llm_settings.scenario.timeout
            )
            
            system_prompt = response.choices[0].message.content.strip()
            return system_prompt
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    LLM API call failed (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}. Retrying...")
                time.sleep(retry_delay)
            else:
                print(f"    LLM API call failed after {max_retries} attempts: {str(e)[:100]}")
                raise

def process_conversation(conv):
    try:
        scenario = conv["metadata"]["scenario"]
        disposition = conv["metadata"]["disposition"]
        
        tool_names = extract_tool_names_from_dml(conv["conversation_dml"])
        
        tool_info_list = get_tool_info_for_names(sorted(tool_names))
        
        system_prompt = generate_system_prompt(scenario, disposition, tool_info_list)
        
        conv["system_prompt"] = system_prompt
        
        return (True, conv)
        
    except Exception as e:
        print(f"\nError processing conversation: {str(e)[:200]}")
        print("Skipping this conversation...")
        return (False, None)


def main(max_conversations=None, input_path=None, output_path=None, n_jobs=None):
    if input_path is None:
        input_path = config.paths.output.conversations
    if output_path is None:
        output_path = config.paths.output.conversations_with_system_prompts
    if n_jobs is None:
        n_jobs = config.processing.system_prompt_generation.semaphore_limit
    
    if max_conversations:
        output_path = output_path.replace('.jsonl', '_test.jsonl')
        print(f"TEST MODE: Processing only {max_conversations} conversations")
    
    print(f"Loading conversations from {input_path}")
    print(f"Output will be written to {output_path}")
    
    all_conversations = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if max_conversations and i >= max_conversations:
                break
            
            data = json.loads(line)
            all_conversations.append(data)
    
    print(f"Loaded {len(all_conversations)} conversations")
    
    print("\nGenerating system prompts for each conversation...")
    print(f"Using {n_jobs} parallel workers")
    
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_conversation)(conv) for conv in tqdm(all_conversations, desc="Processing conversations")
    )
    
    successful_count = 0
    with open(output_path, 'w') as f_out:
        for success, conv in results:
            if success and conv is not None:
                f_out.write(json.dumps(conv) + "\n")
                successful_count += 1
    
    print(f"\nDone! Wrote {successful_count}/{len(all_conversations)} conversations with system prompts to {output_path}")

if __name__ == "__main__":
    main()
