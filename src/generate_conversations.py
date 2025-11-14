import json
import random
from collections import defaultdict
from litellm import completion
from dml_parser import validate_and_clean_conversation
from tqdm import tqdm
from joblib import Parallel, delayed
import time
from datetime import datetime
from pydantic import BaseModel
from filelock import FileLock
import threading
from sentence_transformers import SentenceTransformer
import os
from config import config


class RestartFromScratchError(Exception):
    pass

def load_prompt(prompt_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", f"{prompt_name}.txt")
    with open(prompt_path, 'r') as f:
        return f.read()

print("Loading prompts...")
PROMPTS = {
    "scenario_generation": load_prompt("scenario_generation"),
    "scenario_generation_with_examples": load_prompt("scenario_generation_with_examples"),
    "user_request_generation": load_prompt("user_request_generation"),
    "disposition_generation": load_prompt("disposition_generation"),
    "tool_names_selection": load_prompt("tool_names_selection"),
    "example_conversation_with_thoughts": load_prompt("example_conversation_with_thoughts"),
    "example_conversation_without_thoughts": load_prompt("example_conversation_without_thoughts"),
    "dml_format_with_thoughts": load_prompt("dml_format_with_thoughts"),
    "dml_format_without_thoughts": load_prompt("dml_format_without_thoughts"),
    "conversation_system": load_prompt("conversation_system"),
    "thought_instructions_with_thoughts": load_prompt("thought_instructions_with_thoughts"),
    "thought_instructions_without_thoughts": load_prompt("thought_instructions_without_thoughts"),
    "conversation_user": load_prompt("conversation_user"),
    "correction_prompt": load_prompt("correction_prompt"),
    "correction_system": load_prompt("correction_system"),
    "dml_instructions_with_thoughts": load_prompt("dml_instructions_with_thoughts"),
    "dml_instructions_without_thoughts": load_prompt("dml_instructions_without_thoughts"),
}
print("Prompts loaded successfully!")

wanted_toolkits_config = json.load(open(config.paths.data.wanted_toolkits))
single_toolkits = wanted_toolkits_config["toolkits"]
toolkit_bundles = wanted_toolkits_config["bundles"]

tool_defs = json.load(open(config.paths.data.tool_definitions))

INCLUDE_ASSISTANT_THOUGHTS = config.generation.include_assistant_thoughts

if INCLUDE_ASSISTANT_THOUGHTS:
    dml_grammar = open(config.paths.grammars.with_thoughts).read()
else:
    dml_grammar = open(config.paths.grammars.without_thoughts).read()

api_key = config.llm_api.main.api_key
base_url = config.llm_api.main.base_url
model = config.llm_api.main.model

conversation_api_key = config.llm_api.conversation.api_key
conversation_base_url = config.llm_api.conversation.base_url
conversation_model = config.llm_api.conversation.model

print(f"Loading SentenceTransformer model: {config.embedding.model_name}")
similarity_model = SentenceTransformer(config.embedding.model_name, device=config.embedding.device)
print("Model loaded successfully!")

scenario_history = defaultdict(list)

USER_CLARITY_LEVELS = config.generation.clarity_levels.to_dict()

def select_random_clarity_level():
    levels = list(USER_CLARITY_LEVELS.keys())
    weights = [USER_CLARITY_LEVELS[level]["weight"] for level in levels]
    return random.choices(levels, weights=weights)[0]

def compute_similarity_score(text_1, text_2, instruction=None):
    embeddings = similarity_model.encode([text_1, text_2])
    
    similarity = similarity_model.similarity(embeddings[0:1], embeddings[1:2])
    
    similarity_score = similarity[0][0].item()
    normalized_score = (similarity_score + 1) / 2
    
    return normalized_score

def is_scenario_diverse(scenario, toolkit_names, threshold=None):
    if threshold is None:
        threshold = config.generation.diversity.similarity_threshold
    
    toolkit_key = frozenset(toolkit_names)
    existing_scenarios = scenario_history[toolkit_key]
    
    if not existing_scenarios:
        return True, None, 0.0
    
    all_scores = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_similarity_score)(scenario, existing_scenario, instruction='Given two business scenarios, determine if they describe substantially similar use cases that would feel redundant or repetitive to a user. Answer "yes" only if the scenarios are highly similar in their core business problem, workflow, and objectives - not just because they use similar tools or are in the same industry domain.')
        for existing_scenario in existing_scenarios
    )
    
    max_similarity = max(all_scores) if all_scores else 0.0
    most_similar_idx = all_scores.index(max_similarity) if all_scores else None
    most_similar_scenario = existing_scenarios[most_similar_idx] if most_similar_idx is not None else None
    
    print(f"  Max similarity: {max_similarity:.4f} (threshold: {threshold})")
    
    return max_similarity < threshold, most_similar_scenario, max_similarity

def completion_with_retry(**kwargs):
    max_retries = config.generation.correction.max_retries
    retry_delay = config.generation.correction.retry_delay
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = completion(**kwargs)
            return response
        except Exception as e:
            error_str = str(e)
            retry_count += 1
            
            if "ContextWindowExceededError" in error_str or "context_length_exceeded" in error_str or "endpoint's maximum context length is" in error_str:
                print(f"Context window exceeded: {error_str[:150]}. Restarting from scratch...")
                raise RestartFromScratchError(f"Context window exceeded: {error_str}")
            print(f"LLM API call failed with error: {error_str[:100]}. Retrying in {retry_delay} seconds...")
            print(error_str)
            time.sleep(retry_delay)

    print(f"Max retries reached. Last error: {error_str[:100]}")
    raise RestartFromScratchError(f"Max retries reached. Last error: {error_str[:100]}")

def sample_toolkit_configuration():
    toolkits = defaultdict(list)
    for tool in tool_defs:
        toolkit_name = tool["toolkit_name"]
        tool = {
            "tool_id": tool["tool_id"],
            "tool_description": tool["tool_description"],
            "tool_input_parameters": tool["tool_input_parameters"],
            "tool_response_parameters": tool["tool_response_parameters"]
        }
        toolkits[toolkit_name].append(tool)
    
    all_configurations = []
    
    for toolkit_name in single_toolkits:
        all_configurations.append([toolkit_name])
    
    for bundle in toolkit_bundles:
        all_configurations.append(bundle)
    
    selected_config = random.choice(all_configurations)
    
    result = {}
    for toolkit_name in selected_config:
        if toolkit_name in toolkits:
            result[toolkit_name] = toolkits[toolkit_name]
        else:
            print(f"Warning: toolkit '{toolkit_name}' not found in tool definitions")
    
    return result

def create_scenario(selected_toolkits):
    toolkit_names = list(selected_toolkits.keys())
    toolkit_descriptions = []
    
    for toolkit_name, tools in selected_toolkits.items():
        tool_summaries = [f"- {tool['tool_id']}: {tool['tool_description']}" for tool in tools]
        toolkit_descriptions.append(f"**{toolkit_name}**:\n" + "\n".join(tool_summaries))
    
    toolkit_key = frozenset(toolkit_names)
    existing_scenarios = scenario_history[toolkit_key]
    
    if not existing_scenarios:
        prompt = PROMPTS["scenario_generation"].format(
            toolkit_names=', '.join(toolkit_names),
            toolkit_descriptions='Tool:\n'.join(toolkit_descriptions)
        )
    else:
        num_samples = min(config.generation.diversity.num_seed_scenarios, len(existing_scenarios))
        seed_scenarios = random.sample(existing_scenarios, num_samples)
        
        seed_scenarios_text = "Scenario:\n\n".join([f"{i+1}. {scenario}" for i, scenario in enumerate(seed_scenarios)])
        
        prompt = PROMPTS["scenario_generation_with_examples"].format(
            toolkit_names=', '.join(toolkit_names),
            toolkit_descriptions='Tool:\n'.join(toolkit_descriptions),
            seed_scenarios_text=seed_scenarios_text
        )
    
    workflow_description = getattr(config.generation, 'workflow_description', '')
    if workflow_description and workflow_description.strip():
        prompt = prompt + f"\n\nAdditional scenario generation instruction:\n{workflow_description}"

    response = completion_with_retry(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
        base_url=base_url,
        temperature=config.generation.llm_settings.scenario.temperature,
        top_k=config.generation.llm_settings.scenario.top_k,
        top_p=config.generation.llm_settings.scenario.top_p,
        timeout=config.generation.llm_settings.scenario.timeout,
        include_reasoning=False,
        extra_body={"reasoning": {"summary": "auto", "effort": "medium"}}
    )
    
    if "</think>" in response.choices[0].message.content:
        return response.choices[0].message.content.split("</think>")[1].strip()
    else:
        return response.choices[0].message.content.strip()

def format_tools_for_llm(selected_toolkits):
    tools = []
    for toolkit_name, toolkit_tools in selected_toolkits.items():
        for tool in toolkit_tools:
            tools.append(tool)
    return tools

class ToolNamesResponse(BaseModel):
    reasoning: str
    tool_ids: list[str]

def create_correction_prompt(faulty_conversation, validation_errors, dml_format_instructions):
    errors_section = "\n\n" + "="*80 + "\n\n".join(validation_errors)
    
    prompt = PROMPTS["correction_prompt"].format(
        errors_section=errors_section,
        dml_format_instructions=dml_format_instructions,
        faulty_conversation=faulty_conversation
    )
    
    return prompt

def generate_conversation(scenario, selected_toolkits, clarity_level="good"):
    user_request_prompt = PROMPTS["user_request_generation"].format(scenario=scenario)

    user_request_response = completion_with_retry(
        model=model,
        messages=[{"role": "user", "content": user_request_prompt}],
        api_key=api_key,
        base_url=base_url
    )
    
    if "</think>" in user_request_response.choices[0].message.content:
        user_persona_desc = user_request_response.choices[0].message.content.split("</think>")[1].strip()
    else:
        user_persona_desc = user_request_response.choices[0].message.content.strip()

    disposition_prompt = PROMPTS["disposition_generation"].format(
        scenario=scenario,
        user_persona_desc=user_persona_desc
    )

    disposition_response = completion_with_retry(
        model=model,
        messages=[{"role": "user", "content": disposition_prompt}],
        api_key=api_key,
        base_url=base_url
    )

    if "</think>" in disposition_response.choices[0].message.content:
        disposition_desc = disposition_response.choices[0].message.content.split("</think>")[1].strip()
    else:
        disposition_desc = disposition_response.choices[0].message.content.strip()

    all_formatted_tools = format_tools_for_llm(selected_toolkits)
    all_formatted_tools_json = json.dumps(all_formatted_tools, indent=2)

    toolnames_prompt = PROMPTS["tool_names_selection"].format(
        scenario=scenario,
        user_persona_desc=user_persona_desc,
        disposition_desc=disposition_desc,
        all_formatted_tools_json=all_formatted_tools_json
    )
    
    tool_ids_response = completion_with_retry(
        model=model,
        messages=[{"role": "user", "content": toolnames_prompt}],
        api_key=api_key,
        base_url=base_url,
        temperature=config.generation.llm_settings.tool_selection.temperature,
        timeout=config.generation.llm_settings.tool_selection.timeout,
        response_format=ToolNamesResponse
    )

    if "</think>" in tool_ids_response.choices[0].message.content:
        tool_ids_desc = tool_ids_response.choices[0].message.content.split("</think>")[1].strip()
    else:
        tool_ids_desc = tool_ids_response.choices[0].message.content.strip()
    tool_ids = json.loads(tool_ids_desc)["tool_ids"]

    selected_tools = [tool for tool in all_formatted_tools if tool["tool_id"] in tool_ids]
    
    remaining_tools = [tool for tool in all_formatted_tools if tool not in selected_tools]
    num_additional = min(config.generation.max_tools_in_context, len(remaining_tools))
    if num_additional > 0:
        additional_tools = random.sample(remaining_tools, num_additional)
        selected_tools.extend(additional_tools)
    
    example_conversation = PROMPTS["example_conversation_with_thoughts"] if INCLUDE_ASSISTANT_THOUGHTS else PROMPTS["example_conversation_without_thoughts"]

    formatted_tools = json.dumps(selected_tools, indent=2)
    
    dml_format_rules = PROMPTS["dml_format_with_thoughts"].format(
        example_conversation=example_conversation
    ) if INCLUDE_ASSISTANT_THOUGHTS else PROMPTS["dml_format_without_thoughts"].format(
        example_conversation=example_conversation
    )
    
    conversation_sys_prompt = PROMPTS["conversation_system"].format(dml_format_rules=dml_format_rules)
    
    clarity_description = USER_CLARITY_LEVELS[clarity_level]["description"]
    
    if INCLUDE_ASSISTANT_THOUGHTS:
        thought_instructions = PROMPTS["thought_instructions_with_thoughts"].format(
            clarity_level=clarity_level.upper(),
            clarity_description=clarity_description
        )
    else:
        thought_instructions = PROMPTS["thought_instructions_without_thoughts"].format(
            clarity_level=clarity_level.upper(),
            clarity_description=clarity_description
        )
    
    conversation_prompt = PROMPTS["conversation_user"].format(
        scenario=scenario,
        user_persona_desc=user_persona_desc,
        disposition_desc=disposition_desc,
        formatted_tools=formatted_tools,
        thought_instructions=thought_instructions
    )

    conversation_response = completion_with_retry(
        model=conversation_model,
        messages=[{"role": "system", "content": conversation_sys_prompt}, {"role": "user", "content": conversation_prompt}],
        api_key=conversation_api_key,
        base_url=conversation_base_url,
        max_tokens=config.generation.llm_settings.conversation.max_tokens,
        timeout=config.generation.llm_settings.conversation.timeout,
        include_reasoning=False,
        min_p=config.generation.llm_settings.conversation.min_p,
        extra_body={"reasoning": {"summary": "auto", "effort": "high"}, "structured_outputs": {"grammar": dml_grammar}}
    )

    conversation = conversation_response.choices[0].message.content.strip()
    conversation = conversation.replace("```xml", "").replace("```", "").strip()
    
    return conversation, scenario, user_persona_desc, disposition_desc, clarity_level

def generate_one_conversation_with_toolkits(selected_toolkits):
    clarity_level = select_random_clarity_level()
    toolkit_names = list(selected_toolkits.keys())
    
    max_scenario_retries = config.generation.diversity.max_scenario_retries
    scenario = None
    
    for scenario_retry in range(max_scenario_retries):
        scenario = create_scenario(selected_toolkits)


        is_diverse, most_similar_scenario, max_similarity = is_scenario_diverse(scenario, toolkit_names)
        if is_diverse:
            print(f"  Diverse scenario generated after {scenario_retry + 1} attempt(s)!")
            toolkit_key = frozenset(toolkit_names)
            scenario_history[toolkit_key].append(scenario)
            break
        else:
            print(f"  Scenario too similar (attempt {scenario_retry + 1}/{max_scenario_retries}, similarity={max_similarity:.4f})")
            print(f"  NEW SCENARIO: {scenario}")
            print(f"  DUPLICATE OF: {most_similar_scenario}")
            print(f"  Retrying with different seed scenarios...")
    
    if scenario is None:
        raise RuntimeError(f"Failed to generate diverse scenario after {max_scenario_retries} attempts")
    
    dml_format_instructions = PROMPTS["dml_instructions_with_thoughts"] if INCLUDE_ASSISTANT_THOUGHTS else PROMPTS["dml_instructions_without_thoughts"]
    
    while True:
        try:
            conversation, scenario, user_persona, disposition, clarity_level_used = generate_conversation(
                scenario, selected_toolkits, clarity_level=clarity_level
            )
            
            max_correction_attempts = config.generation.correction.max_attempts
            current_conversation = conversation
            
            for correction_attempt in range(max_correction_attempts + 1):
                is_valid, cleaned_conversation, error_list = validate_and_clean_conversation(
                    current_conversation, toolkit_names, tool_defs, require_thoughts=INCLUDE_ASSISTANT_THOUGHTS
                )
                num_user_messages = current_conversation.count("</user>")
                
                has_validation_errors = not is_valid
                is_too_short = num_user_messages <= config.generation.min_user_messages
                
                if not has_validation_errors and not is_too_short:
                    final_conversation = cleaned_conversation if cleaned_conversation else current_conversation
                    
                    correction_info = f" (after {correction_attempt} correction(s))" if correction_attempt > 0 else ""
                    print(f"✓ Validation passed{correction_info} (user_messages={num_user_messages}, errors={len(error_list)}, clarity={clarity_level})")
                    return {
                        "conversation_dml": final_conversation,
                        "metadata": {
                            "scenario": scenario,
                            "user_persona": user_persona,
                            "disposition": disposition,
                            "selected_toolkits": toolkit_names,
                            "clarity_level": clarity_level_used,
                            "timestamp": datetime.now().isoformat(),
                            "correction_attempts": correction_attempt
                        },
                        "selected_toolkits_dict": selected_toolkits
                    }
                
                elif correction_attempt < max_correction_attempts:
                    failure_reasons = []
                    if has_validation_errors:
                        failure_reasons.append(f"{len(error_list)} validation error(s)")
                    if is_too_short:
                        failure_reasons.append(f"too short ({num_user_messages} user messages, need >{config.generation.min_user_messages})")
                    
                    failure_msg = " AND ".join(failure_reasons)
                    print(f"✗ Validation failed on attempt {correction_attempt + 1}/{max_correction_attempts + 1}: {failure_msg}")
                    
                    if has_validation_errors:
                        print(f"  Asking LLM to fix validation errors...")
                        
                        correction_prompt = create_correction_prompt(
                            current_conversation,
                            error_list,
                            dml_format_instructions
                        )
                        
                        try:
                            correction_response = completion_with_retry(
                                model=conversation_model,
                                messages=[{"role": "system", "content": PROMPTS["correction_system"]}, {"role": "user", "content": correction_prompt}],
                                api_key=conversation_api_key,
                                base_url=conversation_base_url,
                                max_tokens=config.generation.llm_settings.correction.max_tokens,
                                timeout=config.generation.llm_settings.correction.timeout,
                                include_reasoning=False,
                                min_p=config.generation.llm_settings.correction.min_p,
                                extra_body={"reasoning": {"summary": "auto", "effort": "high"}, "structured_outputs": {"grammar": dml_grammar}}
                            )
                            
                            corrected_conversation = correction_response.choices[0].message.content.strip()
                            corrected_conversation = corrected_conversation.replace("```xml", "").replace("```", "").strip()
                            current_conversation = corrected_conversation
                            print(f"  LLM returned corrected conversation. Re-validating...")
                            
                        except RestartFromScratchError as e:
                            print(f"  Context window error during correction: {str(e)[:100]}. Regenerating from scratch...")
                            break
                    else:
                        print(f"  Conversation too short, regenerating from scratch...")
                        break
                else:
                    failure_reasons = []
                    if has_validation_errors:
                        failure_reasons.append(f"{len(error_list)} validation error(s)")
                    if is_too_short:
                        failure_reasons.append(f"too short ({num_user_messages} user messages)")
                    
                    failure_msg = " AND ".join(failure_reasons)
                    print(f"✗ Failed after {max_correction_attempts} correction attempts ({failure_msg}). Regenerating from scratch...")
                    break
                    
        except RestartFromScratchError as e:
            print(f"Context window error caught during generation: {str(e)[:100]}. Retrying with same toolkits and clarity level...")
            continue

def generate_one_conversation():
    selected_toolkits = sample_toolkit_configuration()
    print(f"Selected toolkits: {list(selected_toolkits.keys())}")
    
    result = generate_one_conversation_with_toolkits(selected_toolkits)
    
    return result


def generate_conversations(total_conversations: int = None, output_file: str = None, n_jobs: int = None):
    if total_conversations is None:
        total_conversations = config.generation.total_conversations
    if output_file is None:
        output_file = config.paths.output.conversations
    if n_jobs is None:
        n_jobs = config.processing.conversation_generation.n_jobs
    conversation_id_lock = threading.Lock()
    conversation_id = [0]
    
    lock_file = f"{output_file}.lock"
    
    print(f"Generating {total_conversations} conversations with uniform sampling")
    print(f"Output file: {output_file}")
    print(f"{'='*80}")
    
    with open(output_file, 'w') as f:
        pass
    
    def generate_and_save():
        result = generate_one_conversation()
        
        with conversation_id_lock:
            conversation_id[0] += 1
            current_id = conversation_id[0]
        
        output_obj = {
            "id": f"conv_{current_id:05d}",
            "conversation_dml": result["conversation_dml"],
            "metadata": {
                **result["metadata"],
                "conversation_id": current_id
            }
        }
        
        with FileLock(lock_file):
            with open(output_file, 'a') as f:
                f.write(json.dumps(output_obj) + "\n")
                f.flush()
        
        return current_id
    
    saved_ids = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(generate_and_save)() for _ in tqdm(range(total_conversations), desc="Generating conversations")
    )
    
    final_count = conversation_id[0]
    print(f"\nAll conversations saved: {final_count}")
    print(f"Scenario history size: {sum(len(v) for v in scenario_history.values())} scenarios across {len(scenario_history)} toolkit combinations")
    
    print(f"\n{'='*80}")
    print(f"All {total_conversations} conversations generated and saved to {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    generate_conversations()
