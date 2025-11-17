import json
import random
from collections import defaultdict
from dml_parser import parse_dml
from config import config

tool_defs = json.load(open(config.paths.data.tool_definitions))
jsonl_path = config.paths.output.accepted_conversations

random.seed(config.processing.random_seed)

def sanitize_function_name(name: str) -> str:
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized


def extract_used_tool_names(messages: list) -> set:
    used_tools = set()
    for msg in messages:
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            for tool_call in msg['tool_calls']:
                if 'function' in tool_call and 'name' in tool_call['function']:
                    used_tools.add(tool_call['function']['name'])
    return used_tools


def limit_tools_to_max(tools: list, used_tool_names: set, max_tools: int = None) -> list:
    if max_tools is None:
        max_tools = config.processing.max_tools_per_conversation
    
    if len(tools) <= max_tools:
        return tools
    
    used_tools = []
    unused_tools = []
    
    for tool in tools:
        tool_name = tool['function']['name']
        if tool_name in used_tool_names:
            used_tools.append(tool)
        else:
            unused_tools.append(tool)
    
    num_random = max(0, max_tools - len(used_tools))
    
    if num_random > 0 and unused_tools:
        random_tools = random.sample(unused_tools, min(num_random, len(unused_tools)))
    else:
        random_tools = []
    
    return used_tools + random_tools


def toolkit_to_openai_tools(toolkit_name) -> list[dict]:
    openai_tools = []
    
    toolkit_names = toolkit_name.split('-')
    
    for tool in tool_defs:
        if tool['toolkit_name'] in toolkit_names:
            input_params = json.loads(tool['tool_input_parameters'])
            
            properties = {}
            required = []
            
            for param_name, param_spec in input_params.items():
                prop = {k: v for k, v in param_spec.items() if k != 'required'}
                
                if prop.get('type') == 'array' and 'items' not in prop:
                    prop['items'] = {'type': 'string'}
                
                properties[param_name] = prop
                
                if param_spec.get('required', False):
                    required.append(param_name)
            
            openai_tool = {
                "type": "function",
                "function": {
                    "name": sanitize_function_name(tool['tool_id']),
                    "description": tool['tool_description'],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
            
            openai_tools.append(openai_tool)
    
    return openai_tools


def has_missing_tool_responses(messages):
    num_missing = 0
    
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            tool_calls = msg['tool_calls']
            
            expected_tool_call_ids = {tc.get('id') for tc in tool_calls if tc.get('id')}
            
            j = i + 1
            found_tool_call_ids = set()
            
            while j < len(messages) and messages[j].get('role') == 'tool':
                tool_msg = messages[j]
                tool_call_id = tool_msg.get('tool_call_id')
                if tool_call_id:
                    found_tool_call_ids.add(tool_call_id)
                j += 1
            
            missing_tool_call_ids = expected_tool_call_ids - found_tool_call_ids
            num_missing += len(missing_tool_call_ids)
        
        i += 1
    
    return num_missing > 0, num_missing


conversations = defaultdict(list)
total_problematic_conversations = 0
total_problematic_examples = 0
total_examples = 0
tools_limited_count = 0
original_tools_counts = []
limited_tools_counts = []
filtered_conversations = 0
filtered_by_toolkit = defaultdict(int)

with open(jsonl_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        new_data = {}
        
        messages = parse_dml(data['conversation_dml'])
        sys_prompt = data['system_prompt']
        selected_toolkits = data['metadata']['selected_toolkits']
        original_toolkit_count = len(selected_toolkits)
        selected_toolkits = ["-".join(selected_toolkits)]
        assert len(selected_toolkits) == 1, "Only one toolkit is supported for now"
        toolkit = selected_toolkits[0]
        
        has_missing, num_missing = has_missing_tool_responses(messages)
        if has_missing:
            filtered_conversations += 1
            filtered_by_toolkit[toolkit] += 1
            continue
        
        used_tool_names_original = extract_used_tool_names(messages)
        
        tools = toolkit_to_openai_tools(toolkit)
        
        used_tool_names_sanitized = set()
        for msg in messages:
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                for tool_call in msg['tool_calls']:
                    if 'function' in tool_call and 'name' in tool_call['function']:
                        original_name = tool_call['function']['name']
                        sanitized_name = sanitize_function_name(original_name)
                        tool_call['function']['name'] = sanitized_name
                        used_tool_names_sanitized.add(sanitized_name)
            elif msg.get('role') == 'tool' and 'name' in msg:
                msg['name'] = sanitize_function_name(msg['name'])
        
        original_tool_count = len(tools)
        tools = limit_tools_to_max(tools, used_tool_names_sanitized)
        limited_tool_count = len(tools)
        
        max_tools = config.processing.max_tools_per_conversation
        if original_tool_count > max_tools:
            tools_limited_count += 1
            original_tools_counts.append(original_tool_count)
            limited_tools_counts.append(limited_tool_count)

        new_data['messages'] = messages
        new_data['system_prompt'] = sys_prompt
        new_data['tools'] = tools
        new_data['toolkit'] = toolkit
        new_data['original_toolkit_count'] = original_toolkit_count
        conversations[toolkit].append(new_data)

print(f"\n{'='*70}")
print("CONVERSATION FILTERING (Missing Tool Responses)")
print(f"{'='*70}")
print(f"Filtered conversations: {filtered_conversations}")
if filtered_conversations > 0:
    print(f"\nFiltered by toolkit:")
    for toolkit_name, count in sorted(filtered_by_toolkit.items()):
        print(f"  {toolkit_name}: {count}")
    print(f"\nReason: These conversations have tool calls without corresponding")
    print(f"        tool message responses, which violates OpenAI API requirements.")
else:
    print(f"No conversations filtered - all conversations are valid!")
print(f"{'='*70}\n")

max_tools = config.processing.max_tools_per_conversation
print(f"\n{'='*70}")
print(f"TOOL ARRAY LIMITING ({max_tools} MAX)")
print(f"{'='*70}")
if tools_limited_count > 0:
    print(f"Conversations with >{max_tools} tools: {tools_limited_count}")
    print(f"  Original tool counts: min={min(original_tools_counts)}, max={max(original_tools_counts)}, avg={sum(original_tools_counts)/len(original_tools_counts):.1f}")
    print(f"  Limited tool counts:  min={min(limited_tools_counts)}, max={max(limited_tools_counts)}, avg={sum(limited_tools_counts)/len(limited_tools_counts):.1f}")
    print(f"\nStrategy: Keep all used tools + random sample of unused tools (up to {max_tools} total)")
else:
    print(f"No conversations exceeded {max_tools} tools - no limiting applied")
print(f"{'='*70}\n")

parsed_conversations = []
for toolkit, convs in conversations.items():
    for conv in convs:
        parsed_conversations.append(conv)

with open(config.paths.output.parsed_conversations, 'w') as f:
    for conversation in parsed_conversations:
        f.write(json.dumps(conversation) + '\n')

print(f"\n✓ Total: {len(parsed_conversations)} parsed conversations")
print(f"✓ Parsed conversations saved to: {config.paths.output.parsed_conversations}")
