import re
import json

import re
import json
from json_repair import repair_json

def parse_json_safe(json_str: str, context: str = "JSON") -> dict:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = re.sub(r'"\s*\.\s*"\s*\]', '"]', json_str)
        json_str = re.sub(r'"\s*\.\s*"\s*}', '"}', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                repaired = repair_json(json_str)
                return json.loads(repaired)
            except Exception as e:
                raise ValueError(f"Error parsing {context}: {e}\nOriginal: {json_str[:500]}")

def format_assistant_content(thought: str, content: str) -> str:
    if not content:
        content = ""
    return f"<analysis>{thought}</analysis>\n<final>{content}</final>"

def parse_dml(dml_string: str) -> list:
    from collections import deque

    role_pattern = re.compile(r'(</user>|</assistant>)\s*', re.DOTALL)
    tool_pattern = re.compile(r'<tool>\s*(.+?)\s*</tool>', re.DOTALL)
    tool_response_pattern = re.compile(r'<tool_response>\s*(.+?)\s*</tool_response>', re.DOTALL)
    assistant_thought_pattern = re.compile(r'<assistant_thought>\s*(.+?)\s*</assistant_thought>', re.DOTALL)

    messages = []
    pending_tool_calls = deque()
    tool_call_counter = 0
    pos = 0
    current_role = None
    current_assistant_thought = None

    while pos < len(dml_string):
        thought_match = assistant_thought_pattern.match(dml_string, pos)
        if thought_match:
            current_assistant_thought = thought_match.group(1).strip()
            pos = thought_match.end()
            continue
        
        role_match = role_pattern.match(dml_string, pos)
        if role_match:
            tag = role_match.group(1)
            current_role = "user" if tag == "</user>" else "assistant"
            pos = role_match.end()

            content_parts = []

            while pos < len(dml_string):
                thought_match_inner = assistant_thought_pattern.match(dml_string, pos)
                if thought_match_inner:
                    current_assistant_thought = thought_match_inner.group(1).strip()
                    pos = thought_match_inner.end()
                    continue
                
                t_match = tool_pattern.match(dml_string, pos)
                if t_match:
                    content = ''.join(content_parts).strip()
                    content_parts = []

                    collected_tool_calls = []
                    temp_pos = pos
                    
                    while True:
                        t_match = tool_pattern.match(dml_string, temp_pos)
                        if not t_match:
                            break
                        
                        tool_json_str = t_match.group(1).strip()
                        try:
                            tool_data = parse_json_safe(tool_json_str, "tool JSON")
                            tool_name = tool_data["name"]
                            tool_args = tool_data.get("arguments", {})
                        except (ValueError, KeyError) as e:
                            raise ValueError(f"Error parsing tool JSON: {e}\nContent: {tool_json_str}")

                        tool_call_id = f"call_{tool_call_counter}"
                        tool_call_counter += 1
                        
                        pending_tool_calls.append((tool_call_id, tool_name))
                        
                        collected_tool_calls.append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args,
                            }
                        })
                        
                        temp_pos = t_match.end()
                        
                        while temp_pos < len(dml_string) and dml_string[temp_pos].isspace():
                            temp_pos += 1
                        
                        if not dml_string[temp_pos:].startswith('<tool>'):
                            break
                    
                    if current_role == "assistant" and current_assistant_thought is not None:
                        content = format_assistant_content(current_assistant_thought, content)
                        current_assistant_thought = None
                    
                    if (content and 
                        messages and 
                        messages[-1].get("role") == "assistant" and 
                        "tool_calls" not in messages[-1] and
                        "content" in messages[-1]):
                        messages[-1]["content"] = (messages[-1]["content"] + " " + content).strip()
                        messages[-1]["tool_calls"] = collected_tool_calls
                    elif (messages and 
                          messages[-1].get("role") == "assistant" and 
                          "tool_calls" not in messages[-1] and
                          "content" in messages[-1] and
                          not content):
                        messages[-1]["tool_calls"] = collected_tool_calls
                    else:
                        new_msg = {"role": "assistant", "tool_calls": collected_tool_calls}
                        if content:
                            new_msg["content"] = content
                        messages.append(new_msg)

                    pos = temp_pos
                    current_role = "assistant"
                    continue

                tr_match = tool_response_pattern.match(dml_string, pos)
                if tr_match:
                    content = ''.join(content_parts).strip()
                    if content:
                        if current_role == "assistant" and current_assistant_thought:
                            content = format_assistant_content(current_assistant_thought, content)
                            current_assistant_thought = None
                        
                        if messages and messages[-1].get("role") == current_role and "tool_calls" not in messages[-1]:
                            messages[-1]["content"] = (messages[-1]["content"] + " " + content).strip()
                        else:
                            messages.append({"role": current_role, "content": content})
                    content_parts = []

                    temp_pos = pos
                    
                    while True:
                        tr_match = tool_response_pattern.match(dml_string, temp_pos)
                        if not tr_match:
                            break
                        
                        response_json_str = tr_match.group(1).strip()
                        try:
                            parse_json_safe(response_json_str, "tool response JSON")
                        except ValueError as e:
                            raise ValueError(f"Error parsing tool response JSON: {e}\nContent: {response_json_str}")

                        if pending_tool_calls:
                            tool_call_id, tool_name = pending_tool_calls.popleft()
                        else:
                            tool_call_id, tool_name = "unknown", "unknown"

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": response_json_str,
                        })
                        
                        temp_pos = tr_match.end()
                        
                        while temp_pos < len(dml_string) and dml_string[temp_pos].isspace():
                            temp_pos += 1
                        
                        if not dml_string[temp_pos:].startswith('<tool_response>'):
                            break
                    
                    pos = temp_pos
                    current_role = "assistant"
                    continue

                next_role_match = role_pattern.match(dml_string, pos)
                if next_role_match:
                    content = ''.join(content_parts).strip()
                    if content:
                        if current_role == "assistant" and current_assistant_thought:
                            content = format_assistant_content(current_assistant_thought, content)
                            current_assistant_thought = None
                        
                        if messages and messages[-1].get("role") == current_role and "tool_calls" not in messages[-1]:
                            messages[-1]["content"] = (messages[-1]["content"] + " " + content).strip()
                        else:
                            messages.append({"role": current_role, "content": content})
                    content_parts = []
                    break

                next_tag_pos = min(
                    [p for p in [
                        dml_string.find('</user>', pos),
                        dml_string.find('</assistant>', pos),
                        dml_string.find('<tool>', pos),
                        dml_string.find('<tool_response>', pos),
                        dml_string.find('<assistant_thought>', pos),
                    ] if p != -1] or [len(dml_string)]
                )

                if next_tag_pos == pos:
                    pos += 1
                else:
                    content_parts.append(dml_string[pos:next_tag_pos])
                    pos = next_tag_pos

            if content_parts:
                content = ''.join(content_parts).strip()
                if content:
                    if current_role == "assistant" and current_assistant_thought is not None:
                        content = format_assistant_content(current_assistant_thought, content)
                        current_assistant_thought = None
                    
                    if messages and messages[-1].get("role") == current_role and "tool_calls" not in messages[-1]:
                        messages[-1]["content"] = (messages[-1]["content"] + " " + content).strip()
                    else:
                        messages.append({"role": current_role, "content": content})

            continue

        pos += 1

    return messages

def is_valid_dml(dml_string: str, require_thoughts: bool = True) -> tuple[bool, list[str]]:
    errors = []
    
    if not dml_string or not dml_string.strip():
        errors.append("""ERROR: Empty or whitespace-only DML string

RULE: DML conversations must contain actual content, not just whitespace.

REASON FOR FAILURE: The provided DML string is empty or contains only whitespace characters.

CORRECT FORMAT EXAMPLE:
</user> Hello, how are you?
<assistant_thought>
The user is greeting me. I should respond politely.
</assistant_thought>
</assistant> I'm doing well, thank you!""")
        return False, errors
    
    tag_pattern = re.compile(r"</user>|</assistant>|<assistant_thought>|</assistant_thought>|<tool>|</tool>|<tool_response>|</tool_response>")
    tags = tag_pattern.findall(dml_string)

    if not tags:
        errors.append("""ERROR: No valid DML tags found

RULE: DML conversations must contain valid tags like </user>, <assistant_thought>, </assistant>, <tool>, etc.

REASON FOR FAILURE: The string doesn't contain any recognized DML tags.

CORRECT FORMAT EXAMPLE:
</user> What's the weather like?
<assistant_thought>
User wants weather info. I should call the weather API.
</assistant_thought>
</assistant> Let me check that for you.""")
        return False, errors

    if require_thoughts:
        valid_start_tags = ["</user>", "<assistant_thought>"]
    else:
        valid_start_tags = ["</user>", "</assistant>", "<assistant_thought>"]
    
    if tags[0] not in valid_start_tags:
        if require_thoughts:
            errors.append(f"""ERROR: Invalid conversation start tag

RULE: DML conversations must start with either </user> or <assistant_thought> tag.

REASON FOR FAILURE: Your conversation starts with '{tags[0]}' instead of </user> or <assistant_thought>.

CORRECT FORMAT EXAMPLES:
# Starting with user message:
</user> I need help with something.

# Starting with assistant message:
<assistant_thought>
I will proactively help the user.
</assistant_thought>
</assistant> Hello! How can I help you today?""")
        else:
            errors.append(f"""ERROR: Invalid conversation start tag

RULE: DML conversations must start with either </user> or </assistant> tag.

REASON FOR FAILURE: Your conversation starts with '{tags[0]}' instead of </user> or </assistant>.

CORRECT FORMAT EXAMPLES:
# Starting with user message:
</user> I need help with something.

# Starting with assistant message:
</assistant> Hello! How can I help you today?""")

    tool_stack = []
    thought_stack = []
    
    for i in range(len(tags)):
        current_tag = tags[i]
        
        if current_tag == "<assistant_thought>":
            thought_stack.append("<assistant_thought>")
        
        elif current_tag == "</assistant_thought>":
            if not thought_stack or thought_stack[-1] != "<assistant_thought>":
                errors.append(f"""ERROR: Unmatched </assistant_thought> tag at position {i}

RULE: Every </assistant_thought> closing tag must have a matching <assistant_thought> opening tag before it.

REASON FOR FAILURE: Found </assistant_thought> at tag position {i} but there's no corresponding <assistant_thought> opening tag.

CORRECT FORMAT EXAMPLE:
<assistant_thought>
This is my internal reasoning about what to do next.
</assistant_thought>
</assistant> This is my response to the user.""")
                return False, errors
            thought_stack.pop()
        
        elif current_tag == "</assistant>":
            if require_thoughts and (i == 0 or tags[i-1] != "</assistant_thought>"):
                context_tags = tags[max(0, i-3):min(len(tags), i+2)]
                errors.append(f"""ERROR: Missing <assistant_thought> before </assistant> at position {i}

RULE: Every </assistant> tag must be IMMEDIATELY preceded by a </assistant_thought> tag. This ensures that every assistant response has documented internal reasoning.

REASON FOR FAILURE: The </assistant> tag at position {i} is not immediately preceded by </assistant_thought>. Context: {context_tags}

CORRECT FORMAT EXAMPLE:
<assistant_thought>
The user wants me to search for information. I'll use the search tool and then present the results.
</assistant_thought>
</assistant> Let me search for that information.
<tool>
{{"name": "search", "arguments": {{"query": "example"}}}}
</tool>

INCORRECT FORMAT (what you have):
</assistant> Let me search for that.  # Missing thought before this!""")
        
        elif current_tag == "<tool>":
            if i == 0 or tags[i-1] not in ["</assistant>", "</tool_response>"]:
                context_tags = tags[max(0, i-2):min(len(tags), i+2)]
                errors.append(f"""ERROR: Misplaced <tool> tag at position {i}

RULE: A <tool> tag must be preceded by </assistant> or </tool_response>.
Each tool call must have its response before the next tool call (interleaved format).

REASON FOR FAILURE: The <tool> tag at position {i} is preceded by '{tags[i-1] if i > 0 else 'nothing'}'. Context: {context_tags}

CORRECT FORMAT EXAMPLES:
# After assistant message:
</assistant> Let me search for that.
<tool>
{{"name": "search", "arguments": {{}}}}
</tool>

# Multiple tools in sequence (each must have response before next tool):
<tool>
{{"name": "tool1", "arguments": {{}}}}
</tool>
<tool_response>
{{"result": "from tool1"}}
</tool_response>
<tool>
{{"name": "tool2", "arguments": {{}}}}
</tool>
<tool_response>
{{"result": "from tool2"}}
</tool_response>""")
            tool_stack.append("<tool>")
        
        elif current_tag == "</tool>":
            if not tool_stack or tool_stack[-1] != "<tool>":
                errors.append(f"""ERROR: Unmatched </tool> closing tag at position {i}

RULE: Every </tool> closing tag must have a matching <tool> opening tag before it.

REASON FOR FAILURE: Found </tool> at position {i} but there's no corresponding <tool> opening tag, or the tags are mismatched.

CORRECT FORMAT EXAMPLE:
<tool>
{{"name": "my_tool", "arguments": {{"param": "value"}}}}
</tool>""")
                return False, errors
            tool_stack.pop()
        
        elif current_tag == "<tool_response>":
            if i == 0 or tags[i-1] != "</tool>":
                context_tags = tags[max(0, i-2):min(len(tags), i+2)]
                errors.append(f"""ERROR: Misplaced <tool_response> tag at position {i}

RULE: A <tool_response> tag must immediately follow </tool>.
Each tool call must have its response before the next tool call (interleaved format).

REASON FOR FAILURE: The <tool_response> at position {i} is preceded by '{tags[i-1] if i > 0 else 'nothing'}'. Context: {context_tags}

CORRECT FORMAT EXAMPLE:
<tool>
{{"name": "get_weather", "arguments": {{"city": "London"}}}}
</tool>
<tool_response>
{{"temperature": 15, "condition": "cloudy"}}
</tool_response>

IMPORTANT: With multiple tool calls, each tool call must be immediately followed by its response:
<tool>
{{"name": "tool1", "arguments": {{}}}}
</tool>
<tool_response>
{{"result": "from tool1"}}
</tool_response>
<tool>
{{"name": "tool2", "arguments": {{}}}}
</tool>
<tool_response>
{{"result": "from tool2"}}
</tool_response>""")
            tool_stack.append("<tool_response>")
        
        elif current_tag == "</tool_response>":
            if not tool_stack or tool_stack[-1] != "<tool_response>":
                errors.append(f"""ERROR: Unmatched </tool_response> closing tag at position {i}

RULE: Every </tool_response> closing tag must have a matching <tool_response> opening tag before it.

REASON FOR FAILURE: Found </tool_response> at position {i} but there's no corresponding <tool_response> opening tag, or the tags are mismatched.

CORRECT FORMAT EXAMPLE:
<tool_response>
{{"status": "success", "data": "result"}}
</tool_response>""")
                return False, errors
            tool_stack.pop()
    
    if tool_stack:
        errors.append(f"""ERROR: Unclosed tags at end of conversation

RULE: All opening tags must have corresponding closing tags. No tags should be left open.

REASON FOR FAILURE: The following tags are still open at the end: {tool_stack}

CORRECT FORMAT: Every <tool> must have </tool>, every <tool_response> must have </tool_response>, etc.

EXAMPLE:
<tool>
{{"name": "example", "arguments": {{}}}}
</tool>  # Must close the tag!""")
    
    if thought_stack:
        errors.append(f"""ERROR: Unclosed <assistant_thought> tags at end of conversation

RULE: All <assistant_thought> opening tags must have corresponding </assistant_thought> closing tags.

REASON FOR FAILURE: {len(thought_stack)} <assistant_thought> tag(s) were never closed.

CORRECT FORMAT EXAMPLE:
<assistant_thought>
My reasoning goes here.
</assistant_thought>  # Must close the thought tag!
</assistant> My response to the user.""")

    return len(errors) == 0, errors


def has_missing_tool_responses(messages):
    missing_details = []
    
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            tool_calls = msg['tool_calls']
            
            expected_tool_call_ids = {tc.get('id'): tc.get('function', {}).get('name', 'unknown') 
                                     for tc in tool_calls if tc.get('id')}
            
            j = i + 1
            found_tool_call_ids = set()
            
            while j < len(messages) and messages[j].get('role') == 'tool':
                tool_msg = messages[j]
                tool_call_id = tool_msg.get('tool_call_id')
                if tool_call_id:
                    found_tool_call_ids.add(tool_call_id)
                j += 1
            
            missing_tool_call_ids = set(expected_tool_call_ids.keys()) - found_tool_call_ids
            if missing_tool_call_ids:
                for missing_id in missing_tool_call_ids:
                    tool_name = expected_tool_call_ids[missing_id]
                    missing_details.append((i, missing_id, tool_name))
        
        i += 1
    
    if missing_details:
        error_msg = f"""ERROR: Missing tool responses for {len(missing_details)} tool call(s)

RULE: Every <tool> call must have a corresponding <tool_response> before the next </assistant> message.
The OpenAI API format requires that all tool calls receive responses.

REASON FOR FAILURE: The following tool calls don't have matching responses:
"""
        for msg_idx, call_id, tool_name in missing_details:
            error_msg += f"  - Tool call at message {msg_idx}: '{tool_name}' (id: {call_id})\n"
        
        error_msg += """
CORRECT FORMAT EXAMPLE:
<assistant_thought>
I need to call the search tool.
</assistant_thought>
</assistant> Let me search for that.
<tool>
{"name": "search", "arguments": {"query": "example"}}
</tool>
<tool_response>
{"results": ["item1", "item2"]}
</tool_response>

INCORRECT FORMAT (what you have):
<assistant_thought>
I need to call the search tool.
</assistant_thought>
</assistant> Let me search for that.
<tool>
{"name": "search", "arguments": {"query": "example"}}
</tool>
# Missing <tool_response> here!
<assistant_thought>
Next step...
</assistant_thought>
</assistant> Moving on...  # ERROR: Can't have another assistant message without tool response!"""
        
        return True, error_msg
    
    return False, None


def has_problematic_evaluation_examples(messages):
    problem_locations = []
    
    for i, msg in enumerate(messages):
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            if i > 0 and messages[i-1].get('role') == 'tool':
                prev_tool_name = messages[i-1].get('name', 'unknown')
                curr_tool_names = [tc.get('function', {}).get('name', 'unknown') 
                                  for tc in msg.get('tool_calls', [])]
                problem_locations.append((i, prev_tool_name, curr_tool_names))
    
    if problem_locations:
        error_msg = f"""ERROR: Problematic evaluation pattern - {len(problem_locations)} instance(s) found

RULE: An assistant message with new tool calls should not immediately follow tool response messages.
The assistant should first synthesize/process the tool results before making new tool calls.

REASON FOR FAILURE: This pattern creates problematic training examples where tool responses are left unprocessed.
The assistant should have a message that interprets/uses the tool results before calling more tools.

PROBLEMATIC LOCATIONS:
"""
        for msg_idx, prev_tool, next_tools in problem_locations:
            error_msg += f"  - Message {msg_idx}: After tool response from '{prev_tool}', immediately calls {next_tools}\n"
        
        error_msg += """
CORRECT FORMAT EXAMPLE:
<assistant_thought>
I'll call the weather API.
</assistant_thought>
</assistant> Let me check the weather.
<tool>
{"name": "get_weather", "arguments": {"city": "London"}}
</tool>
<tool_response>
{"temperature": 15, "condition": "cloudy"}
</tool_response>
<assistant_thought>
Got the weather: 15°C and cloudy. Now I need to search for restaurants.
</assistant_thought>
</assistant> It's 15°C and cloudy. Let me find restaurants for you.
<tool>
{"name": "search_restaurants", "arguments": {"city": "London"}}
</tool>

INCORRECT FORMAT (what you have):
<assistant_thought>
I'll call the weather API.
</assistant_thought>
</assistant> Let me check the weather.
<tool>
{"name": "get_weather", "arguments": {"city": "London"}}
</tool>
<tool_response>
{"temperature": 15, "condition": "cloudy"}
</tool_response>
# ERROR: Next line immediately calls another tool without processing the weather result!
<assistant_thought>
Now searching restaurants.
</assistant_thought>
</assistant>
<tool>
{"name": "search_restaurants", "arguments": {"city": "London"}}
</tool>"""
        
        return True, error_msg
    
    return False, None


def get_tools_from_toolkit(toolkit_name, tool_defs):
    tools = []
    for tool in tool_defs:
        if tool["toolkit_name"] == toolkit_name:
            tools.append(tool["tool_id"])
    return tools


def reject_based_on_toolname(conversation_dml, selected_toolkits, tool_defs):
    # are there any hallucinated tool calls?
    conversation_dml_lines = conversation_dml.split('\n')
    
    tool_calls = []
    tool_call = []
    keep = False
    for line in conversation_dml_lines:
        if keep:
            tool_call.append(line)
        if '<tool>' in line:
            keep = True
        if '</tool>' in line:
            tool_calls.append("\n".join(tool_call))
            tool_call = []
            keep = False
    
    tool_names = []
    for tc in tool_calls:
        if '"name": "' in tc:
            tool_names.append(tc.split('"name": "')[1].split("\n")[0].split('"')[0])
    
    unique_tool_calls = list(set(tool_names))
    
    applicable_tool_ids = []
    for toolkit in selected_toolkits:
        for tool_id in get_tools_from_toolkit(toolkit, tool_defs):
            applicable_tool_ids.append(tool_id)
    
    invalid_tools = []
    for tool in unique_tool_calls:
        if tool not in applicable_tool_ids:
            invalid_tools.append(tool)
    
    if invalid_tools:
        error_msg = f"""ERROR: Invalid tool name(s) used in conversation

RULE: All tool calls must use tool IDs (e.g., "TOOL_12345") that exist in the selected toolkits.
You cannot call tools that aren't available in the current toolkit set.

REASON FOR FAILURE: The following tool name(s) don't exist in the selected toolkits {selected_toolkits}:
"""
        for tool in invalid_tools:
            error_msg += f"  - '{tool}'\n"
        
        error_msg += f"""
AVAILABLE TOOLS for your selected toolkits:
"""
        for tool_id in applicable_tool_ids:
            error_msg += f"  - {tool_id}\n"
        
        error_msg += """
CORRECT FORMAT EXAMPLE:
<tool>
{"name": "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL", "arguments": {"channel": "#general"}}
</tool>

INCORRECT FORMAT (what you might have):
<tool>
{"name": "send_slack_message", "arguments": {"channel": "#general"}}  # Wrong! Use the full tool ID
</tool>

FIX: Replace invalid tool names with valid tool IDs from the available tools list above."""
        
        return True, None, error_msg
    
    return False, conversation_dml, None


def validate_and_clean_conversation(conversation_dml, selected_toolkits, tool_defs, require_thoughts=True):
    all_errors = []
    cleaned_dml = conversation_dml
    
    is_valid_structure, structure_errors = is_valid_dml(conversation_dml, require_thoughts=require_thoughts)
    if not is_valid_structure:
        all_errors.extend(structure_errors)
        return False, None, all_errors
    
    should_reject, temp_cleaned_dml, tool_name_error = reject_based_on_toolname(cleaned_dml, selected_toolkits, tool_defs)
    if should_reject:
        all_errors.append(tool_name_error)
        return False, None, all_errors
    else:
        cleaned_dml = temp_cleaned_dml
    
    try:
        messages = parse_dml(cleaned_dml)
    except Exception as e:
        parse_error = f"""ERROR: Failed to parse DML into message format

RULE: DML must be parseable into a valid conversation structure.

REASON FOR FAILURE: Parser encountered an error: {str(e)}

This usually means:
- Malformed JSON in <tool> or <tool_response> tags
- Unexpected tag ordering
- Missing required fields in tool calls

Please check your JSON syntax and tag structure."""
        all_errors.append(parse_error)
        return False, None, all_errors
    
    has_missing, missing_error = has_missing_tool_responses(messages)
    if has_missing:
        all_errors.append(missing_error)
    
    has_problems, problem_error = has_problematic_evaluation_examples(messages)
    if has_problems:
        all_errors.append(problem_error)
    
    if all_errors:
        return False, None, all_errors
    else:
        return True, cleaned_dml, []

