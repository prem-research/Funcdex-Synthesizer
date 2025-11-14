import argparse
import json
import os
import re
import yaml
from pathlib import Path
import streamlit as st


def find_project_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / "config.yaml").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def process_env_vars(config):
    if isinstance(config, dict):
        return {k: process_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [process_env_vars(item) for item in config]
    elif isinstance(config, str):
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_env(match):
            env_var = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(env_var, default)
        
        return re.sub(pattern, replace_env, config)
    else:
        return config


def load_config(config_path: Path, project_root: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Process environment variables
    config = process_env_vars(raw_config)
    
    # Resolve the accepted_conversations path
    accepted_path = config.get('paths', {}).get('output', {}).get('parsed_conversations', 'outputs/parsed_conversations.jsonl')
    accepted_path = Path(accepted_path)
    if not accepted_path.is_absolute():
        accepted_path = project_root / accepted_path
    
    return str(accepted_path)


@st.cache_data
def load_dataset(path: str) -> list[dict]:
    conversations = []
    with open(path, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
    return conversations


def get_toolkit_counts(conversations: list[dict]) -> dict:
    toolkit_counts = {}
    for conv in conversations:
        # Handle both 'toolkit' field and 'metadata.selected_toolkits' array
        if 'toolkit' in conv:
            toolkit = conv['toolkit']
        elif 'metadata' in conv and 'selected_toolkits' in conv['metadata']:
            toolkit = '+'.join(sorted(conv['metadata']['selected_toolkits']))
        else:
            toolkit = 'unknown'
        toolkit_counts[toolkit] = toolkit_counts.get(toolkit, 0) + 1
    return toolkit_counts


def filter_by_toolkit(conversations: list[dict], toolkit: str) -> list[dict]:
    if toolkit == "All":
        return conversations
    
    filtered = []
    for conv in conversations:
        # Handle both 'toolkit' field and 'metadata.selected_toolkits' array
        if 'toolkit' in conv:
            conv_toolkit = conv['toolkit']
        elif 'metadata' in conv and 'selected_toolkits' in conv['metadata']:
            conv_toolkit = '+'.join(sorted(conv['metadata']['selected_toolkits']))
        else:
            conv_toolkit = 'unknown'
        
        if conv_toolkit == toolkit:
            filtered.append(conv)
    
    return filtered


def display_conversation(conversation: dict):
    
    # Extract toolkit info
    if 'toolkit' in conversation:
        toolkit_display = conversation['toolkit']
    elif 'metadata' in conversation and 'selected_toolkits' in conversation['metadata']:
        toolkit_display = '+'.join(sorted(conversation['metadata']['selected_toolkits']))
    else:
        toolkit_display = 'unknown'
    
    # Metadata
    st.markdown(f"**Toolkit:** `{toolkit_display}` | **Messages:** {len(conversation.get('messages', []))} | **Tools:** {len(conversation.get('tools', []))}")
    
    # System Prompt
    with st.expander("System Prompt"):
        st.code(conversation['system_prompt'], language=None)
    
    # Messages (raw)
    st.markdown("### Messages")
    for i, msg in enumerate(conversation['messages']):
        role = msg.get('role', 'unknown')
        
        # Compact role indicator with color
        if role == 'user':
            st.markdown(f"**[{i+1}] 👤 USER**")
            st.code(json.dumps(msg, indent=2), language='json')
        elif role == 'assistant':
            st.markdown(f"**[{i+1}] 🤖 ASSISTANT**")
            st.code(json.dumps(msg, indent=2), language='json')
        elif role == 'tool':
            st.markdown(f"**[{i+1}] ⚙️ TOOL**")
            st.code(json.dumps(msg, indent=2), language='json')
        else:
            st.markdown(f"**[{i+1}] {role.upper()}**")
            st.code(json.dumps(msg, indent=2), language='json')
    
    # Tools list
    with st.expander("Available Tools"):
        tools_summary = [t['function']['name'] for t in conversation.get('tools', [])]
        st.write(f"Total: {len(tools_summary)}")
        st.code('\n'.join(tools_summary), language=None)


def main():
    st.set_page_config(page_title="Accepted Conversations Inspector", layout="wide", page_icon="✅")
    
    st.title("✅ Accepted Conversations Inspector")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inspect accepted conversations from the pipeline")
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    
    # Get args from sys.argv (Streamlit compatible)
    args, _ = parser.parse_known_args()
    
    # Find project root and resolve config path
    project_root = find_project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"**Config:** `{config_path.name}`")
        st.markdown(f"**Project Root:** `{project_root.name}`")
        st.divider()
        
        # Load dataset path from config
        try:
            dataset_path = load_config(config_path, project_root)
            st.success(f"📁 Loaded config")
        except FileNotFoundError as e:
            st.error(f"❌ Config not found: {config_path}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading config: {str(e)}")
            st.stop()
        
        # Load dataset
        try:
            conversations = load_dataset(dataset_path)
            st.success(f"✅ {len(conversations)} conversations")
        except FileNotFoundError:
            st.error(f"❌ File not found: {dataset_path}")
            st.info("Run the pipeline first to generate accepted conversations.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            st.stop()
        
        # Toolkit filter
        toolkit_counts = get_toolkit_counts(conversations)
        toolkit_options = ["All"] + sorted(toolkit_counts.keys())
        selected_toolkit = st.selectbox("Toolkit", toolkit_options)
        
        # Filter conversations
        filtered_conversations = filter_by_toolkit(conversations, selected_toolkit)
        st.info(f"📊 {len(filtered_conversations)} conversations")
    
    # Main content
    if len(filtered_conversations) == 0:
        st.warning("No conversations found.")
        return
    
    # Initialize session state for pagination
    if 'conv_index' not in st.session_state:
        st.session_state.conv_index = 0
    
    # Reset index if it's out of bounds
    if st.session_state.conv_index >= len(filtered_conversations):
        st.session_state.conv_index = 0
    
    # Navigation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            if st.session_state.conv_index > 0:
                st.session_state.conv_index -= 1
                st.rerun()
    
    with col2:
        st.markdown(f"<h4 style='text-align: center;'>{st.session_state.conv_index + 1} / {len(filtered_conversations)}</h4>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ➡️", use_container_width=True):
            if st.session_state.conv_index < len(filtered_conversations) - 1:
                st.session_state.conv_index += 1
                st.rerun()
    
    st.divider()
    
    # Display current conversation
    current_conversation = filtered_conversations[st.session_state.conv_index]
    display_conversation(current_conversation)


if __name__ == "__main__":
    main()

