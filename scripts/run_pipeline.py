import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_project_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / "config.yaml").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def create_outputs_directory(project_root):
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    print(f"✓ Outputs directory ready: {outputs_dir}")
    return outputs_dir


def run_script(script_path, script_name, env):
    print("\n" + "="*80)
    print(f"Running: {script_name}")
    print("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            check=True,
            cwd=script_path.parent.parent  # Run from project root
        )
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_name} failed with exit code {e.returncode}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Error running {script_name}: {str(e)}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the full conversation generation pipeline"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = find_project_root()
    print(f"Project root: {project_root}")
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using config: {config_path}")
    
    # Create outputs directory
    create_outputs_directory(project_root)
    
    # Prepare environment with CONFIG_PATH
    env = os.environ.copy()
    env['CONFIG_PATH'] = str(config_path)
    
    # Define the scripts to run in order
    scripts = [
        (project_root / "src" / "generate_conversations.py", "generate_conversations.py"),
        (project_root / "src" / "generate_system_prompt.py", "generate_system_prompt.py"),
        (project_root / "src" / "score_conversations.py", "score_conversations.py"),
        (project_root / "src" / "reject_low_quality.py", "reject_low_quality.py"),
        (project_root / "src" / "parse_conversations.py", "parse_conversations.py"),
    ]
    
    # Run each script in sequence
    print("\n" + "="*80)
    print("STARTING PIPELINE")
    print("="*80)
    
    for script_path, script_name in scripts:
        if not script_path.exists():
            print(f"✗ Script not found: {script_path}", file=sys.stderr)
            sys.exit(1)
        
        success = run_script(script_path, script_name, env)
        if not success:
            print("\n" + "="*80)
            print("PIPELINE FAILED")
            print("="*80)
            print(f"Pipeline stopped due to failure in {script_name}")
            sys.exit(1)
    
    # Success!
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nAll steps completed successfully!")
    print(f"Output files are in: {project_root / 'outputs'}")


if __name__ == "__main__":
    main()

