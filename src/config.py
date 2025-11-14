import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigSection:
    
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return f"ConfigSection({self.__dict__})"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigSection):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class Config:
    
    _instance: Optional['Config'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.project_root = self._find_project_root()
        
        # Check for CONFIG_PATH environment variable, otherwise use default
        config_path_env = os.environ.get('CONFIG_PATH')
        if config_path_env:
            self.config_path = Path(config_path_env)
            if not self.config_path.is_absolute():
                self.config_path = self.project_root / self.config_path
        else:
            self.config_path = self.project_root / "config.yaml"
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create config.yaml in the project root or copy from config.example.yaml"
            )
        
        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        self._raw_config = self._process_env_vars(raw_config)
        
        llm_api_config = self._raw_config.get('llm_api', {})
        
        if any(key in llm_api_config for key in ['main', 'conversation', 'scorer', 'system_prompt']):
            self.llm_api = ConfigSection(llm_api_config)
        else:
            unified_config = {
                'main': llm_api_config,
                'conversation': llm_api_config,
                'scorer': llm_api_config,
                'system_prompt': llm_api_config,
            }
            self.llm_api = ConfigSection(unified_config)
        
        self.paths = self._resolve_paths(self._raw_config.get('paths', {}))
        self.generation = ConfigSection(self._raw_config.get('generation', {}))
        self.processing = ConfigSection(self._raw_config.get('processing', {}))
        self.scoring = ConfigSection(self._raw_config.get('scoring', {}))
        self.embedding = ConfigSection(self._raw_config.get('embedding', {}))
    
    def _find_project_root(self) -> Path:
        current = Path(__file__).resolve().parent
        
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        
        return Path(__file__).resolve().parent.parent
    
    def _process_env_vars(self, config: Any) -> Any:
        if isinstance(config, dict):
            return {k: self._process_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._process_env_vars(item) for item in config]
        elif isinstance(config, str):
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replace_env(match):
                env_var = match.group(1)
                default = match.group(2) if match.group(2) is not None else ""
                return os.environ.get(env_var, default)
            
            return re.sub(pattern, replace_env, config)
        else:
            return config
    
    def _resolve_paths(self, paths_config: Dict[str, Any]) -> ConfigSection:
        resolved = {}
        
        for key, value in paths_config.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_paths(value)
            elif isinstance(value, str):
                path = Path(value)
                if not path.is_absolute():
                    path = self.project_root / path
                resolved[key] = str(path)
            else:
                resolved[key] = value
        
        return ConfigSection(resolved)
    
    def get_path(self, *keys) -> Path:
        value = self.paths
        for key in keys:
            value = getattr(value, key)
        return Path(value)
    
    def reload(self):
        delattr(self, '_initialized')
        self.__init__()


config = Config()


def load_config(config_file: Optional[str] = None) -> Config:
    global config
    
    if config_file:
        config.config_path = Path(config_file)
        config.reload()
    
    return config