"""
Configuration module for the AI agent laboratory.

This code was developed with the assistance of Claude Code.
"""
import os
import yaml
from typing import Dict, List, Optional, Any


class LabConfig:
    """Configuration for the AI agent laboratory."""

    def __init__(self, config_path: str):
        """
        Initialize configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration file has all required fields.
        
        Args:
            config: Configuration dictionary.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        required_fields = ['user', 'anthropic_api_key', 'agents']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        # Validate user configuration
        if 'name' not in config['user']:
            raise ValueError("User configuration must include a name")
        
        # Validate agent configurations
        if not config['agents'] or not isinstance(config['agents'], list):
            raise ValueError("Configuration must include a list of agents")
        
        for i, agent in enumerate(config['agents']):
            if 'name' not in agent:
                raise ValueError(f"Agent at index {i} missing required field: name")
            if 'specialty' not in agent:
                raise ValueError(f"Agent at index {i} missing required field: specialty")
    
    @property
    def user_name(self) -> str:
        """Get the user's name from config."""
        return self.config['user']['name']
    
    @property
    def anthropic_api_key(self) -> str:
        """Get the Anthropic API key from config."""
        return self.config['anthropic_api_key']
    
    @property
    def model(self) -> str:
        """Get the model name from config, defaulting to claude-3-5-sonnet-20240620."""
        return self.config.get('model', 'claude-3-5-sonnet-20240620')
    
    @property
    def agents(self) -> List[Dict[str, str]]:
        """Get the list of agent configurations."""
        return self.config['agents']
    
    @property
    def documents_dir(self) -> Optional[str]:
        """Get the documents directory if provided."""
        return self.config.get('documents_dir')


