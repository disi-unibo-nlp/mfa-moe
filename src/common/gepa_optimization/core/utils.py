from typing import Any, Dict, Optional, List
import argparse
import logging
import sys
import yaml
from pathlib import Path
import random
import numpy as np
import torch


class ConfigArgumentParser:
    """
    A wrapper around argparse that integrates YAML config files.
    This parser allows loading argument values from a YAML configuration file,
    while still permitting command-line overrides. It also validates config keys and types.

    Priority order (highest to lowest):
    1. Explicit command line arguments
    2. YAML config file values
    3. Default values specified in add_argument()
    """
    
    def __init__(self, description: str = "", config_arg_name: str = "--config", config_dir: Path = Path("./config")):
        self.parser = argparse.ArgumentParser(description=description)
        self.config_arg_name = config_arg_name
        self._arguments = {}
        
        self.parser.add_argument(
            config_arg_name,
            type=str,
            default=None,
            help="YAML configuration file path (relative to config directory)",
        )
        self.config_dir = config_dir
        
    def add_argument(self, *args, **kwargs):
        """
        Add an argument to the parser and store its specification.
        
        This wraps argparse.add_argument() and stores metadata for validation.
        """
        # Store the argument specification for later validation
        # Extract the destination name
        if 'dest' in kwargs:
            dest = kwargs['dest']
        else:
            # Parse destination from the argument name
            dest = args[0].lstrip('-').replace('-', '_')
            if args[0].startswith('--'):
                dest = args[0].lstrip('--').replace('-', '_')
            elif args[0].startswith('-'):
                dest = args[0].lstrip('-')
            else:
                dest = args[0]
        
        # Store argument metadata
        self._arguments[dest] = {
            'type': kwargs.get('type', str),
            'choices': kwargs.get('choices', None),
            'default': kwargs.get('default', None),
            'action': kwargs.get('action', 'store'),
            'nargs': kwargs.get('nargs', None)
        }
        
        # Add to the actual parser
        return self.parser.add_argument(*args, **kwargs)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate that all config keys exist in arguments and have correct types.
        """
        for key, value in config.items():
            # Skip the config argument itself
            if key == self.config_arg_name.lstrip('-').replace('-', '_'):
                continue
                
            # Check if key exists in arguments
            if key not in self._arguments:
                available_args = [k for k in self._arguments.keys() 
                                 if k != self.config_arg_name.lstrip('-').replace('-', '_')]
                raise ValueError(
                    f"Config key '{key}' is not a valid argument. "
                    f"Available arguments: {sorted(available_args)}"
                )
            
            arg_spec = self._arguments[key]
            
            # Skip validation for special actions like store_true/store_false
            if arg_spec['action'] in ['store_true', 'store_false']:
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Config key '{key}' expects a boolean value, got {type(value).__name__}"
                    )
                continue
            
            # Skip None values (will use defaults)
            if value is None:
                continue
            
            # Validate type (if specified and not using special nargs)
            if arg_spec['type'] is not None and arg_spec['nargs'] is None:
                expected_type = arg_spec['type']
                if expected_type == float and isinstance(value, (int, float)):
                    # Allow int for float fields
                    pass
                elif not isinstance(value, expected_type):
                    raise ValueError(
                        f"Config key '{key}' expects type {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
            
            # Validate choices
            if arg_spec['choices'] is not None and value not in arg_spec['choices']:
                raise ValueError(
                    f"Config key '{key}' has invalid value '{value}'. "
                    f"Valid choices: {arg_spec['choices']}"
                )
            
            # Handle list arguments (nargs)
            if arg_spec['nargs'] is not None:
                if not isinstance(value, list):
                    raise ValueError(
                        f"Config key '{key}' expects a list, got {type(value).__name__}"
                    )
                # Validate each element in the list
                if arg_spec['type'] is not None:
                    for item in value:
                        if not isinstance(item, arg_spec['type']):
                            raise ValueError(
                                f"Config key '{key}' list items expect type {arg_spec['type'].__name__}, "
                                f"got {type(item).__name__} for item {item}"
                            )
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse arguments with config file integration.

        This method:
        1. First parses to get the config file path
        2. Loads and validates the config file
        3. Merges config values with defaults
        4. Re-parses command line to override config values

        Args:
            args: List of argument strings (for testing), defaults to sys.argv

        Returns:
            argparse.Namespace with final argument values
        """
        # First parse to get config file
        initial_args, unknown = self.parser.parse_known_args(args)

        # Get config file path
        config_path = getattr(initial_args, self.config_arg_name.lstrip('-').replace('-', '_'))

        # Load config if provided
        config_values = {}
        if config_path:
            config_path = Path(config_path)

            # Add .yaml extension if not present
            if not config_path.suffix or config_path.suffix != '.yaml':
                config_path = Path(str(config_path) + '.yaml') if not config_path.suffix else config_path

            # Resolve the config path
            resolved_path = None

            if config_path.is_absolute():
                # Use absolute path as-is
                resolved_path = config_path
            else:
                # Try multiple resolution strategies for relative paths
                candidates = []

                # Strategy 1: Prepend config_dir (most common case)
                # e.g., "test_eval.yaml" -> "./config/test_eval.yaml"
                # e.g., "subdir/test.yaml" -> "./config/subdir/test.yaml"
                candidates.append(self.config_dir / config_path)

                # Strategy 2: If path starts with config_dir name, strip it and prepend actual config_dir
                # e.g., "config/test_eval.yaml" -> "./config/test_eval.yaml"
                if len(config_path.parts) > 0 and config_path.parts[0] == self.config_dir.name:
                    remaining_parts = config_path.parts[1:]
                    if remaining_parts:
                        candidates.append(self.config_dir / Path(*remaining_parts))

                # Strategy 3: Try as-is from current directory
                # e.g., if user explicitly provides "./config/test.yaml"
                candidates.append(config_path)

                # Find first existing path
                for candidate in candidates:
                    if candidate.exists():
                        resolved_path = candidate
                        break

                # If nothing found, use first candidate for error message
                if resolved_path is None:
                    resolved_path = candidates[0]

            if not resolved_path.exists():
                raise FileNotFoundError(f"Config file not found: {resolved_path}")

            config_path = resolved_path

            with open(config_path, 'r') as f:
                config_values = yaml.safe_load(f) or {}

            # Validate config
            self._validate_config(config_values)
        
        # Set defaults from config
        for key, value in config_values.items():
            if key in self._arguments:
                # Only set as default if not explicitly provided in command line
                self.parser.set_defaults(**{key: value})
        
        # Parse again with config defaults
        final_args = self.parser.parse_args(args)
        
        return final_args


def init_logger(name=None, level="INFO", output_path=None):
    """
    Initialize a simple logger with console and optional file output.

    Args:
        name: Logger name (usually __name__). If None, returns root logger.
        level: Log level as string ("DEBUG", "INFO", "WARNING", "ERROR")
        output_path: Optional path for log file. If specified, logs to both file and console.

    Returns:
        Logger instance
    """
    # IMPORTANT: Configure root logger first to set level for all loggers
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='[%(levelname)s] %(name)s - %(message)s',
        force=True  # Override any existing configuration
    )

    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if output_path is specified
    if output_path:
        # Create parent directories if they don't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(output_path, encoding='utf-8', mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    
    return logger

def set_seed(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True