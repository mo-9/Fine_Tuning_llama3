import os
import json
import logging
from typing import Dict, List
from datetime import datetime

class ModelRegistry:
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        self.registry_file = os.path.join(registry_path, "registry.json")
        self.logger = logging.getLogger(__name__)
        
        # Initialize registry file if it doesn't exist
        if not os.path.exists(self.registry_file):
            self._save_registry({})

    def _load_registry(self) -> Dict:
        """Load the model registry from file."""
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_registry(self, registry: Dict):
        """Save the model registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

    def register_model(self, model_name: str, version: str, model_path: str, 
                      metadata: Dict = None) -> str:
        """Register a new model version."""
        registry = self._load_registry()
        
        if model_name not in registry:
            registry[model_name] = {}
        
        model_id = f"{model_name}:{version}"
        registry[model_name][version] = {
            "model_path": model_path,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "status": "registered"
        }
        
        self._save_registry(registry)
        self.logger.info(f"Registered model {model_id}")
        return model_id

    def get_model(self, model_name: str, version: str = "latest") -> Dict:
        """Get model information by name and version."""
        registry = self._load_registry()
        
        if model_name not in registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version == "latest":
            # Get the most recent version
            versions = list(registry[model_name].keys())
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            version = max(versions)  # Assumes semantic versioning
        
        if version not in registry[model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        return registry[model_name][version]

    def list_models(self) -> List[Dict]:
        """List all registered models."""
        registry = self._load_registry()
        models = []
        
        for model_name, versions in registry.items():
            for version, info in versions.items():
                models.append({
                    "model_name": model_name,
                    "version": version,
                    **info
                })
        
        return models

    def update_model_status(self, model_name: str, version: str, status: str):
        """Update the status of a model version."""
        registry = self._load_registry()
        
        if model_name in registry and version in registry[model_name]:
            registry[model_name][version]["status"] = status
            self._save_registry(registry)
            self.logger.info(f"Updated status of {model_name}:{version} to {status}")
        else:
            raise ValueError(f"Model {model_name}:{version} not found")

