import os 
from typing import Dict, Optional 
 
 
ALLOWED_ENVS = ("off", "local", "dev", "qa", "prod") 
_ENV_ORDER: Dict[str, int] = { 
    "off": 0, 
    "local": 1, 
    "dev": 2, 
    "qa": 3, 
    "prod": 4, 
} 
 
 
# Define feature gates as the minimum environment where the feature is enabled. 
# Edit this mapping to add your features with one of: "off", "local", "dev", "qa", "prod". 
# When a feature gets to prod, it should be removed from this mapping. 
# Set the environment variable WISER_ENV to the desired environment to enable the features. 
FEATURE_GATES = { 
    "sff": "prod", 
    "sam": "prod", 
    "sff_sam_image_cube": "dev",
} 
 
 
def _normalize_env(env_value: Optional[str]) -> str: 
    if not env_value: 
        return "local" 
    env = env_value.strip().lower() 
    return env if env in ALLOWED_ENVS else "local" 
 
 
class FeatureFlags: 
    """Singleton-like feature flag accessor using minimum-level gates. 
 
    - Current environment is taken from WISER_ENV (normalized, default "local"). 
    - Each feature's configured value is one of: "off", "local", "dev", "qa", "prod". 
    - A feature is enabled if current_env >= feature_min_level according to _ENV_ORDER. 
    - Unknown features resolve to False. 
    - Supports attribute-style access: FLAGS.sff 
    """  
 
    def __init__(self) -> None: 
        self._current_env: str = _normalize_env(os.environ.get("WISER_ENV")) 
        self._feature_gates: Dict[str, str] = dict(FEATURE_GATES) 
 
    # Public API 
    @property 
    def env(self) -> str: 
        return self._current_env 
 
    def configure(self, env: Optional[str] = None, feature_gates: Optional[Dict[str, str]] = None) -> None: 
        if env is not None: 
            self._current_env = _normalize_env(env) 
        if feature_gates is not None: 
            # Replace entire table to avoid stale entries 
            self._feature_gates = {k.strip().lower(): _normalize_env(v) for k, v in feature_gates.items()} 
 
    def set_feature_level(self, feature_name: str, min_env: str) -> None: 
        self._feature_gates[feature_name.strip().lower()] = _normalize_env(min_env) 
 
    def is_enabled(self, feature_name: str) -> bool: 
        feature = feature_name.strip().lower() 
        feature_level: str = self._feature_gates.get(feature, "off") 
        # Its only enabled if the flags level is greater than the current
        # environment's level
        return _ENV_ORDER[feature_level] >= _ENV_ORDER[self._current_env] 
 
    def to_dict(self) -> Dict[str, bool]: 
        # Snapshot of enabled booleans for current env 
        return {f: self.is_enabled(f) for f in self._feature_gates} 
 
    # Convenience attribute access: FLAGS.sff 
    def __getattr__(self, name: str) -> bool: 
        return self.is_enabled(name) 
 
 
# Process-wide singleton instance 
FLAGS = FeatureFlags() 
 
 
def set_feature_env(env: str) -> None: 
    """Set the active feature environment and export to process env.""" 
    normalized = _normalize_env(env) 
    os.environ["WISER_ENV"] = normalized 
    FLAGS.configure(env=normalized) 
 
