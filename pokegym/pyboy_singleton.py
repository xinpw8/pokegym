# pyboy_singleton.py
from pyboy import PyBoy
from io import BytesIO

_pyboy_instance = None

def get_pyboy_instance(gb_path=None, symbols_path=None, headless=False):
    global _pyboy_instance
    if _pyboy_instance is None:
        if gb_path is None or symbols_path is None:
            raise ValueError("gb_path and symbols_path must be provided for the first initialization")
        
        pyboy_kwargs = {
            "window": "null" if headless else "SDL2",
            "symbols": symbols_path,
        }
        
        _pyboy_instance = PyBoy(gb_path, **pyboy_kwargs)
    return _pyboy_instance

def load_state(state_path):
    global _pyboy_instance
    if _pyboy_instance is None:
        raise RuntimeError("PyBoy instance is not initialized. Call get_pyboy_instance first.")
    
    with open(state_path, 'rb') as f:
        state = BytesIO(f.read())
    
    state.seek(0)
    _pyboy_instance.load_state(state)
