class Backend:
    def __init__(self):
        self.backend = None

    def set_backend(self, name):
        if name == "torch":
            from src.backends import torch_backend
            self.backend = torch_backend
        elif name == "python":
            from src.backends import python_backend
            self.backend = python_backend
        else:
            raise ValueError(f"Unknown backend: {name}")

    def __getattr__(self, name):
        return getattr(self.backend, name)

backend = Backend()
backend.set_backend("torch")
