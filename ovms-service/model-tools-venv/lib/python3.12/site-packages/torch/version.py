from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.12.0+cpu'
debug = False
cuda: Optional[str] = None
git_version = '7661cd9c6b841b62b7f411aa52ec51f05457263b'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
