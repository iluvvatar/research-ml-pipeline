import os
from typing import Union
import re


PathLike = Union[str, bytes, os.PathLike]
spaces_pattern = re.compile(r'[\u00A0\u007F\u1680\u180e\u2000-\u200d\u2028\u2029\u202f\u205f\u2060\u3000\ufeff]')
# clear_pattern = re.compile(r'[\u00AD]')
