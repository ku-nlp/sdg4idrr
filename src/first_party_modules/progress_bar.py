from functools import partial

from tqdm import tqdm

tqdm = partial(tqdm, dynamic_ncols=True)
