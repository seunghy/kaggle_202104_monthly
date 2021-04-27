import datetime
import json
from collections import OrderedDict
import os

def make_log(update_dict,  path_):
    if os.path.exists(path_):
        with open(path_,mode='r+') as f:
            data=json.load(f)
        data.update(update_dict)
        with open(path_,'w+') as f:
            json.dump(data,f)
    else:
        with open(path_, mode='w+') as f:
            json.dump(update_dict,f)