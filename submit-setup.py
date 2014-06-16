#!/usr/bin/env python

import numpy as np
import os.path

submit_template = """seed = {seed:d}
multiple_name = {name:s}

trigger_path = /home/yiminghu/RatesNSignificance/$(multiple_name)
output_path_base = /home/w-farr/Hamlet/runs/$(multiple_name)
exec_path = /home/w-farr/src/LIGOHamlet

output_path = $(output_path_base)/$(seed)

Universe = vanilla
Executable = $(exec_path)/run.py

Getenv = True

Log = $(output_path)/rates.log
Output = $(output_path)/rates.out
Error = $(output_path)/rates.err

Arguments = "--coincs $(trigger_path)/CoincTriggers_seed$(seed).txt --trigs1 $(trigger_path)/Triggers1_seed$(seed).txt --trigs2 $(trigger_path)/Triggers2_seed$(seed).txt --outdir $(output_path)"

Queue
"""

if __name__ == __main__:
    for iname in range(1, 15):
        if iname == 1:
            name = 'Multiple'
        else:
            name = '{:d}Multiple'.format(iname)

        for j in range(10):
            seed = np.random.randint(100000)

            opath = '/home/w-farr/Hamlet/runs/{name:s}/{seed:d}'.format(name=name, seed=seed)

            os.path.makedirs(opath)
            with open(os.path.join(opath, 'submit'), 'w') as out:
                out.write(submit_template.format(seed=seed, name=name))
