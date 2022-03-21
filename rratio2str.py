
#%RRATIO2STR converts the numerical R-ratio into a string without characters
#%which are incompatible with paths or filenames
#%   e.g.:   R = 0.01 -> R = 'R001'
#%           R = -1 -> R = 'Rm1'
import numpy as np

Rratio = 0.01
Rstr = 'R'

if Rratio < 0:
    Rstr += 'm'

ratioValStr = '{:g}'.format(np.abs(Rratio))
if Rratio%1 > 0: #if Rratio is a decimal number
    ratioValStr = ratioValStr.replace('.','')

print Rstr + ratioValStr

