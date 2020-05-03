import sys
import struct

from event_trace import *

fn = sys.argv[1]
parser = EventTrace.from_file(fn)

for i in range(10):
    lp_id = struct.unpack('=q', parser.meta[i].buf)
    print(lp_id[0])
