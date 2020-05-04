import sys
import struct
from enum import Enum

from event_trace import *

data_fn = sys.argv[1]
parser = EventTrace.from_file(data_fn)

config_fn = sys.argv[2]
config_file = open(config_fn)
config = config_file.readlines()

NUM_AGGREGATORS = int(config[0].split(':')[1])
NUM_SELECTORS = int(config[0].split(':')[1])
NUM_CLIENTS_PER_SELECTOR = int(config[0].split(':')[1])
SIZE_OF_MESSAGE_TYPE = int(config[0].split(':')[1])

COORDINATOR_ID = 0
MASTER_AGGREGATOR_ID = 1
aggregator_ids = set()
selector_ids = set()
client_ids = set()
channel_ids = set()

counter = MASTER_AGGREGATOR_ID + 1
for i in range(NUM_AGGREGATORS):
    aggregator_ids.add(counter + i)
counter = counter + NUM_AGGREGATORS
for i in range(NUM_SELECTORS):
    selector_ids.add(counter + i)
counter = counter + NUM_SELECTORS
for i in range(NUM_SELECTORS*NUM_CLIENTS_PER_SELECTOR):
    client_ids.add(counter + i)
counter = counter + NUM_SELECTORS*NUM_CLIENTS_PER_SELECTOR
for i in range(NUM_SELECTORS*NUM_CLIENTS_PER_SELECTOR):
    channel_ids.add(counter + i)

print('Aggregator ids: ' + str(aggregator_ids))
print('Selector ids: ' + str(selector_ids))
print('Client ids: ' + str(client_ids))
print('Channel ids: ' + str(channel_ids))

class MessageType(Enum):
    DEVICE_REGISTER = 1
    DEVICE_AVAILABLE = 2
    ASSIGN_JOB = 3
    SCHEDULING_INTERVAL = 4
    TASK_ARRIVED = 5
    REQUEST_DATA = 6
    RETURN_DATA = 7
    SEND_RESULTS = 8
    NOTIFY_NEW_JOB = 9

for event in parser.meta:
    if event.model_data_size == 0:
        continue
    message_type = struct.unpack('=I', event.buf[0:4]) 
    t = MessageType(message_type[0])
    print(t)
    print(event.source_lp)
    print(event.destination_lp)
    print(event.virtual_send_time)
    print(event.virtual_recv_time)
    print(event.real_time)
