import sys
import struct
from enum import Enum
import matplotlib.pyplot as plt

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

num_scheduling = 0
scheduling_times = []
clients_start_time = {} # Dictionary: id: (Startime, duration)

aggregator_tasks = {} # id: [{task_id: [time_added, num_clients, [times_decremented]]}]
aggregator_tasks[MASTER_AGGREGATOR_ID] = dict()
for agg in aggregator_ids:
    aggregator_tasks[agg] = dict()

clients2 = {} # id: {task_id: [recv time, job start, job end, upload done]}
for cli in client_ids:
    clients2[cli] = dict()

num_zero = 0
zero_msg_set = set()

index = 0
for event in parser.meta:
    if event.model_data_size == 0:
        continue
    #print(event.buf)
    #print(event.buf[0:4])
    message_type = struct.unpack('=I', event.buf[0:4]) 
    t = MessageType(message_type[0])
    #print(t)
    #print(index)
    if event.source_lp == 0 and event.destination_lp == COORDINATOR_ID and t == MessageType.SCHEDULING_INTERVAL:
        num_scheduling = num_scheduling+1
        scheduling_times.append(event.virtual_recv_time)
    if event.source_lp in selector_ids and event.destination_lp == COORDINATOR_ID and t == MessageType.DEVICE_REGISTER:
        client_id = struct.unpack('=I', event.buf[4:8])
        client_id = client_id[0]
        clients_start_time[client_id] = event.virtual_recv_time
    
    if (event.destination_lp == MASTER_AGGREGATOR_ID or event.destination_lp in aggregator_ids) and t == MessageType.NOTIFY_NEW_JOB:
        (task_id, num_clients) = struct.unpack('=II', event.buf[4:12])
        aggregator_tasks[event.destination_lp][task_id] = [event.virtual_recv_time, num_clients, []]
    
    if (event.destination_lp == MASTER_AGGREGATOR_ID or event.destination_lp in aggregator_ids) and t == MessageType.SEND_RESULTS:
        task_id = struct.unpack('=I', event.buf[4:8])
        task_id = task_id[0]
        if task_id in aggregator_tasks[event.destination_lp].keys():
            aggregator_tasks[event.destination_lp][task_id][2].append(event.virtual_recv_time)

    if event.destination_lp in aggregator_ids and event.source_lp in channel_ids and t == MessageType.SEND_RESULTS:
        task_id = struct.unpack('=I', event.buf[4:8])
        task_id = task_id[0]
        client_id = event.source_lp - NUM_CLIENTS_PER_SELECTOR*NUM_SELECTORS
        clients2[client_id][task_id][3] = event.virtual_recv_time
    
    if event.destination_lp in client_ids and t == MessageType.ASSIGN_JOB:
        task_id = struct.unpack('=I', event.buf[4:8])
        task_id = task_id[0]
        clients2[event.destination_lp][task_id] = [event.virtual_recv_time, -1, -1, -1]

    if event.destination_lp in client_ids:
        pass
        #print(t)
    
    if event.destination_lp in client_ids and t == MessageType.RETURN_DATA:
        task_id = struct.unpack('=I', event.buf[4:8])
        task_id = task_id[0]
        clients2[event.destination_lp][task_id][1] = event.virtual_recv_time

    if event.destination_lp in channel_ids and t == MessageType.SEND_RESULTS:
        (task_id, client_id) = struct.unpack('=II', event.buf[4:12])
        clients2[client_id][task_id][2] = event.virtual_recv_time
    index = index + 1

#print(clients2)    

'''
print('Number of scheduling events: ' + str(num_scheduling))
print(clients)

print('Master aggregator:')
print (aggregator_tasks[MASTER_AGGREGATOR_ID])

for agg in aggregator_ids:
    print('Aggregator ' + str(agg))
    print(aggregator_tasks[agg])
    print(' \n')
'''

index = 1
for c in clients2.keys():
    for task in clients2[c].keys():
        y_axis = [index] * 2
        arr = clients2[c][task]
        plt.plot(arr[0:2], y_axis, 'b') # Download
        plt.plot(arr[1:3], y_axis, 'r') # Compute
        plt.plot(arr[2:4], y_axis, 'g') # Upload
    index = index+1
plt.show()

print(aggregator_tasks)