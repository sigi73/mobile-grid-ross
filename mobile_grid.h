//The header file template for a ROSS model
//This file includes:
// - the state and message structs
// - extern'ed command line arguments
// - custom mapping function prototypes (if needed)
// - any other needed structs, enums, unions, or #defines

#ifndef _model_h
#define _model_h

#include "ross.h"

#define COORDINATOR_ID	 		0
#define MASTER_AGGREGATOR_ID 	1

#define NUM_FIXED_ACTORS		2
#define AGGREGATOR_BASE_INDEX   NUM_FIXED_ACTORS

#define SEED 1024

/*
 * Global state settings
 */

/*
 *  Command line arguments
 */

extern unsigned int g_num_clients;
extern unsigned int g_num_used_lps;

extern double g_data_center_delay;

struct s_num_actors
{
	unsigned int num_aggregators;
	unsigned int num_selectors;
	unsigned int num_clients_per_selector;
} num_actors;


struct s_channel_settings
{
	unsigned int mean_length;
	unsigned int stdev_length;
	unsigned int mean_bandwidth;
	unsigned int stdev_bandwidth;
} channel_settings;

struct s_client_settings
{
	unsigned int mean_flops;
	unsigned int stddev_flops;
} client_settings;

struct s_coordinator_settings
{
	unsigned int mean_data_size;
	unsigned int stdev_data_size;
	unsigned int mean_flop_per_task;
	unsigned int stdev_flop_per_task;
	double scheduling_interval;
	int num_tasks;
} coordinator_settings;


/*
 *  Map
 */
tw_peid mobile_grid_map(tw_lpid gid);
tw_lpid mobile_grid_typemap (tw_lpid gid);

// Returns the gid of the channel connected to the passed client
static inline tw_lpid client_to_channel(tw_lpid gid)
{
	return gid + g_num_clients;
}

// Returns the gid of the client connected to the passed channel
static inline tw_lpid channel_to_client(tw_lpid gid)
{
	return gid - g_num_clients;
}

static inline tw_lpid client_to_selector(tw_lpid gid)
{
	return + NUM_FIXED_ACTORS + num_actors.num_aggregators + (gid - num_actors.num_aggregators - num_actors.num_selectors - NUM_FIXED_ACTORS) / num_actors.num_clients_per_selector;
}

/*
 * Task
 */
typedef struct client_task client_task;
struct client_task
{
	unsigned int task_id;
	unsigned int data_size;	
	unsigned int flops;
	unsigned int results_size;

	tw_lpid aggregator_id;
};

/*
 *  Message
 */
typedef enum {
	DEVICE_AVAILABLE,
	ASSIGN_JOB,
	SCHEDULING_INTERVAL,
	TASK_ARRIVED,
	REQUEST_DATA,
	RETURN_DATA,
	SEND_RESULTS,
	NOTIFY_NEW_JOB
} message_type;
typedef struct message message;
struct message
{
	message_type type;
	client_task task;

	tw_lpid client_id;
	
	long rng_count;        // Required by ROSS rng

	unsigned int num_clients_for_task; // Used when sending a new task to aggregator
};


/*
 *  Client
 */
extern unsigned int g_num_clients;
extern unsigned int *g_client_flops;
extern float *g_client_dropout;

void setup_client_capabilities();

static inline unsigned int get_client_flops(tw_lpid gid)
{
    return g_client_flops[gid - NUM_FIXED_ACTORS - num_actors.num_aggregators - num_actors.num_selectors];
}

static inline unsigned int get_client_dropout(tw_lpid gid)
{
    return g_client_dropout[gid - NUM_FIXED_ACTORS - num_actors.num_aggregators - num_actors.num_selectors];
}

typedef struct client_state client_state;
struct client_state
{
	unsigned int flops;
};

void client_init(client_state *s, tw_lp *lp);
void client_event_handler(client_state *s, tw_bf *bf, message *m, tw_lp *lp);
void client_event_handler_rc(client_state *s, tw_bf *bf, message *m, tw_lp *lp);
void client_finish(client_state *s, tw_lp *lp);

void client_event_trace(message *m, tw_lp *lp, char *buffer, int *collect_flag);

/*
 *  Channel
 */
typedef struct channel_state channel_state;

struct channel_state
{
	unsigned int dummy; // I don't think an empty struct is valid
};

void channel_init(channel_state *s, tw_lp *lp);
void channel_event_handler(channel_state *s, tw_bf *bf, message *m, tw_lp *lp);
void channel_event_handler_rc(channel_state *s, tw_bf *bf, message *m, tw_lp *lp);
void channel_finish(channel_state *s, tw_lp *lp);

/*
 *  Coordinator
 */
#define NUM_CLIENTS_PER_TASK    5
#define NUM_TOTAL_TASKS			128
typedef struct coordinator_state coordinator_state;
struct coordinator_state
{
	unsigned int tasks_remaining; // Number of tasks to be dispatched
	unsigned int tasks_completed; // Number of tasks that have been completed
	unsigned int tasks_received;  // Number of tasks that have received so far
};

void coordinator_init(coordinator_state *s, tw_lp *lp);
void coordinator_pre_init(coordinator_state *s, tw_lp *lp);
void coordinator_event_handler(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void coordinator_event_handler_rc(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void coordinator_finish(coordinator_state *s, tw_lp *lp);
void schedule(tw_lp *lp); 
client_task* generate_map_reduce_task(int task_id, int n, tw_lp *lp);

void coordinator_event_trace(message *m, tw_lp *lp, char *buffer, int *collect_flag);

/*
 *  Selector
 */
typedef struct selector_state selector_state;
struct selector_state
{
	unsigned int num_clients;
	tw_lpid *client_gids;
};

void selector_init(selector_state *s, tw_lp *lp);
void selector_pre_init(selector_state *s, tw_lp *lp);
void selector_event_handler(selector_state *s, tw_bf *bf, message *m, tw_lp *lp);
void selector_event_handler_rc(selector_state *s, tw_bf *bf, message *m, tw_lp *lp);
void selector_finish(selector_state *s, tw_lp *lp);

/*
 *  Aggregator
 */

typedef struct aggregator_task aggregator_task;
struct aggregator_task
{
	unsigned int task_id;
	unsigned int num_remaining;
	aggregator_task *next;
};

typedef struct aggregator_state aggregator_state;
struct aggregator_state
{
	aggregator_task *tasks; //Linked list of tasks
};

void aggregator_init(aggregator_state *s, tw_lp *lp);
void aggregator_pre_init(aggregator_state *s, tw_lp *lp);
void aggregator_event_handler(aggregator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void aggregator_event_handler_rc(aggregator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void aggregator_finish(aggregator_state *s, tw_lp *lp);


#endif
