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

#define MAX_GRID_SIZE			50

#define SEED 1024

#define EPSILON 0.0001

/*
 * Global state settings
 */

/*
 *  Command line arguments
 */

extern unsigned int g_num_clients;
extern unsigned int g_num_used_lps;

extern double g_data_center_delay;
extern double g_min_delay;

struct s_num_actors
{
	unsigned int num_aggregators;
	unsigned int num_selectors;
	unsigned int num_clients_per_selector;
} num_actors;


struct s_client_settings
{
	unsigned int mean_flops;
	unsigned int stddev_flops;
	float mean_duration;
} client_settings;

struct s_coordinator_settings
{
	unsigned int mean_data_size;
	unsigned int stdev_data_size;
	unsigned int mean_flop_per_task;
	unsigned int stdev_flop_per_task;
	double scheduling_interval;
	int num_tasks;
	int task_size;
	int scheduling_algorithm;
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
	unsigned int subtask_id;
	unsigned int data_size;	
	unsigned int flops;
	unsigned int results_size;

	tw_lpid aggregator_id;
};

/*
 *  Message
 */
typedef enum {
	DEVICE_REGISTER,
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
struct s_client_parameters
{
	unsigned int *client_flops;
	float *client_start_time;
	float *client_duration;
	float *client_churn_prob;
	float *client_x;
	float *client_y;
} client_parameters;

void allocate_client_parameters();
void free_client_parameters();

static inline unsigned int get_client_flops(tw_lpid gid)
{
    return client_parameters.client_flops[gid - NUM_FIXED_ACTORS - num_actors.num_aggregators - num_actors.num_selectors];
}
static inline float get_client_start_time(tw_lpid gid)
{
    return client_parameters.client_start_time[gid - NUM_FIXED_ACTORS - num_actors.num_aggregators - num_actors.num_selectors];
}
static inline float get_client_duration(tw_lpid gid)
{
    return client_parameters.client_duration[gid - NUM_FIXED_ACTORS - num_actors.num_aggregators - num_actors.num_selectors];
}
static inline float get_client_churn_prob(tw_lpid gid)
{
    return client_parameters.client_churn_prob[gid - NUM_FIXED_ACTORS - num_actors.num_aggregators - num_actors.num_selectors];
}
static inline unsigned int get_client_x(tw_lpid gid)
{
    return client_parameters.client_x[gid - NUM_FIXED_ACTORS - num_actors.num_aggregators - num_actors.num_selectors];
}
static inline unsigned int get_client_y(tw_lpid gid)
{
    return client_parameters.client_y[gid - NUM_FIXED_ACTORS - num_actors.num_aggregators - num_actors.num_selectors];
}


typedef struct client_state client_state;
struct client_state
{
	unsigned int flops;
	float duration;
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
	float *x_pos;		// X and y positions of associated client
	float *y_pos;
	float *capacity;	// Channel capacity calculated
	// Make these arrays of length 1 since the gpu code is generalized to calculate for multiple clients. Possible extension: multiple clients sharing a channel
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

typedef struct task_node task_node;
struct task_node
{
	client_task* task;
	task_node* next;
};

typedef struct worker worker;
struct worker
{
	tw_lpid client_id;
	unsigned int flops;
	double duration;
	double churn_prob;
	int assigned;   // 0 -> unassigned, 1 -> assigned
};

typedef struct worker_node worker_node;
struct worker_node
{
	worker* worker;
	worker_node* next;
};

typedef struct coordinator_worker coordinator_worker;
struct coordinator_worker 
{
	worker* worker;
	double comp_time;
	//unsigned int churn_prob;
	double churn_prob;
};

/*
typedef struct assignment;
struct assignment
{
	client_task* task;
	worker* worker;
};
*/

struct coordinator_state
{
	unsigned int tasks_remaining; // Number of tasks to be dispatched
	unsigned int tasks_completed; // Number of tasks that have been completed
	unsigned int tasks_received;  // Number of tasks that have received so far
	unsigned int tasks_started;  // Number of tasks that the coordinator has started scheduling 
	task_node* task_stage;      // Head of linked list of tasks ready to be distributed in the next round
	//worker** workers;	
	worker_node* workers;
	int num_workers;
};

void coordinator_init(coordinator_state *s, tw_lp *lp);
void coordinator_pre_init(coordinator_state *s, tw_lp *lp);
void coordinator_event_handler(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void coordinator_event_handler_rc(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void coordinator_finish(coordinator_state *s, tw_lp *lp);

void schedule(coordinator_state *s, tw_lp *lp); 
worker* schedule_naive(client_task* task, coordinator_state *s, tw_lp *lp);
worker* schedule_rcta(client_task* task, worker** workers, int num_workers, coordinator_state *s, tw_lp *lp);

client_task* generate_map_reduce_task(int task_id, int n, tw_lp *lp);

void stage_task(task_node* head, client_task* task);
client_task* pop_task(task_node* head);
void free_task_stage(task_node* head);

void add_worker(worker_node* head, worker* worker);
void delete_worker(worker_node* head, tw_lpid client_id);
worker* pop_worker(worker_node* head);
void free_workers(worker_node* head);

worker** convert_workers_to_array(worker_node* head, int count);
coordinator_worker** merge_sort_workers(coordinator_worker** workers, int n);

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
