//The header file template for a ROSS model
//This file includes:
// - the state and message structs
// - extern'ed command line arguments
// - custom mapping function prototypes (if needed)
// - any other needed structs, enums, unions, or #defines

#ifndef _model_h
#define _model_h

#include "ross.h"


/*
 * Global state settings
 */
extern tw_lpid g_coordinator_id;

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

} coordinator_settings;


/*
 *  Map
 */
tw_peid mobile_grid_map(tw_lpid gid);
tw_lpid mobile_grid_typemap (tw_lpid gid);

/*
 * Task
 */
typedef struct client_task client_task;
struct client_task
{
	unsigned int task_id;
	unsigned int data_size;	
	unsigned int flops;
};

/*
 *  Message
 */
typedef enum {
	DEVICE_AVAILABLE,
	ASSIGN_JOB
} message_type;
typedef struct message message;
struct message
{
	message_type type;
	client_task task;

	tw_lpid client_id;
};


/*
 *  Client
 */
typedef struct client_state client_state;
struct client_state
{
	unsigned int flops;
};

void client_init(client_state *s, tw_lp *lp);
void client_event_handler(client_state *s, tw_bf *bf, message *m, tw_lp *lp);
void client_event_handler_rc(client_state *s, tw_bf *bf, message *m, tw_lp *lp);
void client_finish(client_state *s, tw_lp *lp);

/*
 *  Channel
 */
typedef struct channel_state channel_state;

struct channel_state
{
	unsigned int length;
	unsigned int bandwidth;
};

void channel_init(channel_state *s, tw_lp *lp);
void channel_event_handler(channel_state *s, tw_bf *bf, message *m, tw_lp *lp);
void channel_event_handler_rc(channel_state *s, tw_bf *bf, message *m, tw_lp *lp);
void channel_finish(channel_state *s, tw_lp *lp);

/*
 *  Coordinator
 */
typedef struct coordinator_state coordinator_state;
struct coordinator_state
{
	unsigned int tasks_remaining; // Number of tasks to be dispatched
	unsigned int tasks_completed; // Number of tasks that have been completed
};

void coordinator_init(coordinator_state *s, tw_lp *lp);
void coordinator_pre_init(coordinator_state *s, tw_lp *lp);
void coordinator_event_handler(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void coordinator_event_handler_rc(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void coordinator_finish(coordinator_state *s, tw_lp *lp);

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
};

typedef struct aggregator_state aggregator_state;
struct aggregator_state
{
	unsigned int num_tasks;
	aggregator_task *tasks; //Pointer to array of tasks
};

void aggregator_init(aggregator_state *s, tw_lp *lp);
void aggregator_pre_init(aggregator_state *s, tw_lp *lp);
void aggregator_event_handler(aggregator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void aggregator_event_handler_rc(aggregator_state *s, tw_bf *bf, message *m, tw_lp *lp);
void aggregator_finish(aggregator_state *s, tw_lp *lp);

/*
 *  Data server
 */
typedef struct data_server_state data_server_state;
struct data_server_state
{
	unsigned char dummy; // I don't think empty structs are allowed. No state
};

void data_server_init(data_server_state *s, tw_lp *lp);
void data_server_pre_init(data_server_state *s, tw_lp *lp);
void data_server_event_handler(data_server_state *s, tw_bf *bf, message *m, tw_lp *lp);
void data_server_event_handler_rc(data_server_state *s, tw_bf *bf, message *m, tw_lp *lp);
void data_server_finish(data_server_state *s, tw_lp *lp);

#endif
