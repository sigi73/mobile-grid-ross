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
 *  Command line arguments
 */
extern unsigned int g_num_clients;
extern unsigned int g_num_total_lps;

struct s_synchronizer_settings
{
	unsigned int mean_data_size;
	unsigned int stdev_data_size;
	unsigned int mean_flop_per_task;
	unsigned int stdev_flop_per_task;
	unsigned int bandwidth;
} synchronizer_settings;

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

/*
 *  Map
 */
tw_peid mobile_grid_map(tw_lpid gid);
tw_lpid mobile_grid_typemap (tw_lpid gid);

/*
 *  Message
 */
typedef enum {
  SYNCH_TO_CHANNEL,
  CHANNEL_TO_CLIENT,
  CLIENT_TO_CHANNEL,
  CHANNEL_TO_SYNCH
} message_type;
typedef struct message message;
struct message
{
	message_type type;
    unsigned int task_data_size;
    unsigned int task_flops;
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
 *  Synchronizer
 */
typedef struct synchronizer_state synchronizer_state;
struct synchronizer_state
{
	unsigned int tasks_remaining; // Number of tasks to be dispatched
	unsigned int tasks_completed; // Number of tasks that have been completed
};

void synchronizer_init(synchronizer_state *s, tw_lp *lp);
void synchronizer_pre_init(synchronizer_state *s, tw_lp *lp);
void synchronizer_event_handler(synchronizer_state *s, tw_bf *bf, message *m, tw_lp *lp);
void synchronizer_event_handler_rc(synchronizer_state *s, tw_bf *bf, message *m, tw_lp *lp);
void synchronizer_finish(synchronizer_state *s, tw_lp *lp);

#endif
