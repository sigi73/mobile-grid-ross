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
 *  Map
 */
tw_peid mobile_grid_map(tw_lpid gid);

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
    unsigned long int task_data_size;
    unsigned long int task_flops;
};

/*
 *  Client
 */
typedef struct client_state client_state;
struct client_state
{
	unsigned long int flops_per_task;
	//long int dropout_chance;
	unsigned int has_work : 1;
};

void client_init(client_state *s, tw_lp *lp);
void client_event_handler(client_state *s, tw_bf *bf, message *m, tw_lp *lp);
void client_event_handler_rc(client_state *s, tw_bf *bf, message *m, tw_lp *lp);
void client_finish(client_state *s, tw_lp *lp);
//void client_pre_run(client_state *s, tw_lp *lp);
//void client_commit(client_state *s, tw_bf *bf, client_message *m, tw_lp *lp);

/*
 *  Channel
 */
typedef struct channel_state channel_state;

struct channel_state
{
	unsigned long int channel_length;
	unsigned int client_gid;
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
	unsigned long int tasks_remaining; // Number of tasks to be dispatched

	unsigned long int mean_data_size; // Size of data to be transferred. Changes time to send to client
	unsigned long int stddev_dat_size;
	unsigned long int mean_flops_per_task; // How long the computation takes
	unsigned long int stddev_flops_per_task;
};

void synchronizer_init(synchronizer_state *s, tw_lp *lp);
void synchronizer_event_handler(synchronizer_state *s, tw_bf *bf, message *m, tw_lp *lp);
void synchronizer_event_handler_rc(synchronizer_state *s, tw_bf *bf, message *m, tw_lp *lp);
void synchronizer_finish(synchronizer_state *s, tw_lp *lp);

#endif
