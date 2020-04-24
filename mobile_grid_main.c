//The C main file for a ROSS model
//This file includes:
// - definition of the LP types
// - command line argument setup
// - a main function

//includes
#include "ross.h"
#include "mobile_grid.h"


// Define LP types
//   these are the functions called by ROSS for each LP
//   multiple sets can be defined (for multiple LP types)
tw_lptype model_lps[] = {
  {
    (init_f) synchronizer_init,
    (pre_run_f) synchronizer_pre_init,
    (event_f) synchronizer_event_handler,
    (revent_f) synchronizer_event_handler_rc,
    (commit_f) NULL,
    (final_f) synchronizer_finish,
    (map_f) mobile_grid_map,
    sizeof(synchronizer_state)
  },
  {
    (init_f) channel_init,
    (pre_run_f) NULL,
    (event_f) channel_event_handler,
    (revent_f) channel_event_handler_rc,
    (commit_f) NULL,
    (final_f) channel_finish,
    (map_f) mobile_grid_map,
    sizeof(channel_state)
  },
  {
    (init_f) client_init,
    (pre_run_f) NULL,
    (event_f) client_event_handler,
    (revent_f) client_event_handler_rc,
    (commit_f) NULL,
    (final_f) client_finish,
    (map_f) mobile_grid_map,
    sizeof(client_state)
  },
  { // Dummy LP for extra lps when running on multiple nodes
    (init_f) NULL,
    (pre_run_f) NULL,
    (event_f) NULL,
    (revent_f) NULL,
    (commit_f) NULL,
    (final_f) NULL,
    (map_f) mobile_grid_map,
	0
  }, 
  { 0 },
};


/*
 *  Globals
 */
unsigned int g_num_total_lps;


/* 
 *	Command line arguments
 */
const tw_optdef model_opts[] = {
	TWOPT_GROUP("Mobile Grid Model"),
	TWOPT_UINT("num_clients", g_num_clients, "Number of clients to simulate"),

	TWOPT_UINT("mean_data_size", synchronizer_settings.mean_data_size, "Average size of data in workunit"),
	TWOPT_UINT("stddev_data_size", synchronizer_settings.stdev_data_size, "Standard deviation of size of data in workunit"),
	TWOPT_UINT("mean_flop_per_task", synchronizer_settings.mean_flop_per_task, "Average amount of work in workunit"),
	TWOPT_UINT("stddev_flop_per_task", synchronizer_settings.stdev_flop_per_task, "Standard deviation of work in workunit"),
	TWOPT_UINT("server_bandwidth", synchronizer_settings.bandwidth, "Bandwidth available to the syncrhonizer"),

	TWOPT_UINT("mean_channel_length", channel_settings.mean_length, "Average length of channel between synchronizer and client"),
	TWOPT_UINT("stdev_channel_length", channel_settings.stdev_length, "Standard deviation of length of channel between synchronizer and client"),
	TWOPT_UINT("mean_channel_bandwidth", channel_settings.mean_bandwidth, "Average bandwidth of channel between synchronizer and client"),
	TWOPT_UINT("stdev_channel_length", channel_settings.stdev_bandwidth, "Standard deviation of bandwidth of channel between synchronizer and client"),

	TWOPT_UINT("mean_flops", client_settings.mean_flops, "Average number of floating point operations per second the client is capable of"),
	TWOPT_UINT("stdev_flops", client_settings.stddev_flops, "Standard deviation of floating point operations per second the client is capable of"),

	TWOPT_END(),
};

unsigned int g_num_clients = 4;
void defaultSettings()
{
	synchronizer_settings.mean_data_size = 100;
	synchronizer_settings.stdev_data_size = 0;
	synchronizer_settings.mean_flop_per_task = 100;
	synchronizer_settings.stdev_flop_per_task = 0;
	synchronizer_settings.bandwidth = 100;

	channel_settings.mean_length = 100;
	channel_settings.stdev_length = 0;
	channel_settings.mean_bandwidth = 10;
	channel_settings.stdev_bandwidth = 0;

	client_settings.mean_flops = 1;
	client_settings.stddev_flops = 0;
}


//for doxygen
#define mobile_grid_main main

int mobile_grid_main(int argc, char* argv[]) {
	defaultSettings();

	tw_opt_add(model_opts);
	tw_init(&argc, &argv);

	//Do some error checking?
	//Print out some settings?

	//Custom Mapping
	/*
	g_tw_mapping = CUSTOM;
	g_tw_custom_initial_mapping = &model_custom_mapping;
	g_tw_custom_lp_global_to_local_map = &model_mapping_to_lp;
	*/

	//Useful ROSS variables and functions
	// tw_nnodes() : number of nodes/processors defined
	// g_tw_mynode : my node/processor id (mpi rank)

	//Useful ROSS variables (set from command line)
	// g_tw_events_per_pe
	// g_tw_lookahead
	// g_tw_nlp
	// g_tw_nkp
	// g_tw_synchronization_protocol

	g_num_total_lps = g_num_clients*2+1;
	int num_lps_per_pe = (int)ceil((float)g_num_total_lps/(float)tw_nnodes());
	printf("g_num_total_lps: %d, num_lps_per_pe: %d\n", g_num_total_lps, num_lps_per_pe);

	//set up LPs within ROSS
	tw_define_lps(num_lps_per_pe, sizeof(message));
	// note that g_tw_nlp gets set here by tw_define_lps

	// IF there are multiple LP types
	//    you should define the mapping of GID -> lptype index
	g_tw_lp_typemap = &mobile_grid_typemap;

	// set the global variable and initialize each LP's type
	g_tw_lp_types = model_lps;
	tw_lp_setup_types();

	// Do some file I/O here? on a per-node (not per-LP) basis

	tw_run();

	tw_end();

	return 0;
}
