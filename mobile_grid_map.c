//The C mapping for a ROSS model
//This file includes:
// - the required LP GID -> PE mapping function
// - Commented out example of LPType map (when there multiple LP types)
// - Commented out example of one set of custom mapping functions:
//   - setup function to place LPs and KPs on PEs
//   - local map function to find LP in local PE's array

#include "mobile_grid.h"

//Given an LP's GID (global ID)
//return the PE (aka node, MPI Rank)
tw_peid mobile_grid_map(tw_lpid gid){
  //printf("gid: %u, g_tw_nlp: %u, node: %u\n", gid, g_tw_nlp, (tw_peid) gid / g_tw_nlp);
  return (tw_peid) gid / g_tw_nlp;
}

// Multiple LP Types mapping function
//    Given an LP's GID
//    Return the index in the model_lps array (defined in mobile_grid_main.c)
tw_lpid mobile_grid_typemap (tw_lpid gid) {
  if (gid < g_num_used_lps)
  {
    if (gid == 0)
      return 0; // Coordinator
    else if (gid == 1)
      return 1; // Master Aggregator
    else if (gid <= (NUM_FIXED_ACTORS - 1) + num_actors.num_aggregators)
      return 1; // Aggregator
    else if (gid <= (NUM_FIXED_ACTORS - 1) + num_actors.num_aggregators + num_actors.num_selectors)
      return 2; // Selector
    else if (gid <= (NUM_FIXED_ACTORS - 1) + num_actors.num_aggregators + num_actors.num_selectors + g_num_clients)
      return 3; // Client
    else
      return 4; // Channel
  }
  else
  {
    return 5;
  }
  
  /*
  if (gid < g_num_total_lps)
  {
    if (gid == 0)
      return 0; // gid 0 is synchronizer
    else if (gid % 2 == 1)
      return 1; // Odd gids are channels
    else
      return 2; // Remaning even gids are clients
  }
  else
  {
    return 3; // Extra lps are unused
  }
  */
  
}

/*
// Multiple LP Types mapping function
//    Given an LP's GID
//    Return the index in the LP type array (defined in model_main.c)
tw_lpid model_typemap (tw_lpid gid) {
  // since this model has one type
  // always return index of 1st LP type
  return 0;
}
*/

/*
//Custom mapping functions are used so
// - no LPs are unused
// - event activity is balanced

extern unsigned int nkp_per_pe;
//#define VERIFY_MAPPING 1 //useful for debugging

//This function maps LPs to KPs on PEs and is called at the start
//This example is the same as Linear Mapping
void model_custom_mapping(void){
  tw_pe *pe;
  int nlp_per_kp;
  int lp_id, kp_id;
  int i, j;

  // nlp should be divisible by nkp (no wasted LPs)
  nlp_per_kp = ceil((double) g_tw_nlp / (double) g_tw_nkp);
  if (!nlp_per_kp) tw_error(TW_LOC, "Not enough KPs defined: %d", g_tw_nkp);

  //gid of first LP on this PE (aka node)
  g_tw_lp_offset = g_tw_mynode * g_tw_nlp;

#if VERIFY_MAPPING
  prinf("NODE %d: nlp %lld, offset %lld\n", g_tw_mynode, g_tw_nlp, g_tw_lp_offset);
#endif

  // Loop through each PE (node)
  for (kp_id = 0, lp_id = 0, pe = NULL; (pe = tw_pe_next(pe)); ) {

    // Define each KP on the PE
    for (i = 0; i < nkp_per_pe; i++, kp_id++) {

      tw_kp_onpe(kpid, pe);

      // Define each LP on the KP
      for (j = 0; j < nlp_per_kp && lp_id < g_tw_nlp; j++, lp_id++) {

	tw_lp_onpe(lp_id, pe, g_tw_lp_offset + lp_id);
	tw_lp_onkp(g_tw_lp[lp_id], g_tw_kp[kp_id]);

#if VERIFY_MAPPING
	if (0 == j % 20) { // print detailed output for only some LPs
	  printf("PE %d\tKP %d\tLP %d\n", pe->id, kp_id, (int) lp_id + g_tw_lp_offset);
	}
#endif
      }
    }
  }

  //Error checks for the mapping
  if (!g_tw_lp[g_tw_nlp - 1]) {
    tw_error(TW_LOC, "Not all LPs defined! (g_tw_nlp=%d)", g_tw_nlp);
  }

  if (g_tw_lp[g_tw_nlp - 1]->gid != g_tw_lp_offset + g_tw_nlp - 1) {
    tw_error(TW_LOC, "LPs not sequentially enumerated");
  }
}

//Given a gid, return the local LP (global id => local id mapping)
tw_lp * model_mapping_to_lp(tw_lpid){
  int local_id = lp_id - g_tw_offset;
  return g_tw_lp[id];
}
*/
