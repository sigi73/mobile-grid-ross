#include "mobile_grid.h"

void coordinator_init(coordinator_state *s, tw_lp *lp)
{
   // Initialize coordinator state

   printf("Initializing coordinator, gid: %u\n", lp->gid);
}

void coordinator_pre_init(coordinator_state *s, tw_lp *lp)
{
}

void coordinator_event_handler(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
}

void coordinator_event_handler_rc(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
}

void coordinator_finish(coordinator_state *s, tw_lp *lp)
{
}
