#include "mobile_grid.h"

void aggregator_init(aggregator_state *s, tw_lp *lp)
{
   printf("Initializing aggregator, gid: %u\n", lp->gid);
}

void aggregator_pre_init(aggregator_state *s, tw_lp *lp)
{

}

void aggregator_event_handler(aggregator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
}

void aggregator_event_handler_rc(aggregator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{

}

void aggregator_finish(aggregator_state *s, tw_lp *lp)
{

}