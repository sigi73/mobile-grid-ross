#include "mobile_grid.h"

void channel_init(channel_state *s, tw_lp *lp)
{
   // TODO: Use stddev as well
   s->length = channel_settings.mean_length;
   s->bandwidth = channel_settings.mean_bandwidth;

   printf("Initializing channel, gid: %u\n", lp->gid);
}

void channel_event_handler(channel_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    if (m->type == SYNCH_TO_CHANNEL)
    {
        printf("Channel with gid %u received message from synchronizer\n", lp->gid);
        tw_stime delay = 1;
        // Send to client with delay based on bandwidth, size, and length
        tw_event *e = tw_event_new(lp->gid+1, delay, lp);
        message *msg = tw_event_data(e);

        msg->type = CHANNEL_TO_CLIENT;
        msg->task_data_size = m->task_data_size;
        msg->task_flops = m->task_flops;

        tw_event_send(e);
    }
    if (m->type == CLIENT_TO_CHANNEL)
    {
        printf("Channel with gid %u received message from a client\n", lp->gid);
        tw_stime delay = 1;
        // Send to syncrhonizer with delay based on bandwidth, size, and length
        tw_event *e = tw_event_new(0, delay, lp);
        message *msg = tw_event_data(e);

        msg->type = CHANNEL_TO_SYNCH;
        msg->task_data_size = m->task_data_size;
        msg->task_flops = m->task_flops;

        tw_event_send(e);
    }
}

void channel_event_handler_rc(channel_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    // No internal state to be updated
    (void) s;
    (void) bf;
    (void) m;
    (void) lp;
}

void channel_finish(channel_state *s, tw_lp *lp)
{
    (void)s;
    (void)lp;
}