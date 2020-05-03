#include "mobile_grid.h"

void channel_init(channel_state *s, tw_lp *lp)
{
   // TODO: Use stddev as well
   s->length = channel_settings.mean_length;
   s->bandwidth = channel_settings.mean_bandwidth;

   printf("Initializing channel, gid: %u (Client %u)\n", lp->gid, channel_to_client(lp->gid));
}

void channel_event_handler(channel_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    if (m-> type == ASSIGN_JOB)
    {
        // For now just a delay
        // TODO: Replace with TLM propogation simulation
        double temp_channel_delay = 5;
        tw_event *e = tw_event_new(m->client_id, temp_channel_delay, lp);
        message *msg = tw_event_data(e);

        msg->type = ASSIGN_JOB;
        msg->task = m->task;
        msg->client_id = m->client_id;

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
