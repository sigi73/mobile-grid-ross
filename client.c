#include "mobile_grid.h"

void client_init(client_state *s, tw_lp *lp)
{
   // TODO: Use stddev as well
   s->flops = client_settings.mean_flops;
   printf("Initializing client, gid: %u\n", lp->gid);
}

void client_event_handler(client_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    /*
    if (m->type == CHANNEL_TO_CLIENT)
    {
        tw_output(lp, "Client with gid %u received message from a synchronizer\n", lp->gid);
        tw_stime data_download_delay = m->task_data_size;
        tw_stime computation_delay = m->task_flops / s->flops;
        tw_stime delay = data_download_delay + computation_delay;


        tw_event *e = tw_event_new(lp->gid-1, delay, lp);
        message *msg = tw_event_data(e);

        msg->type = CLIENT_TO_CHANNEL;
        msg->task_data_size = m->task_data_size;
        msg->task_flops = m->task_flops;

        tw_event_send(e);
    } else
    {
        // error
    }
    */
    
}

void client_event_handler_rc(client_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    // No internal state to be updated
    (void) s;
    (void) bf;
    (void) m;
    (void) lp;
}

void client_finish(client_state *s, tw_lp *lp)
{
    (void) s;
    (void) lp;
}
