#include <math.h>
#include "mobile_grid.h"

unsigned int g_num_clients;
unsigned int *g_client_flops;
float *g_client_dropout;

/* generate a random value weighted within the normal (gaussian) distribution */
/* https://github.com/rflynn/c/blob/master/rand-normal-distribution.c */
// TODO: Remember to add acknolwedgement in report
static double gauss(void)
{
  double x = (double)random() / RAND_MAX,
         y = (double)random() / RAND_MAX,
         z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  return z;
}

void setup_client_capabilities()
{
    srand(SEED);

    g_client_flops = malloc(g_num_clients * sizeof(unsigned int));
    g_client_dropout = malloc(g_num_clients * sizeof(float));
    for (unsigned int i = 0; i < g_num_clients; i++)
    {
        double flops = gauss(); // Doesn't need to be reversible, before any events
        flops *= client_settings.stddev_flops;
        flops += client_settings.mean_flops;
        flops = fmax(0, flops); // 0 probably shouldn't be permitted?
        g_client_flops[i] = (unsigned int)flops;

        g_client_dropout[i] = 0.0f;
    }
}


void client_init(client_state *s, tw_lp *lp)
{
   printf("Initializing client, gid: %u (Channel %u)\n", lp->gid, client_to_channel(lp->gid));
   s->flops = get_client_flops(lp->gid);
}

void client_event_handler(client_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    if (m->type == ASSIGN_JOB)
    {
        // Received a job, now download data from data server
        tw_output(lp, "Client:Got job for lp %u\n", m->client_id);
        // Min delay
        tw_event *e = tw_event_new(client_to_channel(lp->gid), g_min_delay, lp);
        message *msg = tw_event_data(e);

        msg->type = REQUEST_DATA;
        msg->task = m->task;
        msg->client_id = m->client_id;

        tw_event_send(e);
    } 
    if (m->type == RETURN_DATA)
    {
        // Received data, run calculation and send results to aggregator
        tw_output(lp, "Client: Got data for lp %u\n", m->client_id);
        //double delay = m->task.flops / s->flops * 1000;
        double delay = g_min_delay;
        tw_event *e = tw_event_new(client_to_channel(lp->gid), delay, lp);
        message *msg = tw_event_data(e);

        msg->type = SEND_RESULTS;
        msg->task = m->task;
        msg->client_id = m->client_id;

        tw_event_send(e);
    }
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


/*
void client_event_trace(message *m, tw_lp *lp, char *buffer, int *collect_flag)
{
    if (m->type == RETURN_DATA)
    {
        memcpy(buffer, &(m->task.flops), sizeof(unsigned int));
    } else
    {
        collect_flag = 0;
    }
}
*/