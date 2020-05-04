#include <math.h>
#include "mobile_grid.h"

unsigned int g_num_clients;
unsigned int *g_client_flops;
float *g_client_duration;

/* generate a random value weighted within the normal (gaussian) distribution */
/* https://github.com/rflynn/c/blob/master/rand-normal-distribution.c */
// TODO: Remember to add acknolwedgement in report
static double gauss(int mean, int stddev)
{
  double x = (double)random() / RAND_MAX,
         y = (double)random() / RAND_MAX,
         z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
         z *= (double)stddev;
         z+= (double)mean;
  return z;
}

/* Generate a random value with a exponential distribution */
/* https://stackoverflow.com/questions/34558230/generating-random-numbers-of-exponential-distribution */
double rand_expo(double lambda)
{
    double u;
    u = rand() / (RAND_MAX + 1.0);
    return -log(1- u) / lambda;
}

double rand_uniform(double low, double high) 
{
    double u = (double)rand() / RAND_MAX; //0 to 1
    u *= (high - low);
    u += low;
    return u;
}

void allocate_client_parameters()
{
    srand(SEED);

    client_parameters.client_flops = malloc(g_num_clients * sizeof(unsigned int));
    client_parameters.client_start_time = malloc(g_num_clients * sizeof(float));
    client_parameters.client_duration = malloc(g_num_clients * sizeof(float));
    client_parameters.client_churn_prob = malloc(g_num_clients * sizeof(float));
    client_parameters.client_x = malloc(g_num_clients * sizeof(float));
    client_parameters.client_y = malloc(g_num_clients * sizeof(float));
    for (unsigned int i = 0; i < g_num_clients; i++)
    {
        double flops = gauss(client_settings.mean_flops, client_settings.stddev_flops); // Doesn't need to be reversible, before any events
        flops = fmax(1, flops); // 0 probably shouldn't be permitted?
        client_parameters.client_flops[i] = (unsigned int)flops;

        client_parameters.client_start_time[i] = rand_uniform(g_min_delay, g_tw_ts_end - client_settings.mean_duration);
        client_parameters.client_duration[i] = rand_expo(client_settings.mean_duration);
        client_parameters.client_churn_prob[i] = gauss(10, 2) / 100;                       // TODO set as actual parameters 
        printf("yooo %f\n", client_parameters.client_churn_prob[i]);
        client_parameters.client_x[i] = rand_uniform(-MAX_GRID_SIZE, MAX_GRID_SIZE);
        client_parameters.client_y[i] = rand_uniform(-MAX_GRID_SIZE, MAX_GRID_SIZE);
    }
}

void free_client_parameters()
{
    free(client_parameters.client_flops);
    free(client_parameters.client_start_time);
    free(client_parameters.client_duration);
    free(client_parameters.client_churn_prob);
    free(client_parameters.client_x);
    free(client_parameters.client_y);
}


void client_init(client_state *s, tw_lp *lp)
{
   printf("Initializing client, gid: %u (Channel %u)\n", lp->gid, client_to_channel(lp->gid));
   s->flops = get_client_flops(lp->gid);
   s->duration = get_client_duration(lp->gid);
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