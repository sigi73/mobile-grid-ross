#include <math.h>
#include "mobile_grid.h"

unsigned int g_num_clients;
unsigned int *g_client_flops;
float *g_client_dropout;

/* generate a random value weighted within the normal (gaussian) distribution */
/* https://github.com/rflynn/c/blob/master/rand-normal-distribution.c */
static double gauss(void)
{
  double x = (double)random() / RAND_MAX,
         y = (double)random() / RAND_MAX,
         z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  return z;
}

void setup_client_capabilities()
{
    srandom(time(NULL));

    g_client_flops = malloc(g_num_clients * sizeof(unsigned int));
    for (unsigned int i = 0; i < g_num_clients; i++)
    {
        double flops = gauss();
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
}

void client_event_handler(client_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    
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
