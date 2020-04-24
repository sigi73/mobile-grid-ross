#include "mobile_grid.h"

void synchronizer_init(synchronizer_state *s, tw_lp *lp)
{
   // Initialize synchronizer state
   s->tasks_remaining = g_num_clients;
   s->tasks_completed = 0;

   printf("Initializing synchronizer, gid: %u\n", lp->gid);

   // In init you can only send messages to yourself, so we wait to send tasks until pre_init
}

void synchronizer_pre_init(synchronizer_state *s, tw_lp *lp)
{
    for (int i = 0; i < g_num_clients; i++)
    {
        if (s->tasks_remaining == 0) break;
        tw_stime delay = .1; // This should depend on size of the work and the bandwidth of the synchronizer

        tw_event *e = tw_event_new(2*i+1, delay, lp);
        message *msg = tw_event_data(e);

        msg->type = SYNCH_TO_CHANNEL;
        // TODO: Use stddev as well
        msg->task_data_size = synchronizer_settings.mean_data_size;
        msg->task_flops = synchronizer_settings.mean_flop_per_task;

        tw_event_send(e);
        printf("Synchronizer sent message to channel %d, gid %d\n", i, 2*i+1);

        s->tasks_remaining--;
    }
}

void synchronizer_event_handler(synchronizer_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    if (m->type == CHANNEL_TO_SYNCH)
    {
        s->tasks_completed++;
    }
}

void synchronizer_event_handler_rc(synchronizer_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    if (m->type == CHANNEL_TO_SYNCH)
    {
        s->tasks_completed--;
    }
}

void synchronizer_finish(synchronizer_state *s, tw_lp *lp)
{
    printf("Tasks completed: %u\n", s->tasks_completed);
    (void)lp;
}