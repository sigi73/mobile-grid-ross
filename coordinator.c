#include <math.h>
#include "mobile_grid.h"

void coordinator_init(coordinator_state *s, tw_lp *lp)
{
   // Initialize coordinator state

   printf("Initializing coordinator, gid: %u\n", lp->gid);
}

void coordinator_pre_init(coordinator_state *s, tw_lp *lp)
{
}

int random_in_range(int low, int high)
{
   return (rand() % (high - low + 1)) + low;
}

void coordinator_event_handler(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
   if (m->type == DEVICE_AVAILABLE)
   {
      // New device is available
      // Dummy coordinator code to test simulation
      // Assigns a job of taskid 0 with random sizes to the new device
      tw_event *e = tw_event_new(client_to_selector(m->client_id), g_data_center_delay, lp);
      message *msg = tw_event_data(e);

      msg->type = ASSIGN_JOB;
      msg->client_id = m->client_id;
      client_task *task = &(msg->task);
      task->task_id = 0;
      task->data_size = random_in_range(100, 200);
      task->flops = random_in_range(100, 200);
      task->results_size = random_in_range(5, 15);
      task->aggregator_id = 2;

      tw_event_send(e);
   }
}

void coordinator_event_handler_rc(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
}

void coordinator_finish(coordinator_state *s, tw_lp *lp)
{
}
