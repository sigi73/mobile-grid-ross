#include <math.h>
#include "mobile_grid.h"

void coordinator_init(coordinator_state *s, tw_lp *lp)
{
   // Initialize coordinator state

   printf("Initializing coordinator, gid: %u\n", lp->gid);

   s->num_completed = 0;
   s->current_task_id = 0;
   s->num_assigned_currently = 0;
}

void update_aggregator(coordinator_state *s, tw_lp *lp)
{
   // Send the new job to aggregator
   tw_event *e = tw_event_new(AGGREGATOR_BASE_INDEX + s->current_task_id % num_actors.num_aggregators, g_data_center_delay, lp);
   message *msg = tw_event_data(e);
   msg->type = NOTIFY_NEW_JOB;
   msg->num_clients_for_task = NUM_CLIENTS_PER_TASK;
   msg->task.task_id = s->current_task_id;
   tw_event_send(e);

   // Send the new job to the master aggregator
   e = tw_event_new(MASTER_AGGREGATOR_ID, g_data_center_delay, lp);
   msg = tw_event_data(e);
   msg->type = NOTIFY_NEW_JOB;
   msg->num_clients_for_task = 1; //We only have 1 aggregator for all tasks currently
   msg->task.task_id = s->current_task_id;
   tw_event_send(e);

   tw_output(lp, "Sent task %u to aggregator %u\n", s->current_task_id, AGGREGATOR_BASE_INDEX + s->current_task_id % num_actors.num_aggregators);
}

void coordinator_pre_init(coordinator_state *s, tw_lp *lp)
{
   update_aggregator(s, lp);
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

      if (s->num_assigned_currently >= NUM_CLIENTS_PER_TASK)
      {
         if (s->current_task_id < NUM_TOTAL_TASKS)
         {
            s->num_assigned_currently = 0;
            s->current_task_id++;
            update_aggregator(s, lp);
         } else
         {
            return;
         }
         
      }

      msg->type = ASSIGN_JOB;
      msg->client_id = m->client_id;
      client_task *task = &(msg->task);
      task->task_id = s->current_task_id;
      task->data_size = random_in_range(100, 200);
      task->flops = random_in_range(100, 200);
      task->results_size = random_in_range(5, 15);
      task->aggregator_id = AGGREGATOR_BASE_INDEX + s->current_task_id % num_actors.num_aggregators;

      s->num_assigned_currently++;

      tw_event_send(e);
   }
}

void coordinator_event_handler_rc(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
}

void coordinator_finish(coordinator_state *s, tw_lp *lp)
{
}
