#include "mobile_grid.h"



void coordinator_init(coordinator_state *s, tw_lp *lp)
{
   // Initialize coordinator state

   printf("Initializing coordinator, gid: %u\n", lp->gid);
}

void coordinator_pre_init(coordinator_state *s, tw_lp *lp)
{
   printf("Coordinator preeeinit\n");

   // Initialize scheduling interval
   tw_event *e = tw_event_new(g_coordinator_id, coordinator_settings.scheduling_interval, lp);
   message *msg = tw_event_data(e);
   msg->type = SCHEDULING_INTERVAL;
   tw_event_send(e);

   // Initialize task requests at random intervals
   double task_interval = (unsigned int) tw_rand_normal_sd(lp->rng, 9000, 2000, (unsigned int*) &lp->rng->count);
   tw_event *e2 = tw_event_new(g_coordinator_id, task_interval, lp);
   message *msg2 = tw_event_data(e2);
   msg2->type = TASK_ARRIVED;
   tw_event_send(e2);
}

void coordinator_event_handler(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
   if (m->type == DEVICE_AVAILABLE)
   {
      // Send the job to the channel for channel simulation
      /*tw_event *e = tw_event_new(client_to_channel(m->client_id), g_data_center_delay, lp);
      message *msg = tw_event_data(e);

      msg->type = ASSIGN_JOB;
      msg->task = m->task;
      msg->client_id = m->client_id;

      tw_event_send(e);*/

      tw_output(lp, "Device available received\n");
   }

   if (m->type == SCHEDULING_INTERVAL)
   {
      tw_output(lp, "SCHEDULING INTERVAL\n");

      tw_event *e = tw_event_new(g_coordinator_id, coordinator_settings.scheduling_interval, lp);
      message *msg = tw_event_data(e);
      msg->type = SCHEDULING_INTERVAL;

      tw_event_send(e);


      schedule(lp);
   }

   if (m->type == TASK_ARRIVED)
   {
      tw_output(lp, "Task Arriving\n");

      double task_interval = (unsigned int) tw_rand_normal_sd(lp->rng, 9000, 2000, (unsigned int*) &lp->rng->count);
      tw_event *e = tw_event_new(g_coordinator_id, task_interval, lp);
      message *msg = tw_event_data(e);
      msg->type = TASK_ARRIVED;
      tw_event_send(e);

      // Generate new task
      generate_map_reduce_task(50, lp);
   }
}

void coordinator_event_handler_rc(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
}

void coordinator_finish(coordinator_state *s, tw_lp *lp)
{
}


void schedule(tw_lp *lp) {
   tw_output(lp, "scheduling\n");
}

// Map reduce task has a tree like structure
void generate_map_reduce_task(int n, tw_lp *lp) {

   long start_count = lp->rng->count;

   
   printf("Generating map reduce task\n");

   client_task tasks[n];
   for (int i = 0; i < n; i++) {
      client_task task;
      task.task_id = i;
      // TODO determine why rng count doesn't increment
      // Mu: 2 M, Sigma: 0.5 M
      task.data_size = (unsigned int) tw_rand_normal_sd(lp->rng, 2000000, 500000, (unsigned int*) &lp->rng->count);
      // Mu: 30 M, Sigma: 10 M TODO make these settings?
      task.flops = (unsigned int) tw_rand_normal_sd(lp->rng, 30000000, 10000000, (unsigned int*) &lp->rng->count);

      tasks[i] = task;
   }
}