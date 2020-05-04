#include <math.h>
#include "mobile_grid.h"



void coordinator_init(coordinator_state *s, tw_lp *lp)
{
   // Initialize coordinator state

   printf("Initializing coordinator, gid: %u\n", lp->gid);
   // Initialize state variable
   s->tasks_received = 0;
   s->tasks_started = 0;

   // Initialize task stage linked list with dummy head node
   s->task_stage = malloc(sizeof(task_node));
   s->task_stage->task = NULL;
   s->task_stage->next = NULL;

   // Initialize worker array.
   /*s->workers = malloc(g_num_clients * sizeof(worker*));
   for (unsigned int i = 0; i < g_num_clients; i++)
   {
      s->workers[i] = malloc(sizeof(worker));
   }*/
   s->workers = malloc(sizeof(worker_node));
   s->workers->worker = NULL;
   s->workers->next = NULL;
   s->num_workers = 0;

   // Initialize scheduling interval
   tw_event *e = tw_event_new(COORDINATOR_ID, coordinator_settings.scheduling_interval, lp);
   message *msg = tw_event_data(e);
   msg->type = SCHEDULING_INTERVAL;
   tw_event_send(e);

   // Initialize task requests at random intervals
   double task_interval = (unsigned int) tw_rand_normal_sd(lp->rng, 9000, 2000, (unsigned int*) &lp->rng->count);
   if (task_interval < g_min_delay) task_interval = g_min_delay;
   tw_event *e2 = tw_event_new(COORDINATOR_ID, task_interval, lp);
   message *msg2 = tw_event_data(e2);
   msg2->type = TASK_ARRIVED;
   tw_event_send(e2);

}


void coordinator_pre_init(coordinator_state *s, tw_lp *lp)
{
   printf("Coordinator preeeinit\n");


}

void coordinator_event_handler(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
   if (m->type == DEVICE_AVAILABLE)
   {
      tw_output(lp, "Device available received %d, %u, %u\n", m->client_id, get_client_flops(m->client_id), get_client_duration(m->client_id));
      /*for (int i = 0; i < s->num_workers; i++)
      {
         if (m->client_id == s->workers[i]->client_id)
         {
            s->workers[i]->assigned = 0;
         }
      }*/
      worker_node* cur = s->workers->next;
      while (cur != NULL) {
         if (cur->worker->client_id == m->client_id) {
            tw_output(lp, "Device available received \n");
            cur->worker->assigned = 0;
            break;
         }
         cur = cur->next;
      }
   }
   if (m->type == DEVICE_REGISTER)
   {
      tw_output(lp, "Device available received %d, %u, %u, %lf\n", m->client_id, get_client_flops(m->client_id), get_client_duration(m->client_id), get_client_churn_prob(m->client_id));

      // Add to list of active workers
      worker* w = malloc(sizeof(worker));
      w->client_id = m->client_id;
      w->flops = get_client_flops(m->client_id);
      //w->dropout = get_client_dropout(m->client_id);
      w->duration = get_client_duration(m->client_id);
      w->assigned = 0;

      //s->workers[s->num_workers] = w; 
      add_worker(s->workers, w);
      s->num_workers++;
   }

   if (m->type == SCHEDULING_INTERVAL)
   {
      tw_event *e = tw_event_new(COORDINATOR_ID, coordinator_settings.scheduling_interval, lp);
      message *msg = tw_event_data(e);
      msg->type = SCHEDULING_INTERVAL;

      tw_event_send(e);

      schedule(s, lp);
   }

   // TODO might make more sense to put this in a uniform distribution at the init stage :)
   if (m->type == TASK_ARRIVED)
   {
      tw_output(lp, "Task Arriving\n");

      if (s->tasks_received + 1< coordinator_settings.num_tasks) {
         tw_output(lp, "Scheduling new task");

         double task_interval = (unsigned int) tw_rand_normal_sd(lp->rng, 9000, 2000, (unsigned int*) &lp->rng->count);
         if (task_interval < g_min_delay) task_interval = g_min_delay;
         tw_event *e = tw_event_new(COORDINATOR_ID, task_interval, lp);
         message *msg = tw_event_data(e);
         msg->type = TASK_ARRIVED;
         tw_event_send(e);
      }

      // Generate new task
      // TODO remove hard-coded sub-task number
      client_task* tasks = generate_map_reduce_task(s->tasks_received, coordinator_settings.task_size, lp);
      s->tasks_received++;

      // Stage tasks
      for (int i = 0; i < coordinator_settings.task_size; i++) {
         stage_task(s->task_stage, &tasks[i]);
      }
      printf("Staged subtask: %d\n", s->task_stage->next->task->subtask_id);

      //task_node* cur = s->task_stage;
      /*while (cur->next != NULL) {
         cur = cur->next;
         printf("%d->", cur->task->flops);
      }
      printf("\n");*/

      //client_task* task = pop_task(s->task_stage);
      //printf("Pooped task: %u, %d\n", task->flops, s->task_stage->next->task->flops);

      //printf("Task: %d\n", tasks[0].flops);
   }
}

void coordinator_event_handler_rc(coordinator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
   printf("scheduler rollback!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
}

void coordinator_finish(coordinator_state *s, tw_lp *lp)
{
   free_task_stage(s->task_stage);
   free(s->task_stage);

/*   for (int i = 0; i < g_num_clients; i++) {
      free(s->workers[i]);
   }*/
   free_workers(s->workers);
   free(s->workers);
}

// Outer loop of scheduling algorithm that loops through each task
void schedule(coordinator_state *s, tw_lp *lp) {
   tw_output(lp, "scheduling\n");

   // Notify aggregators for all tasks that need to be started 
   for (int i = s->tasks_started; i < s->tasks_received; i++) {
      printf("Notifying aggregators\n");

      // Send to all aggregators
      for (int j = 0; j < num_actors.num_aggregators; j++) {

         tw_event *e = tw_event_new(AGGREGATOR_BASE_INDEX + j, g_data_center_delay, lp);
         message *msg = tw_event_data(e);
         msg->type = NOTIFY_NEW_JOB;
         msg->task.task_id = i;
         msg->num_clients_for_task = (coordinator_settings.task_size / num_actors.num_aggregators) + ((j < coordinator_settings.task_size % num_actors.num_aggregators) ? 1 : 0);
//         msg->task.aggregator_id = msg->task.subtask_id % num_actors.num_aggregators;
         printf("Assigned to aggregator num tasks%d\n", msg->num_clients_for_task);
         tw_event_send(e);
      }

      // Send to master aggregator
      tw_event *e = tw_event_new(MASTER_AGGREGATOR_ID, g_data_center_delay, lp);
      message *msg = tw_event_data(e);
      msg->type = NOTIFY_NEW_JOB;
      msg->task.task_id = i;
      msg->num_clients_for_task = num_actors.num_aggregators; 
      tw_event_send(e);
   }
   
   worker** workers_array = NULL;
   if (coordinator_settings.scheduling_algorithm == 1) {
      workers_array = convert_workers_to_array(s->workers, s->num_workers);
      for (int i = 0; i < s->num_workers; i++) {
         printf("%f->", workers_array[i]->duration);
      }
   }

   // All staged tasks must be scheduled
   //task_node* cur = s->task_stage; 
   while (s->task_stage->next != NULL) {
      // Assign task based off of the chosen algorithm
      client_task* task = pop_task(s->task_stage);
      worker* assignment;
      if (coordinator_settings.scheduling_algorithm == 0) {
         // Naive scheduling
         assignment = schedule_naive(task, s, lp);
      } else {
         // Risk-Controlled task assignment
         assignment = schedule_rcta(task, workers_array, s->num_workers, s, lp);
         //assignment = NULL;
      }

      // We are out of available workers.
      // TODO Support for multischeduling tasks to workers?
      if (assignment == NULL) {
         break;
      }
      printf("Assignment task: %d, worker: %d, %d\n", task->flops, assignment->client_id, task->aggregator_id);

      // Initiate task execution by notifying selectors aggregators
      tw_event *e = tw_event_new(client_to_selector(assignment->client_id), g_data_center_delay, lp);
      message *msg = tw_event_data(e);
      msg->type = ASSIGN_JOB;
      msg->client_id = assignment->client_id;
      msg->task.task_id = task->task_id;
      msg->task.data_size = task->data_size;
      msg->task.flops = task->flops;
      msg->task.aggregator_id = task->aggregator_id;

      tw_event_send(e);
   }
}

// Just assign first potential worker
worker* schedule_naive(client_task* task, coordinator_state *s, tw_lp *lp) {
   worker_node* cur = s->workers->next;
   while (cur != NULL) {
      if (cur->worker->assigned == 0) {
         cur->worker->assigned = 1;
         return cur->worker;
      }
      cur = cur->next;
   }
   return NULL;
}


// Risk-controlled task assignment
worker* schedule_rcta(client_task* task, worker** workers, int num_workers, coordinator_state *s, tw_lp *lp) {
   // Part 1: Assign comptime, churnprobability, to all workers
   float lowestRisk = -1;
   coordinator_worker* selected_h = NULL;
   coordinator_worker** cws = malloc(sizeof(coordinator_worker*) * num_workers);
   int unassigned = 0;
   for (int i = 0; i < num_workers; i++) {
      if (workers[i]->assigned == 0) {
         coordinator_worker* cw = malloc(sizeof(coordinator_worker));
         cw->worker = workers[i];
         // Compute comp time
         cw->comp_time = (double) task->flops / workers[i]->flops;
         // Compute churn probability (set to 0 for now)
         cw->churn_prob = get_client_churn_prob(workers[i]->client_id);
         cws[unassigned] = cw;
         unassigned++;
         if (lowestRisk > cw->churn_prob || lowestRisk == -1) {
            selected_h = cw;
            lowestRisk = cw->churn_prob; 
         }
      }
   }
   if (unassigned == 0) {
      return NULL;
   }
   // Part 2: Sort workers by risk
   coordinator_worker** sorted = merge_sort_workers(cws, unassigned);

   // Part 3: Risk based assignment
   for (int i = 0; i < unassigned; i++) {
      if (sorted[i]->comp_time > selected_h->comp_time) {
         continue;
      }
      double added_risk = (sorted[i]->churn_prob - selected_h->churn_prob)/sorted[i]->churn_prob;
      double gain = (selected_h->comp_time - sorted[i]->comp_time) / selected_h->comp_time;
      
      if (gain > added_risk) {
         selected_h = sorted[i];
      }
   }
   if (selected_h == NULL) {
      return NULL;
   }
   selected_h->worker->assigned = 1;
   return selected_h->worker;
}

// Map reduce task has a tree like structure
client_task* generate_map_reduce_task(int task_id, int n, tw_lp *lp) {

   long start_count = lp->rng->count;

   
   printf("Generating map reduce task\n");

   client_task* tasks = malloc(n * sizeof(client_task));

   //client_task tasks[n];
   for (int i = 0; i < n; i++) {
      client_task task;
      task.task_id = task_id;
      task.subtask_id = i;
      task.aggregator_id = AGGREGATOR_BASE_INDEX + (i % num_actors.num_aggregators);
      // TODO determine why rng count doesn't increment
      // Mu: 2 M, Sigma: 0.5 M
      task.data_size = (unsigned int) tw_rand_normal_sd(lp->rng, 2000000, 500000, (unsigned int*) &lp->rng->count);
      // Mu: 30 M, Sigma: 10 M TODO make these settings?
      task.flops = (unsigned int) tw_rand_normal_sd(lp->rng, 30000000, 10000000, (unsigned int*) &lp->rng->count);

      tasks[i] = task;
   }

   return tasks;
}

// To properly stage a task, it must be inserted into the list in sorted order of highest compute requirements
void stage_task(task_node* head, client_task* task)
{
   task_node* cur = head;
   while (cur != NULL) {
      if (cur->next == NULL || cur->next->task->flops < task->flops) {
         // Insert node
         task_node* new_node = malloc(sizeof(task_node));
         new_node->task = task;
         new_node->next = cur->next;
         cur->next = new_node;
         break;
      }
      cur = cur->next;
   }
}

// Return the enclosed client_task for the first node, then free it
client_task* pop_task(task_node* head)
{
   task_node* res_node = head->next;
   if (res_node != NULL) {
      head->next = res_node->next;
   }
   client_task* task = res_node->task;
   free(res_node);
   return task;
}

// Free linked list
void free_task_stage(task_node* head) 
{
   task_node* cur = head->next;
   task_node* prev = NULL;
   while (cur != NULL) {
      prev = cur;
      cur = cur->next;
      free(prev);
   }
}

void coordinator_event_trace(message *m, tw_lp *lp, char *buffer, int *collect_flag)
{
   // Always log type
   memcpy(buffer, &m->type, sizeof(message_type));
   buffer += sizeof(message_type);
   if (m->type == DEVICE_AVAILABLE)
   {
      memcpy(buffer, &m->client_id, 8);
   }
   else if (m->type == DEVICE_REGISTER)
   {
      memcpy(buffer, &m->client_id, 8);
      buffer += 8;
   }
}

// Add worker to head of list
void add_worker(worker_node* head, worker* worker) {
   worker_node* wn = malloc(sizeof(worker_node));
   wn->worker = worker;
   wn->next = head->next;
   head->next = wn;
}

// Delete worker with matching client_id from linked list
void delete_worker(worker_node* head, tw_lpid client_id) {
   worker_node* cur = head->next;
   worker_node* prev = head;
   while (cur != NULL) {
      if (cur->worker->client_id == client_id) {
         prev->next = cur->next;
         free(cur);
         return;
      }
      prev = cur;
      cur = cur->next;
   }
}

worker* pop_worker(worker_node* head) {
   worker_node* res_node = head->next;
   if (res_node != NULL) {
      head->next = res_node->next;
   }
   worker* worker = res_node->worker;
   free(res_node);
   return worker;
}

// Free all workers
void free_workers(worker_node* head) {
   worker_node* cur = head->next;
   worker_node* prev = NULL;
   while (cur != NULL) {
      prev = cur;
      cur = cur->next;
      free(prev);
   }
}

// Convert worker linked list to array so it can be easilly sorted
worker** convert_workers_to_array(worker_node* head, int count) {
   // Allocate array
   worker** res = malloc(sizeof(worker*) * count);

   // Populate array
   worker_node* cur = head->next;
   int i = 0;
   while (cur != NULL) {
      res[i] = cur->worker;
      cur = cur->next;
      i++;
   }
   return res;
} 

coordinator_worker** merge_sort_workers(coordinator_worker** workers, int n) {
   if (n == 1) {
      coordinator_worker** res = malloc(sizeof(coordinator_worker*));
      res[0] = workers[0];
      return res;
   }

   // Sort both halves
   int len1 = n/2;
   int len2 = (n - (n/2));
   coordinator_worker** h1 = merge_sort_workers(workers, len1);
   coordinator_worker** h2 = merge_sort_workers(&workers[n/2], len2);

   // Merge
   coordinator_worker** res = malloc(sizeof(coordinator_worker*) * n);

   int i = 0;
   int j = 0;
   int total = 0;
   while (total < n) {
      if (i < len1 && j < len2) {
         if (h1[i]->churn_prob < h2[j]->churn_prob) {
            res[total] = h1[i];
            i++;
            total++;
         } else {
            res[total] = h2[j];
            j++;
            total++;
         }
      } else if (i < len1 && j >= len2) {
         for (i = i; i < len1; i++) {
            res[total] = h1[i];
            total++;
         }
      } else if (i >= len1 && j < len2) {
         for (j = j; j < len2; j++) {
            res[total] = h2[j];
            total++;
         }
      } else {
         total++;
      }
   }
   free(h1);
   free(h2);

   return res;
}
