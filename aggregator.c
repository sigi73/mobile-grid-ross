#include "mobile_grid.h"


/*
 * Aggregator tasks linked list helper functions
 */
aggregator_task *add_aggregator_task(aggregator_task *head)
{
   aggregator_task *tail = head;
   while (tail->next != NULL)
   {
      tail = tail->next;
   }
   tail->next = malloc(sizeof(aggregator_task));
   return tail->next;
}

void remove_aggregator_task(aggregator_task *head, aggregator_task *rem)
{
   aggregator_task *curr = head;
   aggregator_task *prev = NULL;

   while (curr->next != rem)
   {
      if (curr->next == NULL){
         printf("Aggregator task not found for removal");
         return;
      }

      prev = curr;
      curr = curr->next;
   }
   if (prev == NULL){
      //printf("Trying to remove dummy aggregator tasks");
      //exit(1);
      
      //Note: Commented since we do want to remove the dummy task when cleaning up at the end
   } else
   {
      prev->next = curr->next;
   }
}

aggregator_task *find_aggregator_task(aggregator_task *head, int task_id)
{
   aggregator_task *curr = head;
   while (curr->task_id != task_id)
   {
      if (curr->next == NULL)
      {
         return NULL;
      }
      curr = curr->next;
   }
   return curr;
}

void free_aggregator_task(aggregator_task *head)
{
   aggregator_task *prev;
   while (head->next != NULL)
   {
      prev = head;
      head = head->next;
      free(prev);
   }
   free(head);
}
/*
 * End Aggregator tasks linked list helper functions
 */

void aggregator_init(aggregator_state *s, tw_lp *lp)
{
   printf("Initializing aggregator, gid: %u\n", lp->gid);
   s->tasks = malloc(sizeof(aggregator_task)); // Create a dummy head to the linked list
   s->tasks->task_id = -1;   // This will break if aggregator is actually assigned UINT_MAX tasks
   s->tasks->next = NULL;
}

void aggregator_pre_init(aggregator_state *s, tw_lp *lp)
{

}


void aggregator_event_handler(aggregator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
   if (m->type == NOTIFY_NEW_JOB)
   {
      tw_output(lp, "Aggregator notified of new job\n");
      aggregator_task *job = add_aggregator_task(s->tasks);
      job->task_id = m->task.task_id;
      job->num_remaining = m->num_clients_for_task;
      job->next = NULL;

   } else if (m->type == SEND_RESULTS)
   {
      tw_output(lp, "Received completed task from client %u\n", m->client_id);

      if(s->tasks == NULL)
      {
         printf("ERROR: GID %u received results but not waiting for jobs, client_id %u, task_id %u\n", lp->gid, m->client_id, m->task.task_id);
         // Exit horribly? 
         exit(1);
      }
      aggregator_task *job = find_aggregator_task(s->tasks, m->task.task_id);
      if (job == NULL)
      {
         printf("ERROR: GID %u received results but is not waiting for task_id %u", lp->gid, m->task.task_id);
      }
      job->num_remaining--;

      // If all clients have finished this task, remove it
      if (job->num_remaining == 0)
      {
         unsigned int task_id = job->task_id;
         // Remove from linked list
         remove_aggregator_task(s->tasks, job);

         if (lp->gid != MASTER_AGGREGATOR_ID) {
            // Job complete, send to master aggregator!
            tw_event *e = tw_event_new(client_to_selector(m->client_id), g_data_center_delay, lp);
            message *msg = tw_event_data(e);

            msg->type = SEND_RESULTS;
            msg->task.task_id = m->task.task_id;
            printf("Aggregator: %u complete with task %u\n", lp->gid, task_id);
         } else
         {
            tw_output(lp, "Master aggregator: Task complete: %u\n", task_id);
         }
         

      }
   }
}

void aggregator_event_handler_rc(aggregator_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
   if (m->type == NOTIFY_NEW_JOB)
   {
      // Remove the new job added
      remove_aggregator_task(s->tasks, find_aggregator_task(s->tasks, m->task.task_id));

   } else if (m->type == SEND_RESULTS)
   {
      // If we still have the task increment the num completed
      aggregator_task *job = find_aggregator_task(s->tasks, m->task.task_id);
      if (job != NULL)
      {
         job->num_remaining++;
      } else
      {
         // If we don't have it, add it back
         job = add_aggregator_task(s->tasks);
         job->task_id = m->task.task_id;
         job->num_remaining = 1;
      }
      
   }
}

void aggregator_finish(aggregator_state *s, tw_lp *lp)
{
   free_aggregator_task(s->tasks);
   s->tasks = NULL;
}