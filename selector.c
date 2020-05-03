#include "mobile_grid.h"

void selector_init(selector_state *s, tw_lp *lp)
{
    printf("Initializing selector, gid: %u", lp->gid);

    s->num_clients = num_actors.num_clients_per_selector;
    s->client_gids = malloc(sizeof(tw_lpid)*s->num_clients);

    unsigned int selector_index = lp->gid - NUM_FIXED_ACTORS - num_actors.num_aggregators;
    unsigned int client_gid_offset = NUM_FIXED_ACTORS + num_actors.num_aggregators + num_actors.num_selectors + selector_index * num_actors.num_clients_per_selector;
    printf(", Clients: ");
    for (int i = 0; i < s->num_clients; i++)
    {
        s->client_gids[i] = client_gid_offset + i;
        printf("%u,", s->client_gids[i]);
    }
    printf("\n");
}

void selector_pre_init(selector_state *s, tw_lp *lp)
{
    // At the start we tell the coordinator how many devices we have
    for (unsigned int i = 0; i < s->num_clients; i++)
    {
        tw_event *e = tw_event_new(COORDINATOR_ID, get_client_start_time(s->client_gids[i]), lp);
        message *msg = tw_event_data(e);

        msg->type = DEVICE_REGISTER;
        msg->client_id = s->client_gids[i];

        tw_event_send(e);
    }
}

void selector_event_handler(selector_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    if (m->type == ASSIGN_JOB)
    {
        // Coordinator has given a job to a client we communicate with
        tw_output(lp, "Selector %u: Got job for lp %u\n", lp->gid, m->client_id);
        // Send the job to the channel for channel simulation
        tw_event *e = tw_event_new(client_to_channel(m->client_id), g_data_center_delay, lp);
        message *msg = tw_event_data(e);

        msg->type = ASSIGN_JOB;
        msg->task = m->task;
        msg->client_id = m->client_id;

        tw_event_send(e);
    } else if (m->type == SEND_RESULTS)
    {
        tw_event *e = tw_event_new(COORDINATOR_ID, g_data_center_delay, lp);
        message *msg = tw_event_data(e);

        msg->type = DEVICE_AVAILABLE;
        msg->client_id = m->client_id;

        tw_event_send(e);
    }
}

void selector_event_handler_rc(selector_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    // No internal state to be updated
    (void) s;
    (void) bf;
    (void) m;
    (void) lp;
}

void selector_finish(selector_state *s, tw_lp *lp)
{
    (void) lp;
    free (s->client_gids);
}