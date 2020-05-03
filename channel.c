#include "mobile_grid.h"

void channel_init(channel_state *s, tw_lp *lp)
{
   printf("Initializing channel, gid: %u (Client %u)\n", lp->gid, channel_to_client(lp->gid));
}

void channel_event_handler(channel_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    if (m->type == ASSIGN_JOB)
    {
        // Delay to simulate job being transmitted from selector to client
        // TODO: Replace with TLM propogation simulation
        tw_output(lp, "Channel %u: Got job for lp %u\n", lp->gid, m->client_id);
        double temp_channel_delay = 5;
        tw_event *e = tw_event_new(m->client_id, temp_channel_delay, lp);
        message *msg = tw_event_data(e);

        msg->type = ASSIGN_JOB;
        msg->task = m->task;
        msg->client_id = m->client_id;

        tw_event_send(e);
    } else if (m->type == REQUEST_DATA)
    {
        // Delay to simulate the download of data to the client
        tw_output(lp, "Channel %u: Got data request for lp %u\n", lp->gid, m->client_id);
        double temp_channel_delay = m->task.data_size/1000;

        tw_event *e = tw_event_new(m->client_id, temp_channel_delay, lp);
        message *msg = tw_event_data(e);

        msg->type = RETURN_DATA;
        msg->task = m->task;
        msg->client_id = m->client_id;

        tw_event_send(e);
    } else if (m->type == SEND_RESULTS)
    {
        // Delay to simulate the upload of results from the client
        tw_output(lp, "Channel %u: Sending results for lp %u\n", lp->gid, m->client_id);
        double temp_channel_delay = m->task.results_size/1000;

        tw_event *e = tw_event_new(m->task.aggregator_id, temp_channel_delay, lp);
        message *msg = tw_event_data(e);

        msg->type = SEND_RESULTS;
        msg->task = m->task;
        msg->client_id = m->client_id;

        tw_event_send(e);

        // When upload is done, also notify the selector
        e = tw_event_new(client_to_selector(m->client_id), temp_channel_delay, lp);
        msg = tw_event_data(e);

        msg->type = SEND_RESULTS;
        msg->task = m->task;
        msg->client_id = m->client_id;

        tw_event_send(e);
    }

}

void channel_event_handler_rc(channel_state *s, tw_bf *bf, message *m, tw_lp *lp)
{
    // No internal state to be updated
    (void) s;
    (void) bf;
    (void) m;
    (void) lp;
}

void channel_finish(channel_state *s, tw_lp *lp)
{
    (void)s;
    (void)lp;
}
