# Production Incident Response Runbook

Arbor Systems Ltd. (fictional company used for research purposes). Owner: Site Reliability Engineering. Version 1.7.

## Severity levels

| Severity | Definition | Acknowledgement target |
| --- | --- | --- |
| SEV1 | Full outage of ArborDesk or a confirmed data breach | 15 minutes |
| SEV2 | Major degradation affecting many customers | 30 minutes |
| SEV3 | Minor issue with a workaround available | 4 business hours |

## On-call rotation

The on-call rotation changes weekly. Handover takes place every Monday at 10:00. Each rotation has a primary and a secondary on-call engineer. Engineers receive 150 EUR for each week of on-call duty. Engineers who handle an incident between 22:00 and 06:00 receive time off in lieu for the hours worked.

## Escalation

If the primary on-call engineer does not acknowledge a page within 10 minutes, the secondary on-call engineer is paged automatically. If there is still no acknowledgement after 20 minutes, the engineering manager is paged.

## Running a SEV1 incident

1. The first responder declares the incident and assigns an incident commander.
2. The incident commander opens a dedicated Slack channel named #inc-<date>-<short-name>.
3. The public status page is updated at least every 30 minutes until the incident is resolved.
4. Customer Success sends the first customer communication within 1 hour of the incident being declared.
5. After resolution, the incident commander confirms recovery and closes the incident.

## Postmortems

A blameless postmortem is written for every SEV1 and SEV2 incident within 5 business days of resolution. The postmortem includes a timeline, root cause analysis, and action items with owners and due dates. Postmortems are reviewed in the weekly reliability meeting.
