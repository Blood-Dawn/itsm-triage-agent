"""
data/generator/templates.py
────────────────────────────
Per-category ticket text templates with PRIORITY-CORRELATED context.

KEY CHANGE FROM v1:
    In v1, priority was assigned randomly and was completely independent
    of the ticket text. A P1 "laptop won't turn on" and a P4 "laptop won't
    turn on" had IDENTICAL text — the model could not learn the distinction.

    In v2, every ticket is generated from a PRIORITY-SPECIFIC context string
    that is injected directly into the ticket body. P1 tickets contain phrases
    like "production is completely down" and "immediate data loss risk". P4
    tickets contain phrases like "no rush" and "whenever IT has bandwidth".

    This gives the model learnable text signals to correlate with priority,
    which is a requirement for any real classification task.

HOW IT WORKS:
    The generator picks a category and a priority, then selects:
      1. A subject + symptom → the ticket title / problem statement
      2. A priority-specific context string → injected into the ticket body
         (this is what teaches the model what P1 vs P4 looks like)
      3. A priority-specific next_action → the support agent's recommended step

STRUCTURE OF EACH TEMPLATE:
    subjects:     Things that can have issues
    symptoms:     What the user experiences
    details:      Optional extra context (general)
    p1_contexts:  CRITICAL urgency phrases — production down, data loss, all users blocked
    p2_contexts:  HIGH urgency phrases — multiple users, important deadline, major degradation
    p3_contexts:  MEDIUM urgency phrases — single user, intermittent, workaround available
    p4_contexts:  LOW urgency phrases — no rush, cosmetic, enhancement request
    p1_actions:   Immediate escalation / war-room response actions
    p2_actions:   Urgent but structured response actions
    p3_actions:   Standard resolution actions
    p4_actions:   Scheduled / low-priority resolution actions
"""

from typing import TypedDict


class CategoryTemplate(TypedDict):
    """
    Complete template bank for one ITSM category.

    The p*_contexts and p*_actions lists are the critical addition over v1.
    By injecting priority-appropriate language into the ticket text,
    the model can learn meaningful text → priority correlations.
    """
    subjects:     list[str]
    symptoms:     list[str]
    details:      list[str]
    p1_contexts:  list[str]
    p2_contexts:  list[str]
    p3_contexts:  list[str]
    p4_contexts:  list[str]
    p1_actions:   list[str]
    p2_actions:   list[str]
    p3_actions:   list[str]
    p4_actions:   list[str]


TEMPLATES: dict[str, CategoryTemplate] = {

    # ─── HARDWARE ────────────────────────────────────────────────────────────

    "hardware": {
        "subjects": [
            "laptop", "desktop workstation", "docking station", "external monitor",
            "keyboard", "mouse", "headset", "webcam", "USB hub", "power supply",
            "SSD drive", "RAM module", "RAID array", "server", "NAS device",
            "graphics card", "network interface card", "battery pack",
            "ergonomic keyboard", "touch screen display", "barcode scanner",
            "label printer hardware", "point-of-sale terminal",
        ],
        "symptoms": [
            "won't turn on", "makes a loud clicking noise", "overheating and shutting down",
            "screen is flickering and unreadable", "not detected when plugged in",
            "battery draining in under an hour", "blue screen of death on startup",
            "running extremely slowly", "all USB ports stopped working",
            "fan running at full speed and making grinding noise",
            "display has dead pixels across the entire screen",
            "touchpad not responding to input", "SMART errors detected on disk",
            "RAID array is showing degraded status", "physically damaged after being dropped",
            "producing a burning smell and shutting off",
            "showing 'disk read error' on every boot",
            "freezing randomly multiple times per hour",
            "external display not receiving signal",
            "charging port is physically broken",
            "case cracked and internal components exposed",
            "making a high-pitched whine intermittently",
        ],
        "details": [
            "This started after the latest Windows update.",
            "I've tried restarting multiple times with no change.",
            "Other users on my floor have the same issue.",
            "The device is about 3 years old.",
            "Already swapped cables and the problem persists.",
            "It was working fine yesterday afternoon.",
            "The warranty should still be active.",
            "I've run diagnostics but they show no issues.",
            "I tried a different power outlet with no improvement.",
            "Device is company-issued and critical to my daily work.",
        ],
        "p1_contexts": [
            "URGENT: This is our primary production server and it is completely down. The entire engineering team — 40 people — cannot work. We are losing approximately $5,000 per hour in SLA penalties.",
            "CRITICAL: The RAID array has entered a degraded state and one drive is showing imminent failure. We have not completed today's backup. There is immediate risk of catastrophic data loss.",
            "EMERGENCY: The production point-of-sale terminals in our retail store have gone offline. We cannot process any customer transactions. The store must close if this is not resolved within 30 minutes.",
            "CRITICAL OUTAGE: All workstations on the trading floor are displaying BSOD simultaneously. Our trading operations are completely halted. Every minute of downtime costs the company significant revenue.",
            "URGENT: The main file server NAS has stopped responding. 200+ users across 3 departments have lost access to all shared files. Executive leadership is aware and demanding immediate resolution.",
            "CRITICAL: A server in our data center is producing a burning smell and has shut down. Possible fire hazard — we need immediate on-site response and may need to involve facilities management.",
        ],
        "p2_contexts": [
            "This is affecting 6 people on my team and we have a major client presentation in 2 hours that depends on this hardware. We are unable to complete our work.",
            "Multiple workstations in the finance department have the same issue right now. Month-end reporting is due by EOD today and we cannot afford further delays.",
            "Three developers on my team have reported the same hardware failure. Our sprint demo is in 4 hours and we need these machines working.",
            "This is impacting the entire marketing department. We have a live campaign launch scheduled for this afternoon and need this resolved urgently.",
            "Half of our customer support team is affected. Queue times are rising and customers are being impacted. We need this resolved within the next 2 hours.",
            "My primary work machine is down and I have back-to-back video calls with C-suite executives starting in 90 minutes. I urgently need a replacement device.",
        ],
        "p3_contexts": [
            "This is affecting only me and I've been able to work around it by using a colleague's spare machine. Not urgent, but it is impacting my productivity.",
            "The issue is intermittent — it happens maybe twice a day and a restart resolves it temporarily. Please schedule a repair at your convenience.",
            "I can still work, but this is slowing me down noticeably. If you could look into it within the next few days that would be great.",
            "The device is usable but the symptom is getting progressively worse over the past two weeks. I'd like it looked at before it fails completely.",
            "This is a secondary workstation I use occasionally. The primary is fine. Please fix this when your schedule allows.",
            "I've found a workaround that lets me continue working. Please address this during the next scheduled maintenance window.",
        ],
        "p4_contexts": [
            "Low priority — I just wanted to flag this cosmetic issue whenever IT has bandwidth. It doesn't affect my ability to work at all.",
            "No rush on this — it's a minor annoyance rather than a blocker. Happy to wait for the next tech refresh cycle.",
            "Just a heads-up for the asset team: this device is quite old and starting to show its age. No immediate action needed, but worth considering for the next refresh round.",
            "Informational ticket: I noticed a small crack in the device casing. It's cosmetic only and doesn't affect performance. Please log it in the asset system when convenient.",
            "Enhancement request: I'd like to upgrade my RAM from 8 GB to 16 GB whenever there's budget and availability. No deadline on this.",
            "FYI ticket: My ergonomic stand is slightly wobbly. Not impacting work. Let me know if there's a self-service replacement process.",
        ],
        "p1_actions": [
            "IMMEDIATE: Declare a P1 major incident. Engage the on-call infrastructure team, open a war-room bridge, and dispatch a field technician on-site within 15 minutes. Notify the CTO and begin incident communication every 30 minutes.",
            "CRITICAL: Immediately provision a replacement server from hot-standby inventory. Activate the DR runbook for data recovery. Engage the vendor's critical support line for hardware diagnostics. Do not power down without authorisation.",
            "EMERGENCY RESPONSE: Dispatch on-site technician immediately. Contact facilities management regarding the potential fire hazard. Initiate server room evacuation protocol if smell persists. Engage vendor critical support for hardware swap.",
            "P1 ESCALATION: Replace all affected POS terminals from emergency hardware inventory. If stock is unavailable, engage vendor for emergency advance replacement. Coordinate with store manager on contingency — cash transactions only as fallback.",
            "INCIDENT: Deploy replacement workstations from the emergency loaner pool immediately. Engage imaging team to expedite OS provisioning. Notify management of impact scope and ETA for restoration.",
            "CRITICAL DATA RISK: Immediately replace the failing RAID drive from spare parts. Do NOT reboot the array until a backup is confirmed. Engage storage team to verify backup integrity before bringing array back to full RAID-5.",
        ],
        "p2_actions": [
            "Escalate to senior hardware technician. Provision a loaner device from inventory within 1 hour. Run full diagnostics on the original device and prioritise repair to meet the client presentation deadline.",
            "Dispatch technician to the finance department immediately. Bring 3 replacement laptops from the loaner pool. Coordinate with the finance manager to prioritise the most critical workstations first.",
            "Pull the affected workstations' warranty status. If in warranty, open emergency hardware swap tickets with the vendor (Dell/Lenovo next-business-day dispatch). Provide loaners in the interim.",
            "Schedule a same-day hardware swap for all affected marketing machines. Ensure the imaging team is available to configure replacements immediately. Confirm resolution before the campaign launch time.",
            "Issue loaner devices to customer support agents within 30 minutes. Escalate the root cause investigation to the hardware team with a target resolution of 2 hours. Send status update to the support manager every 30 minutes.",
            "Provision a priority loaner device immediately. Set the original machine for same-day repair or replacement. Check warranty and engage vendor for expedited service if applicable.",
        ],
        "p3_actions": [
            "Schedule a hardware inspection for the next available technician slot (target: within 3 business days). Confirm the workaround is stable in the meantime.",
            "Run hardware diagnostics via Dell SupportAssist or HP PC Hardware Diagnostics. Check warranty status and create an RMA with the vendor if the device is still covered.",
            "Dispatch a field technician to inspect the device during the next scheduled floor visit. Provide standard repair or replacement within the 5-business-day SLA.",
            "Check SMART data and event logs for the failing component. Order replacement parts if needed and schedule the repair at a convenient time for the user.",
            "Log the issue in the asset management system. Schedule a hardware swap during the next maintenance window and notify the user 24 hours in advance.",
            "Verify BIOS and firmware are current. Run an extended memory and disk test overnight. Review results and arrange repair or replacement as needed.",
        ],
        "p4_actions": [
            "Log the cosmetic issue in the asset management system for record-keeping. No repair action required unless the user requests escalation.",
            "Note the device age in the asset register. Flag it for inclusion in the next quarterly tech refresh review — no immediate action.",
            "Create a low-priority hardware upgrade request. Submit to procurement for next-cycle budget consideration. Notify user when approved.",
            "Update the asset record with the physical damage note. Schedule a routine inspection during the next technician visit to the floor.",
            "Add the RAM upgrade request to the next-cycle hardware wish list. Notify the user when inventory allows or when budget is approved.",
            "Log the ergonomic equipment request. Route to the office management team for standard procurement via the self-service catalog.",
        ],
    },

    # ─── SOFTWARE ────────────────────────────────────────────────────────────

    "software": {
        "subjects": [
            "Microsoft Teams", "Outlook", "Excel", "SAP ERP", "Salesforce CRM",
            "Adobe Acrobat", "Zoom", "Chrome browser", "company portal app",
            "VPN client software", "antivirus software", "internal CRM tool",
            "AutoCAD", "JetBrains IDE", "Visual Studio Code", "Tableau",
            "Slack", "ServiceNow", "Jira", "GitHub Desktop",
            "QuickBooks", "SQL Server Management Studio", "Power BI",
        ],
        "symptoms": [
            "crashes immediately on launch", "freezes when opening large files",
            "stuck on the loading screen for over 10 minutes",
            "throwing an authentication error on every login attempt",
            "not syncing data and showing stale information",
            "running at less than 10% normal speed",
            "update failed mid-install, leaving it in a broken state",
            "core features are completely greyed out and inaccessible",
            "keeps logging out every 5 minutes and requiring re-authentication",
            "incompatible with the new OS version after the latest IT-pushed upgrade",
            "disappeared entirely after the latest patch",
            "showing data from a completely different user's account",
            "throwing a critical license validation error on startup",
            "unable to save files — getting a permissions error",
            "high-severity security warning blocking all functionality",
            "corrupting files on save",
            "consuming 100% CPU and making the machine unusable",
            "missing critical plugins after a forced update",
            "SSO integration broken — cannot log in via company account",
            "database connection timeout after 5 seconds",
            "reports generating incorrect data",
            "integration with third-party tool has stopped working",
        ],
        "details": [
            "I've cleared the cache and reinstalled but the issue persists.",
            "This started right after IT pushed an update overnight.",
            "My colleague on the same version and same machine has no issues.",
            "Error code shown: '0x80070005 — Access is denied.'",
            "I tried running as administrator but got the same result.",
            "Task Manager shows it consuming 95% of available memory.",
            "I've already logged a ticket with the software vendor but need a workaround now.",
            "The application worked fine this morning but broke after lunch.",
            "This is the third time this month this application has had this issue.",
            "I submitted the IT change request form 3 weeks ago with no response.",
        ],
        "p1_contexts": [
            "CRITICAL: Our entire ERP system is down. All financial transactions, purchase orders, and inventory management are halted. We cannot ship any orders. This is directly impacting revenue and our SLA with a major customer.",
            "URGENT: The company-wide authentication service is broken and no one in the organisation can log in to ANY company application. We have over 500 users unable to work. The CEO is involved.",
            "EMERGENCY: Production deployment pipeline is broken and we have a critical security patch that must go out by 18:00 today. Regulatory compliance is at risk if we miss this deadline.",
            "CRITICAL: Our customer-facing order management system is throwing errors for all customers. We have received 200+ complaints in the last hour. Revenue is being lost every minute.",
            "P1 INCIDENT: The payroll processing software has crashed mid-run on the day payroll must be submitted. 800 employees will not be paid on time if this is not resolved immediately.",
            "CRITICAL DATA INTEGRITY: The application is actively corrupting data on save. We have already detected corrupted records in the production database. ALL users have been instructed to stop working immediately.",
        ],
        "p2_contexts": [
            "This has taken down the entire sales department — 25 reps cannot log calls or create opportunities in Salesforce. We are in Q4 close and this is a high-severity business impact.",
            "Our analytics team is blocked on a deadline report for the CFO due at 5 PM today. The BI tool is broken and we have no workaround for the data they need.",
            "The broken SSO integration is blocking 3 teams from accessing their tools. An important board meeting at 14:00 relies on dashboards generated by these tools.",
            "Approximately 15 developers cannot push code because the CI/CD integration is throwing errors. Our sprint delivery is at risk for this week's release.",
            "The customer support team lead has confirmed that 12 agents cannot use the ticketing system. Customer response times are deteriorating and SLAs are being breached.",
            "Two hours until a critical demo to a major prospective client and the demo environment software is throwing errors. We urgently need this resolved or a workaround.",
        ],
        "p3_contexts": [
            "Just me experiencing this — my colleagues are unaffected. I've found a workaround by using the web version instead of the desktop client. Please fix when you get a chance.",
            "The issue is intermittent; it happens about once a day. I can usually resolve it by restarting the application. Would appreciate a proper fix within the week.",
            "This is an inconvenience but not a blocker. I can still complete my work, just with a few extra steps. Please address within your standard SLA.",
            "The broken feature is one I use occasionally, not daily. I have a manual workaround. No urgency — please schedule a fix at your convenience.",
            "This only affects my machine, not my teammates. IT can look at it during the next scheduled support visit or remotely whenever there's availability.",
            "The issue is minor — the application works but one specific report takes 3 minutes instead of 10 seconds. Annoying but I can live with it for now.",
        ],
        "p4_contexts": [
            "Low priority feature request: could IT look into enabling the dark mode option in the application? It's a cosmetic preference. No urgency.",
            "No rush — I'm requesting an older version of the software be installed on a secondary machine for compatibility testing purposes. This can wait weeks.",
            "Enhancement idea: the auto-save interval in our CRM is set to 30 minutes. I'd like it set to 5 minutes. Not urgent — just a quality of life improvement.",
            "Informational: I noticed the software version shown in the 'About' screen appears to be out of date. Just flagging it in case it needs updating. No immediate issue.",
            "Low priority: I'd like to request access to the optional analytics module we're not currently using. No deadline — please process whenever convenient.",
            "FYI: One of the keyboard shortcuts in the application was changed in the latest update. The old one is burned into my muscle memory. Could someone look into reverting it? No rush.",
        ],
        "p1_actions": [
            "IMMEDIATE: Declare P1 major incident. Engage the application owner, DBA, and infrastructure lead in a war-room call. Assess rollback to the last known-good version. Notify all affected users via broadcast. Update incident bridge every 30 minutes.",
            "CRITICAL: Rollback the last deployed patch immediately using the approved rollback runbook. Engage the vendor's critical support line. Do not re-apply the patch without root-cause analysis and full regression testing.",
            "EMERGENCY: Escalate to the application vendor's P1 support line and open an emergency case. Engage the CISO if the issue has security implications. Begin data integrity assessment in parallel.",
            "INCIDENT RESPONSE: Isolate the affected production system. Restore from the most recent clean backup. Conduct parallel investigation into data corruption scope. Notify compliance team immediately.",
            "P1 ESCALATION: Engage senior DevOps engineer to hot-patch the authentication service. Activate the emergency password reset flow for affected users. Communicate ETR to all staff via IT broadcast.",
            "CRITICAL: Immediately halt payroll run. Engage the payroll software vendor on their emergency P1 line. Identify the last valid payroll state and prepare a manual payroll run as contingency. Notify HR and Finance VPs.",
        ],
        "p2_actions": [
            "Escalate to the Level 2 application support team. Deploy a known-working configuration from the approved software baseline. Target restoration within 2 hours and send status updates to the department manager every 30 minutes.",
            "Collect crash dumps and application logs from the affected machines. Engage the software vendor's standard support channel with full diagnostic package. Provide workaround (web version or prior release) to affected users immediately.",
            "Check the known issues board for this application version. Apply documented workaround or hotfix. If no workaround exists, escalate to the vendor and commit to a resolution time with the business.",
            "Clear application cache, repair installation via Settings > Apps, and test. If unresolved, perform a clean reinstall from the approved software center. Prioritise the highest-impact users first.",
            "Review Group Policy and permissions settings that may have been modified by the recent patch. Restore previous policy if a regression is identified. Document the change for the change management log.",
            "Provision a temporary licensed instance of the software on an alternative machine for the affected team while the primary issue is investigated. Escalate the root cause to the vendor.",
        ],
        "p3_actions": [
            "Schedule a remote support session within 2 business days. Verify the installed version against the approved software catalog and push an update if outdated.",
            "Clear the application cache and repair the installation via Settings > Apps. If the issue persists, perform a clean reinstall from the software center during off-hours.",
            "Collect event logs from Event Viewer and review for recurring errors. If a pattern is identified, escalate to the application team with the log package.",
            "Verify that Group Policy settings match the expected configuration for the user's role. Restore defaults if a misconfiguration is detected.",
            "Check whether the issue is specific to the user profile. Test with a new temporary profile to isolate. If profile-specific, rebuild the profile using the standard procedure.",
            "Uninstall the application and perform a clean reinstall from the software center. Re-apply the user's preferences and test all core functions before closing the ticket.",
        ],
        "p4_actions": [
            "Log the enhancement request. Add it to the application team's backlog for consideration in the next minor release cycle. Notify the user when it is prioritised.",
            "Route the version installation request to the software asset management team for compatibility review. Approve and deploy when the assessment is complete.",
            "Raise a configuration change request with the application admin team to adjust the auto-save interval. Implement during the next maintenance window.",
            "Verify the version number in the 'About' screen against the software catalog. If a discrepancy is found, schedule an update during the next maintenance window.",
            "Create a provisioning request for the analytics module license. Route to the software asset manager for budget approval. Provision when approved.",
            "Log the keyboard shortcut preference request with the application team. Review whether a user-configurable shortcut option is available in the settings.",
        ],
    },

    # ─── NETWORK ─────────────────────────────────────────────────────────────

    "network": {
        "subjects": [
            "VPN connection", "Wi-Fi", "wired ethernet", "Microsoft Teams calls",
            "internet access", "network drive mapping", "DNS resolution",
            "proxy settings", "remote desktop connection", "cloud storage sync",
            "video conferencing infrastructure", "internal website", "firewall",
            "network switch", "SD-WAN link", "BGP routing", "corporate WAN",
            "site-to-site VPN tunnel", "load balancer", "DHCP server",
            "network monitoring system", "Wi-Fi access point",
        ],
        "symptoms": [
            "drops every few minutes", "running at less than 5% of normal speed",
            "cannot connect at all", "works on Wi-Fi but not ethernet",
            "only affects traffic to specific external sites",
            "disconnects mid-call causing dropped meetings",
            "shows 'connected' but has no internet access",
            "latency spikes above 2,000 ms causing severe packet loss",
            "times out consistently when accessing shared drives",
            "cannot authenticate to the corporate network",
            "40% packet loss causing audio and video to cut out",
            "IP address conflict detected — two machines with same address",
            "DNS queries resolving to incorrect addresses",
            "BGP session has been flapping for the past 3 hours",
            "entire subnet is unreachable",
            "firewall is blocking all traffic on port 443",
            "SD-WAN failover has not activated despite primary link being down",
            "DHCP pool is exhausted — new devices cannot get addresses",
            "network bandwidth saturated at 100% — unknown source",
            "asymmetric routing causing session drops",
            "wireless client cannot obtain a DHCP lease",
            "site-to-site VPN tunnel keeps renegotiating",
        ],
        "details": [
            "I'm working from home on a 100 Mbps connection.",
            "This only happens during peak hours between 9 AM and 11 AM.",
            "Speed test shows 3 Mbps download — should be 200 Mbps.",
            "Other devices on the same network work correctly.",
            "Started after I moved to a different office location.",
            "The issue disappears when I disconnect from VPN.",
            "I'm using the company-issued router.",
            "Traceroute shows the bottleneck at the firewall hop.",
            "The NOC has not yet acknowledged the incident.",
            "This issue started immediately after the network maintenance window last night.",
        ],
        "p1_contexts": [
            "CRITICAL OUTAGE: The entire corporate WAN is down. All three of our office locations — London, New York, and Singapore — have lost connectivity to the data centre. Over 1,200 employees cannot access any internal systems. The BGP session with our ISP has been down for 47 minutes.",
            "EMERGENCY: The primary SD-WAN link for our headquarters is down and the automatic failover to the secondary link has not triggered. All site-to-site VPN tunnels are broken. Our DR site cannot replicate. Estimated business impact: $10,000 per minute.",
            "P1 INCIDENT: The production load balancer has stopped forwarding traffic. Our customer-facing web application and API are completely unreachable by external customers. We have confirmed this from multiple external monitoring locations.",
            "CRITICAL: The DHCP server has crashed and the IP pool is exhausted. No new device on the entire corporate network can obtain an IP address. Employees returning from the weekend cannot get on the network.",
            "URGENT: A firewall misconfiguration deployed 20 minutes ago has blocked all HTTPS traffic on port 443 across the entire organisation. No internal SAAS applications are reachable. Change management has been notified and rollback is needed urgently.",
            "CRITICAL: Network monitoring is showing 40% packet loss on the core switch uplinks. Multiple floor switches have lost connectivity. The NOC has no visibility. We suspect a hardware failure in the core routing infrastructure.",
        ],
        "p2_contexts": [
            "The entire engineering floor — approximately 60 people — has lost wired connectivity. We are operating on Wi-Fi only, which is insufficient for the large file transfers our work requires. Production deployments are being delayed.",
            "Our video conferencing bridge is dropping calls every 20 minutes and the audio quality is severely degraded. We have three external client calls in the next 2 hours that we cannot reschedule.",
            "The VPN connection is unstable for our entire remote workforce today. About 40 remote employees are experiencing repeated disconnections, which is impacting their ability to attend meetings and access file servers.",
            "DNS resolution to our internal services has been broken since this morning. Our dev team cannot reach the internal CI/CD servers, blocking today's planned release. The issue affects everyone on the developer VLAN.",
            "Network speed on the 4th floor is degraded to 1 Mbps instead of the usual 1 Gbps. 25 employees are affected. Our operations team is struggling to process time-sensitive logistics orders.",
            "The site-to-site VPN between our main office and the warehouse has been flapping all morning. Warehouse staff are losing access to the inventory management system, causing delays in order fulfilment.",
        ],
        "p3_contexts": [
            "I'm the only person on my team experiencing this VPN instability. My colleagues on the same subnet are fine. I have a workaround by reconnecting, but it interrupts my workflow a few times per day.",
            "The issue is intermittent — it happens roughly once an hour and connecting again resolves it. Not a blocker but please look into it when you have capacity.",
            "Only my workstation seems affected. I can use the Wi-Fi as a backup. Please look into it during the next scheduled network maintenance.",
            "The network latency is slightly elevated on my machine but not enough to prevent me from working. I'd like it investigated within the week.",
            "My network drive sometimes takes 30 seconds to reconnect in the morning. Once connected it works fine. A bit annoying but not critical — whenever IT has time.",
            "I occasionally notice packet loss during video calls but it's minor and infrequent. Please investigate at your convenience.",
        ],
        "p4_contexts": [
            "Low priority: I'd like to request a static IP address for my workstation to simplify remote access configuration. No urgency — whenever the network team has bandwidth.",
            "Informational: I noticed the Wi-Fi signal strength in the far corner of our office seems weaker than the rest. Not a blocker, but could we consider an additional access point in the next office upgrade?",
            "FYI: The intranet speed test page I use shows slightly lower results than expected. Everything works fine in practice. Just flagging in case it's a known measurement artifact.",
            "Enhancement request: could the network team whitelist a specific external tool our team uses for analysis? It's currently filtered by the proxy. Low priority — please route through the change management process.",
            "No rush: I'd like to understand our current bandwidth allocation and whether we could negotiate more capacity from the ISP in the next contract cycle. Purely informational.",
            "Low priority: I'm setting up a home lab and would like advice on connecting it to the corporate VPN for development purposes. Not urgent — happy to schedule a call whenever convenient.",
        ],
        "p1_actions": [
            "IMMEDIATE P1: Activate the network operations war-room. Engage senior network engineer and ISP escalation contact. Initiate emergency change to restore BGP sessions. Broadcast an all-staff communication about the outage. Update every 30 minutes.",
            "CRITICAL: Initiate manual failover to the secondary SD-WAN link. Engage the SD-WAN vendor's 24/7 P1 support line. Activate the DR replication runbook. Dispatch on-site engineer to investigate primary link hardware.",
            "EMERGENCY ROLLBACK: Identify and roll back the firewall configuration change immediately using the approved emergency change procedure. Do not deploy additional changes until root cause is fully understood.",
            "CRITICAL: Restore DHCP service from backup configuration. Expand the DHCP pool as an immediate workaround. Investigate the root cause of the pool exhaustion or service crash. Notify affected users once DHCP is restored.",
            "P1: Replace the suspected failing core switch hardware immediately from the spare-parts inventory. Engage the hardware vendor's emergency support. Restore NOC visibility and monitor for further drops.",
            "CRITICAL INCIDENT: Engage the load balancer vendor on their emergency P1 line. Deploy the hot-standby load balancer from the DR environment. Restore customer traffic and notify the communications team to post a status page update.",
        ],
        "p2_actions": [
            "Escalate to the senior network engineer with a 2-hour SLA. Investigate the wired switch configuration for the affected floor. Deploy a temporary Wi-Fi boost solution while the root cause is diagnosed.",
            "Engage the video conferencing platform's support team and the internal network team simultaneously. Run a packet capture on the affected calls to identify the source of drops. Provide a dial-in phone number as a fallback for the client calls.",
            "Investigate the VPN server's session table for the affected user group. Check for capacity issues or misconfigured timeout values. Push a VPN client update if a known fix is available.",
            "Check the internal DNS server and zone configuration for the affected domain names. Flush the DNS cache on affected clients. Restore correct DNS records within the 2-hour SLA.",
            "Identify the source of bandwidth saturation on the affected floor switch using NetFlow data. Implement QoS policy to prioritise business-critical traffic while the root cause is investigated.",
            "Stabilise the site-to-site VPN tunnel by renegotiating IKE parameters. Coordinate a maintenance window with the warehouse IT contact to apply the permanent fix.",
        ],
        "p3_actions": [
            "Schedule a remote network diagnostics session. Check VPN client logs and server-side session logs for timeout patterns. Push updated VPN client configuration if available.",
            "Run network diagnostics and verify DNS settings match the corporate standard. Flush the DNS cache and test resolution. Escalate to the NOC if the issue recurs after the fix.",
            "Release and renew the DHCP lease on the affected machine. Verify no IP conflict exists. Flush the DNS cache and test connectivity.",
            "Review proxy auto-configuration (PAC) file accessibility and correctness for the affected user. Update if out of date.",
            "Collect a packet capture during a slow period and analyse for anomalies. Identify whether the issue is client-side, switch-side, or upstream. Apply the appropriate fix.",
            "Schedule a cable infrastructure check during the next floor technician visit. Test the patch panel connection for the affected port.",
        ],
        "p4_actions": [
            "Log the static IP request in the IPAM system. Assign the next available address in the correct subnet and update the DHCP reservation. Notify the user when complete.",
            "Log the Wi-Fi coverage observation. Add it to the office infrastructure improvement backlog for consideration in the next capacity planning review.",
            "Review the intranet speed test tool's methodology and update if the baseline measurement is out of date. No network changes required.",
            "Create a change request to whitelist the requested tool in the proxy configuration. Route through the standard change management process for approval and deployment.",
            "Schedule an informational call with the network team to review bandwidth utilisation reports and contract options. No changes required at this stage.",
            "Log the home lab VPN enquiry. Route to the network security team for a policy review. They will contact the user with guidance when the review is complete.",
        ],
    },

    # ─── SECURITY ────────────────────────────────────────────────────────────

    "security": {
        "subjects": [
            "suspicious email", "phishing attempt", "unauthorised login alert",
            "malware detection", "data breach concern", "compromised account",
            "expired security certificate", "MFA prompt", "firewall alert",
            "encrypted file", "USB device policy violation", "access log anomaly",
            "ransomware warning", "data exfiltration alert", "DLP policy trigger",
            "insider threat indicator", "leaked credentials", "social engineering attempt",
            "supply chain software alert", "privileged account misuse",
            "zero-day vulnerability notification", "security audit finding",
        ],
        "symptoms": [
            "received an email with a suspicious attachment I did not open",
            "got a login notification from a country I have never visited",
            "antivirus quarantined 47 files simultaneously",
            "browser is redirecting to unfamiliar websites",
            "account locked after multiple failed login attempts from unknown IPs",
            "MFA codes are being sent in rapid succession but I did not initiate them",
            "a security warning popup is blocking all work and cannot be dismissed",
            "colleagues are receiving emails that appear to come from my address",
            "a sensitive document was found in a publicly accessible shared folder",
            "an SSL certificate on an internal tool has expired",
            "unexpected admin privilege escalation detected in audit logs",
            "a ransomware-style message appeared on a workstation in the open office",
            "DLP system has flagged a large upload of sensitive data to an external site",
            "an endpoint detection tool has flagged a process as a known RAT",
            "leaked employee credentials found on a dark web intelligence feed",
            "a privileged service account was used to log in interactively after hours",
            "supply chain alert: a third-party library we use has a critical CVE",
            "security audit identified 150 accounts with passwords older than 2 years",
            "a laptop with unencrypted sensitive data was reported lost or stolen",
            "AV tool detected a rootkit on a production server",
            "an internal API key was accidentally committed to a public GitHub repository",
            "a phishing simulation test revealed 30% click rate on a spear-phishing email",
        ],
        "details": [
            "I did NOT click on anything — just reporting it as instructed.",
            "The email looked exactly like it came from our CEO, including her signature.",
            "This happened at 3 AM local time when I was definitely asleep.",
            "I changed my password immediately as a precaution.",
            "Multiple people in my department received the same suspicious email.",
            "The alert came from our CrowdStrike Falcon endpoint detection platform.",
            "I was travelling internationally when this activity was detected.",
            "I'm not certain whether I accidentally opened a link last Tuesday.",
            "I've already isolated my machine from the network as a precaution.",
            "The security team should be aware this is time-sensitive.",
        ],
        "p1_contexts": [
            "CRITICAL SECURITY INCIDENT: Our EDR platform has confirmed an active ransomware attack. Multiple file servers are showing mass encryption events right now. The CISO has been notified and the incident response team needs to be activated immediately.",
            "EMERGENCY: A DLP alert has confirmed that 50,000 customer records — including PII and financial data — have been exfiltrated to an external IP address in the last 2 hours. This is a reportable data breach under GDPR. Legal and compliance must be notified immediately.",
            "CRITICAL: A confirmed compromise of a domain administrator account. The attacker has already moved laterally to 6 servers based on our SIEM correlation. The SOC is requesting an emergency bridge call. This is a full-blown intrusion in progress.",
            "EMERGENCY: A production API key with full read-write access to our customer database was committed to a public GitHub repository 3 hours ago. We do not know if it has been discovered and exploited. Rotate the key immediately and audit all access logs.",
            "CRITICAL: A laptop containing an unencrypted copy of our Q4 financial statements — including unreleased earnings data — has been confirmed stolen from an airport. This is a potential material breach with SEC reporting obligations.",
            "CRITICAL VULNERABILITY: A zero-day exploit with a public PoC is actively targeting our version of [software]. Our vulnerability scanner has confirmed we are exposed. Emergency patching must occur within 4 hours to meet our security policy SLA.",
        ],
        "p2_contexts": [
            "Our SIEM has flagged 15 accounts with login activity from unusual geographic locations over the past 24 hours. While no confirmed compromise has been detected, the volume is abnormal and warrants immediate investigation before more damage can occur.",
            "An employee has confirmed they clicked a link in a phishing email and entered their credentials. The account is still active. We need to immediately lock it down and investigate what access was made.",
            "Audit logs show a service account was used to access production data outside of its normal operational pattern last night. This could be a compromised credential or an insider threat. Needs urgent investigation.",
            "A high-severity CVE (CVSS 9.2) affecting a widely deployed system has been announced. We have confirmed 25 vulnerable instances in our environment. We need to begin emergency patching within 24 hours per our policy.",
            "The security team has confirmed a supply chain alert — a library used across 40 of our internal applications has a known RCE vulnerability. We need to begin impact assessment and remediation immediately.",
            "A DLP alert flagged an employee uploading a file containing ~500 customer records to personal cloud storage. Investigation needs to begin today to determine intent and whether the data has been shared externally.",
        ],
        "p3_contexts": [
            "I received a suspicious email and did not click anything. I'm reporting it for the security team's awareness. No immediate action required — just please analyse it and let me know if it's a known campaign.",
            "I got an MFA push notification I didn't initiate. I denied it. My password is still strong and unchanged. I'm reporting this in case it's part of a broader pattern. Please investigate at your convenience.",
            "My machine was showing some unusual activity — higher CPU usage than normal — for about an hour yesterday. The AV scan showed nothing. I'm probably overreacting but want to log it just in case.",
            "I noticed a colleague's access to a shared drive has not been revoked after they left the team last month. Please revoke their permissions at your next availability.",
            "An internal certificate for a non-critical internal tool has expired. The tool is still accessible via HTTP but shows a security warning. Please renew when you get a chance.",
            "I received what appears to be a spear-phishing email targeting me specifically. I have not clicked anything. Please review it and update the email filter. No immediate personal risk.",
        ],
        "p4_contexts": [
            "Low priority: I'd like to request a security awareness training refresher for my team. No urgency — please schedule whenever the security team has capacity.",
            "FYI: I noticed the office Wi-Fi password has not been rotated in over a year. This is an enhancement suggestion for the security team's next review cycle.",
            "Informational: A colleague asked me about the company's policy on using personal devices. I pointed them to the acceptable use policy but thought I'd flag it in case a policy update communication would be useful.",
            "Low priority: Could the security team review whether our current password complexity requirements are aligned with the latest NIST guidelines? Just a policy hygiene check for next quarter.",
            "Enhancement request: I'd like to enable FIDO2 hardware key authentication for my account in addition to the current app-based MFA. Please advise on the process when convenient.",
            "No rush: I'd like to understand what personal data of mine the company holds as per GDPR right-of-access rules. This is a personal curiosity request — no deadline.",
        ],
        "p1_actions": [
            "CRITICAL INCIDENT RESPONSE: Activate the incident response plan immediately. Isolate all affected endpoints from the network. Engage the CISO, legal, and compliance teams. Open a war-room bridge. Do not power down affected machines — preserve forensic evidence. Update leadership every 30 minutes.",
            "EMERGENCY DATA BREACH: Engage the Data Protection Officer and Legal immediately. Preserve all access logs and DLP alerts as evidence. Begin GDPR breach assessment — 72-hour reporting clock may have started. Notify the affected customers if required by regulation.",
            "ACTIVE INTRUSION: Revoke all credentials for the compromised admin account immediately. Force re-authentication for all admin sessions. Engage the MSSP / SOC for 24/7 monitoring. Initiate forensic imaging of all affected servers before any remediation.",
            "CREDENTIAL EXPOSURE: Rotate the leaked API key immediately — treat this as a P1 regardless of whether exploitation is confirmed. Audit all API access logs for the last 72 hours. Notify the security team and initiate a full credential audit.",
            "LOST DEVICE: Remotely wipe the device immediately using MDM. Begin a GDPR/regulatory breach assessment. Notify legal and the CISO. Preserve all location and access logs. Issue a follow-up communication to affected data subjects if required.",
            "EMERGENCY PATCH: Deploy the emergency patch to all vulnerable systems immediately following the emergency change procedure. If a patch is unavailable, implement the vendor's recommended mitigation within 4 hours. Escalate to the CISO with status every hour until all instances are patched.",
        ],
        "p2_actions": [
            "Investigate the suspicious login activity using SIEM correlation. Force a password reset and MFA re-enrolment for all flagged accounts. Enable enhanced monitoring for these accounts for the next 30 days.",
            "Immediately lock the compromised account and revoke all active sessions. Investigate what resources were accessed during the credential exposure window. Notify the user and their manager. Re-enable the account only after a fresh password and MFA are confirmed.",
            "Investigate the service account anomaly using SIEM and PAM system logs. If compromise is confirmed, revoke the service account and rotate all associated secrets. Escalate to the security team for a full investigation within the 4-hour SLA.",
            "Begin emergency patch assessment for the high-severity CVE. Deploy patches to the highest-risk instances first. Complete all patching within 24 hours. Engage the vendor for mitigation guidance if a patch is not yet available.",
            "Conduct a rapid impact assessment of the vulnerable library across all affected applications. Prioritise remediation by internet-facing vs internal exposure. Begin patching the highest-risk applications within 2 hours.",
            "Secure-copy the DLP alert evidence. Begin a preliminary investigation to determine whether the upload was intentional or accidental. Engage HR if an intentional breach is suspected. Secure the data and ensure it has been removed from external storage.",
        ],
        "p3_actions": [
            "Forward the suspicious email to the SOC phishing inbox for automated and manual analysis. Block the sender domain if it is confirmed malicious. Update the email security filter with the phishing indicator.",
            "Review the failed MFA push attempts in the identity provider logs. Check whether the source IP is associated with known malicious activity. If abnormal, force a password reset as a precaution and enable additional logging on the account.",
            "Run a full endpoint scan on the machine showing unusual activity. Review process execution and network connection logs for the suspicious period. Close the ticket if clean; escalate to security if anomalies are found.",
            "Revoke the former team member's access permissions across all systems. Conduct an access audit for that user and document the findings. Update the offboarding checklist to prevent future gaps.",
            "Renew the internal certificate from the approved CA. Deploy the new certificate to the affected service. Test that the warning is resolved and close the change ticket.",
            "Analyse the spear-phishing email headers and payload in a sandbox. Update email security rules to block similar messages. Add the indicators to the threat intelligence feed.",
        ],
        "p4_actions": [
            "Schedule a security awareness training session for the requesting team. Coordinate with the security team on dates and send calendar invites to participants.",
            "Log the Wi-Fi password rotation suggestion in the security improvement backlog. Review it at the next quarterly security review meeting.",
            "Send an updated acceptable use policy communication to the relevant team. Ensure the policy page on the intranet is current.",
            "Add the NIST password policy review to the agenda for the next security policy review meeting. No immediate changes required.",
            "Log the FIDO2 hardware key request. Route to the identity team for an assessment of the enrolment process and compatibility. Contact the user with instructions when the review is complete.",
            "Provide the user with the GDPR Subject Access Request (SAR) form and process information. Route to the DPO team to fulfil within the statutory 30-day period.",
        ],
    },

    # ─── ACCESS ──────────────────────────────────────────────────────────────

    "access": {
        "subjects": [
            "SharePoint site", "shared network drive", "Jira project",
            "Confluence space", "admin panel", "production database",
            "GitHub repository", "AWS console", "HR portal",
            "expense reporting tool", "customer data dashboard", "build server",
            "Active Directory group", "Okta application", "Salesforce org",
            "production deployment pipeline", "finance reporting system",
            "CRM admin role", "network device management console",
            "identity provider admin panel", "security monitoring dashboard",
            "vendor portal", "executive reporting suite",
        ],
        "symptoms": [
            "getting 'Access Denied' on every attempt to open",
            "can view files but cannot edit or save changes",
            "access was revoked without any notification or explanation",
            "need access urgently for a new critical project assignment",
            "permissions changed incorrectly after an internal system migration",
            "temporary contractor needs time-limited access for a 6-week engagement",
            "account shows the wrong department in the company directory",
            "SSO login loops back to the sign-in page endlessly",
            "can access from the office but not from the VPN remotely",
            "new hire needs immediate day-one access to all standard tools",
            "service account permissions expired and a production job is failing",
            "manager approval has been pending in the queue for over 2 weeks",
            "elevated access needed for an emergency production incident",
            "access provisioned but to the wrong environment (test vs production)",
            "MFA device is lost and the account is locked out",
            "inherited access from a previous role has not been removed",
            "role-based access control changes broke multiple users' permissions",
            "privileged access management (PAM) vault is not releasing credentials",
            "access expiry not extended despite active project continuation",
            "joint-venture partner needs read-only access to specific data",
        ],
        "details": [
            "My manager already approved this in the ticketing system 3 days ago.",
            "I was transferred to this team last Monday and need access to do my job.",
            "I need read-only access only — not full admin rights.",
            "The project deadline is this Friday and I cannot start without this.",
            "I had access until the system migration last weekend removed it.",
            "HR has confirmed my role change in Workday but the systems haven't updated.",
            "I'm a contractor — my access should be time-limited to 6 weeks.",
            "My colleague with the same role and title has this access already.",
            "The access request was submitted 3 weeks ago with no response.",
            "The resource owner has already confirmed in email that I should have access.",
        ],
        "p1_contexts": [
            "CRITICAL: A production outage is in progress RIGHT NOW and the on-call engineer does not have access to the production database they need to execute the hotfix. We cannot resolve the incident without this access being granted in the next 5 minutes.",
            "EMERGENCY: Our incident response team cannot access the security monitoring dashboard during an active security incident. We are blind to attacker activity. This is preventing us from containing a live intrusion.",
            "URGENT: The PAM vault has stopped releasing credentials for 3 critical service accounts. Four automated production jobs have failed. Revenue-generating processes are halted. We need emergency access restoration immediately.",
            "CRITICAL: A key engineer is locked out of the AWS production console during a high-severity infrastructure incident. We have an RDS failover in progress and no one with access is available. This is an emergency break-glass scenario.",
            "P1 BLOCKER: All members of the DevOps team lost access to the production deployment pipeline simultaneously after an IAM policy change. We have a critical security patch that must be deployed within 2 hours and we are completely blocked.",
            "CRITICAL: The service account used by our payment processing system has expired. All payment transactions are failing. We are losing thousands of dollars per minute. The service account must be renewed or replaced immediately.",
        ],
        "p2_contexts": [
            "A new senior hire starts on Monday and does not yet have access to any of the systems they need to do their job. The role is customer-facing and they need to be productive from day one. Please expedite provisioning.",
            "My access to the finance reporting system was removed during migration but no one on my team has noticed the impact yet. Month-end reporting is in 3 days and I am the primary report owner.",
            "The entire dev team lost access to the GitHub repository after an org settings change. We have a release planned for tomorrow and cannot push or review code.",
            "A consultant engaged for an urgent 2-week engagement starts tomorrow and needs time-limited access to 5 systems. The engagement cannot begin without this access.",
            "I need emergency elevated access to investigate a high-priority application issue. My normal access is read-only. The application owner has approved this verbally — I need it formally provisioned within 2 hours.",
            "Three new starters in the customer support team have no access to the CRM system on their first day. They are assigned to live customer queues and cannot take calls without it. This is an urgent onboarding failure.",
        ],
        "p3_contexts": [
            "I need access to a SharePoint site for a project I've joined. My manager approved it last week. I've been working around it by asking colleagues to share files, but it would be better to have proper access.",
            "My access to Confluence expired but the project continues. Please extend it for another 3 months. Not urgent — I can use a colleague's screen share as a workaround for now.",
            "A contractor who finished their engagement last month still appears to have access to our shared drive. Please revoke it when you get a chance — no urgency.",
            "I need read-only access to a reporting dashboard for a new responsibility I've taken on. My manager has approved it. No hard deadline — please process within the standard SLA.",
            "My account is showing the wrong cost centre after an internal reorg. This causes some minor issues with expense approvals. Please update it within the next week.",
            "I need access to a test environment for a project I'm starting next week. The project doesn't begin for 5 business days so no rush — please process within the standard 3-day SLA.",
        ],
        "p4_contexts": [
            "Low priority: I'd like to request optional read-only access to the company analytics dashboard out of professional curiosity. No business requirement — just informational. Process whenever convenient.",
            "No rush: I'd like to be added to the 'all-engineering' distribution group for visibility on technical discussions. Happy to wait a few weeks.",
            "Enhancement: I'd like to request that the access review process for my team be moved from quarterly to semi-annual, as the quarterly cadence creates administrative overhead. Please consider for the next process review.",
            "FYI: A colleague who moved teams 6 months ago still appears in our Jira project as a member. Not causing any issues but might be worth tidying up in the next access cleanup.",
            "Informational: I noticed I have admin access to a tool I no longer use from a previous project. Happy to have it revoked at your convenience to reduce my attack surface.",
            "Low priority: I'd like to understand what systems my service account has access to. A personal audit for documentation purposes — no changes needed. Please share the report when available.",
        ],
        "p1_actions": [
            "EMERGENCY BREAK-GLASS: Invoke the emergency access procedure documented in the runbook. Grant minimum necessary access immediately via the PAM emergency override. Document all actions taken. Review and revoke the emergency access once the incident is resolved.",
            "CRITICAL: Provision emergency elevated access immediately following the break-glass procedure. Verify the requester's identity via phone with their manager. Log all granted permissions in the PAM audit trail. Revoke within 4 hours or when the incident is closed.",
            "P1 RESPONSE: Renew or replace the expired service account immediately. Test the production jobs to confirm they resume. Investigate why the expiry alert was not actioned and implement a preventive control.",
            "URGENT: Restore the DevOps team's pipeline access by reverting the IAM policy change to the last known-good state. Use the emergency change procedure. Test all pipeline functions before closing.",
            "CRITICAL: Grant the on-call engineer emergency read-write access to the production database following the break-glass SOP. Revoke access upon incident resolution and perform an access audit.",
            "EMERGENCY: Unlock the security team's access to the SIEM and monitoring dashboard immediately. Escalate to the identity team's on-call engineer. This is blocking an active incident response.",
        ],
        "p2_actions": [
            "Expedite the onboarding access provisioning for the new hire. Coordinate with the HR system to confirm role alignment. Target all standard tools provisioned by end of day before the start date.",
            "Restore the finance team member's access to the reporting system immediately. Verify which permissions were removed during migration and re-apply the correct role assignment.",
            "Revert the GitHub organisation settings change that removed the dev team's access. Test repository access for all affected members before closing. Escalate to the platform owner for root cause analysis.",
            "Provision time-limited access for the contractor across all requested systems. Set automatic expiry for the end of the engagement. Assign the line manager as the access owner.",
            "Provide elevated temporary access via the PAM system for the investigation period (max 8 hours). Confirm verbal approval with the application owner via email for audit purposes. Revoke automatically after the investigation window.",
            "Escalate the CRM access provisioning for the 3 new starters to the identity team immediately. Target completion within 30 minutes. Notify the customer support manager once access is live.",
        ],
        "p3_actions": [
            "Verify manager approval in the access management system and provision the requested SharePoint access within the standard 24-hour SLA. Notify the user when done.",
            "Extend the Confluence access for the requested 3-month period. Set an expiry date in the access management system and notify the manager 2 weeks before it expires.",
            "Revoke the former contractor's access across all systems. Run a full access report for that user and confirm all permissions have been removed. Document the finding.",
            "Cross-reference the user's role in Active Directory with the access control matrix. Grant the appropriate read-only dashboard role and notify the user when provisioned.",
            "Update the user's cost centre in Active Directory and the HR system. Verify expense approval workflows are functioning correctly after the change.",
            "Create the test environment access request in the provisioning queue. Apply the standard SLA (3 business days). Notify the user when access is available.",
        ],
        "p4_actions": [
            "Log the analytics dashboard access request. Route to the data governance team for review of the access control policy. Provision if approved under the self-service model.",
            "Add the user to the requested distribution group. Update the group membership record in Active Directory.",
            "Log the access review frequency request as a process improvement suggestion. Add to the IT governance team's next quarterly review agenda.",
            "Remove the former team member from the Jira project in the next scheduled access cleanup. Log the change for audit purposes.",
            "Remove the unused admin access at the user's request. Log the change in the access management system. Commend the user for proactive access hygiene.",
            "Generate and share a service account access report with the requester. File a copy in the asset management system for documentation purposes.",
        ],
    },

    # ─── EMAIL ────────────────────────────────────────────────────────────────

    "email": {
        "subjects": [
            "Outlook desktop client", "Outlook web app", "email delivery",
            "calendar invites", "distribution list", "shared mailbox",
            "email signature", "auto-reply settings", "email attachment limit",
            "inbox rules", "spam filter", "email archiving",
            "Exchange mail flow", "email domain", "email encryption",
            "NDR bounce message", "journaling configuration", "email delegation",
            "calendar permissions", "meeting room booking",
            "external email delivery", "email retention policy",
        ],
        "symptoms": [
            "not receiving any emails from external senders",
            "sent emails bouncing with a permanent 550 NDR error",
            "calendar invites showing the wrong timezone for all recipients",
            "shared mailbox not appearing in Outlook for the entire team",
            "unable to send any attachments larger than 5 MB",
            "all inbound emails going to junk with no way to whitelist",
            "auto-reply not activating despite being correctly configured",
            "email signature not displaying in replies or forwarded messages",
            "search returning no results for emails from the last 3 months",
            "synchronisation completely stopped — no new emails loading in 2 hours",
            "distribution list silently dropping 30% of recipients",
            "emails stuck in outbox and failing to send since yesterday",
            "NDR containing 'IP address blocked' for all emails to a major client",
            "meeting requests not being delivered to attendees",
            "encrypted email flow broken after certificate renewal",
            "journaling rule has stopped capturing emails for compliance",
            "email delegation no longer working after a migration",
            "calendar showing as 'busy' to all external users at all times",
            "domain SPF/DKIM check failing causing deliverability issues",
            "incoming email delayed by 4+ hours consistently",
        ],
        "details": [
            "This started 3 days ago with no known change on my end.",
            "The bounce message references 'IP reputation block' on the mail server.",
            "I checked the spam/junk folder thoroughly — nothing there.",
            "Multiple other users can send to the same external address successfully.",
            "I'm on Microsoft 365 Business Premium.",
            "My mailbox is at 48 GB out of 50 GB quota.",
            "This works fine on my phone but not on the desktop Outlook client.",
            "IT made changes to our mail flow rules during the maintenance window last week.",
            "I have a client waiting on this email who has escalated to my manager.",
            "The issue affects all members of my team, not just me.",
        ],
        "p1_contexts": [
            "CRITICAL: Our organisation's outbound email has been completely blacklisted by the top 5 email providers. No email is being delivered to ANY external recipient. We cannot communicate with customers, partners, or vendors. Every customer-facing operation is impacted.",
            "EMERGENCY: The journaling service capturing all emails for regulatory compliance has stopped working. We are a financial services firm regulated by the FCA. A compliance audit starts tomorrow. This is a regulatory reporting failure with potential fines.",
            "CRITICAL: Our primary email domain's MX records were accidentally deleted during a DNS migration. All inbound email is being rejected. We have received zero external emails for the past 3 hours.",
            "URGENT: The encrypted email gateway used by our legal team for privileged communications has failed. Active litigation communications cannot be sent securely. Our outside counsel is waiting for time-sensitive documents due today.",
            "CRITICAL: The email system is down for the entire organisation — Exchange has lost connectivity to Active Directory. Nobody in the company can send or receive email. This is a complete email blackout.",
            "P1 DATA BREACH: A mail rule appears to be silently copying all emails from the CEO and CFO mailboxes to an external address. This is a suspected insider threat or account compromise. Immediate investigation required.",
        ],
        "p2_contexts": [
            "Our entire sales team cannot receive external emails. We have a high-value proposal deadline today and are expecting client responses. The mail flow issue is directly impacting revenue.",
            "All calendar invites sent by our office have been showing the wrong timezone for 24 hours. We had 3 missed meetings with clients today as a result. This needs to be fixed before tomorrow's executive calendar.",
            "The shared customer support mailbox has been inaccessible to the team for 4 hours. We have 200+ unread customer emails piling up. Customers are contacting our social media channels due to no response.",
            "Our domain's IP has been added to a major blacklist. Emails to our largest customer are being rejected. They have called us directly to report the issue. We need the delisting completed urgently.",
            "The distribution list for our all-hands meeting tomorrow is failing to deliver to 30% of staff. The invitations for a company-wide mandatory meeting need to reach everyone by EOD today.",
            "Email encryption has broken between our firm and our primary law firm. We have sensitive contracts to exchange today. The issue must be resolved within 3 hours.",
        ],
        "p3_contexts": [
            "I'm having trouble receiving emails from one specific external domain. My other inbound email works fine. This has been happening for 3 days. Please investigate when you have bandwidth.",
            "My Outlook profile is behaving oddly — it's slow to load and search is intermittent. I can use the web app as a workaround but I prefer the desktop client. Not urgent.",
            "My auto-reply isn't working correctly — it's replying to internal emails when it should only reply to external ones. Not a blocker but please fix when convenient.",
            "The email signature on my replies is slightly out of date — the phone number changed last month. Please help me update it when you get a chance. Low urgency.",
            "Calendar invites I send occasionally arrive without the meeting details in the body. The link still works. Intermittent and not critical — please look into it within the week.",
            "My mailbox is approaching its quota limit. Could IT help me archive older items or increase the quota? No immediate issue but I'd like to sort it before it becomes one.",
        ],
        "p4_contexts": [
            "Low priority: I'd like to set up an email signature for my out-of-office persona that includes my manager's contact details. No urgency — whenever you have 5 minutes.",
            "Enhancement request: could IT enable read receipts for the team's shared mailbox? Not urgent — just a nice-to-have for tracking customer response rates.",
            "Informational: I received an external auto-reply from a vendor that looked oddly formatted in Outlook. Just flagging it in case it indicates a compatibility issue. No action needed.",
            "Low priority: I'd like to request that the IT team review our email retention policy. I think the current 2-year deletion policy may be too aggressive. Please add to the next policy review agenda.",
            "No rush: Could someone help me set up an inbox rule to colour-code emails from my top 5 clients? Happy to schedule a 15-minute remote session at any time that suits.",
            "FYI: A former employee's email address is still listed on our website's contact page. Emails to it are bouncing. Please redirect or remove it when convenient.",
        ],
        "p1_actions": [
            "CRITICAL: Check Exchange Online Protection and the email gateway for the root cause of the complete outbound blacklist. Engage Microsoft's emergency support line. Initiate delisting requests with all major blocklist providers simultaneously. Broadcast an all-staff alert about the email outage.",
            "REGULATORY EMERGENCY: Restore journaling immediately. If the primary journaling destination has failed, activate the secondary journaling target. Engage the compliance team and outside counsel to assess the gap period. Notify the Chief Compliance Officer.",
            "CRITICAL DNS FAILURE: Restore the MX records immediately from the DNS backup or the approved zone file. Propagation may take up to 15 minutes. Monitor inbound mail flow continuously until full recovery is confirmed.",
            "P1: Restore the encrypted email gateway service immediately. If a certificate has expired, renew it via the emergency CA process. Test end-to-end encryption with the law firm before closing the incident.",
            "TOTAL EMAIL OUTAGE: Initiate the Exchange emergency recovery runbook. Restore Exchange-to-AD connectivity. Verify all transport services are running. Send status updates to all staff via SMS/Teams every 30 minutes.",
            "INSIDER THREAT / ACCOUNT COMPROMISE: Immediately remove the suspicious mail rule and revoke all active sessions for the affected accounts. Preserve mail flow logs as evidence. Engage the CISO and initiate a full account forensic investigation.",
        ],
        "p2_actions": [
            "Check Exchange message trace and mail flow rules for the sales team. Identify and remove any rules blocking external inbound delivery. Target restoration within 90 minutes and update the sales manager every 30 minutes.",
            "Correct the timezone configuration in Exchange Online. Run a test calendar invite to confirm. Communicate the fix to all staff affected by the scheduling errors and offer to resend impacted invitations.",
            "Restore access to the shared customer support mailbox immediately. Check whether a permissions change or quota limit is responsible. Send a status update to the customer support manager and implement monitoring to prevent recurrence.",
            "Initiate delisting requests with the blocklist operators immediately. Engage Microsoft support if the listing is in Exchange Online Protection. Provide temporary workaround (alternative sending domain or relay) while delisting is in progress.",
            "Audit the distribution list membership and mail flow rules. Remove invalid or stale addresses. Resend the all-hands invitation from the corrected list and confirm receipt with team leads.",
            "Diagnose the TLS/certificate issue between the two mail systems. Renew or re-exchange certificates as needed. Test encrypted delivery before close of business.",
        ],
        "p3_actions": [
            "Check Exchange message trace for the affected sender domain. Add the sender to the safe senders list and remove any incorrectly applied transport rules that are blocking delivery.",
            "Reconfigure the Outlook profile using the automated repair tool. If the issue persists, rebuild the Outlook profile from scratch and test all core functions.",
            "Review the auto-reply configuration in Exchange admin centre. Correct the scope setting from 'all senders' to 'external senders only'. Test with an internal and an external test message.",
            "Update the email signature template in the approved signature management tool. Push the updated signature to the user's Outlook profile and confirm it displays correctly.",
            "Investigate the calendar invite formatting issue in Exchange Online. Check for any active transport rules that may be stripping meeting body content. Apply a fix and test with a new meeting invite.",
            "Check mailbox quota and archive status. If approaching the limit, enable or expand the online archive. If already over limit, work with the user to identify large items for archiving or deletion.",
        ],
        "p4_actions": [
            "Walk the user through configuring the out-of-office signature settings in Outlook. Update the signature template to include the manager's contact details.",
            "Evaluate read receipt support for the shared mailbox. Configure if supported by the current licensing and document the setting for the team.",
            "Test the auto-reply format from the vendor in Outlook and confirm it is a display formatting quirk rather than a system issue. No changes required.",
            "Add the email retention policy review item to the IT governance team's agenda for the next quarterly meeting. Send the user a confirmation.",
            "Schedule a 15-minute remote session with the user to configure their inbox rules. Document the rule configuration for future reference.",
            "Remove or redirect the former employee's email address on the website. Update the internal contact directory as well. Log the change.",
        ],
    },

    # ─── PRINTER ─────────────────────────────────────────────────────────────

    "printer": {
        "subjects": [
            "office network printer", "personal desk printer", "colour laser printer",
            "multifunction copier", "label printer", "print server",
            "scan-to-email function", "wireless printer", "large-format plotter",
            "badge-release printer", "print queue", "document finishing unit",
            "industrial label printer", "receipt printer", "cheque printer",
            "clinical label printer", "wristband printer", "A3 printer",
        ],
        "symptoms": [
            "print jobs stuck in the queue with no error message",
            "printing blank pages consistently",
            "paper jam that clears but immediately recurs",
            "streaky, faded, or banded output across all documents",
            "not appearing in the network printer list on any machine",
            "showing 'offline' in Windows despite being powered on and connected",
            "printing at the wrong scale — everything is 65% of correct size",
            "double-sided printing producing blank reverse sides",
            "scan function producing completely black images",
            "new toner cartridge showing 'low toner' warning immediately",
            "taking 15 minutes per page on a fast machine",
            "driver installation failing with error code 0x00000709",
            "feeding multiple sheets at once causing constant jams",
            "printing in black and white only despite colour being selected",
            "scan-to-email silently failing — no email received",
            "paper curling severely on every print",
            "print jobs disappearing from the queue without printing",
            "printer producing physical page that is readable but heavily pixelated",
        ],
        "details": [
            "I've already tried turning it off and on again.",
            "Multiple people on the floor are affected — it's not just my computer.",
            "The printer display panel shows 'Ready' and no error messages.",
            "I need to print 300 pages for a board meeting in 90 minutes.",
            "This printer was just serviced 2 weeks ago.",
            "I'm on the 3rd floor — the printer is on the 2nd floor but I'm mapped to it.",
            "I can print from my phone successfully but not from my laptop.",
            "The same document prints correctly on a different printer.",
            "We have tried reinstalling the driver on 3 machines with the same result.",
            "A reboot of the print server temporarily resolves the issue but it returns.",
        ],
        "p1_contexts": [
            "CRITICAL: The cheque printer used by our finance team to issue payroll cheques has failed on payroll run day. We must issue approximately 500 physical cheques by 3 PM today or we will be in breach of our payroll obligations to staff.",
            "EMERGENCY: All 8 clinical label printers in the hospital pharmacy have stopped working. Patient medication labels cannot be printed. This is a patient safety issue — medication cannot be dispensed without accurate labels.",
            "URGENT: The sole receipt printer in our busy restaurant location has failed during peak Saturday lunch service. We cannot process card payments or print itemised bills. The queue of waiting customers is growing.",
            "CRITICAL: The wristband printer in the hospital A&E department is offline. New patients cannot be issued with identification wristbands. This is a patient safety and regulatory compliance issue.",
            "P1: The label printers on the warehouse production line have all gone offline simultaneously. We cannot label or ship any products. 3 customer orders that are due for dispatch today will miss their SLA if this is not resolved immediately.",
            "CRITICAL: The only working plotter in the architectural firm has failed during the final stages of a major planning submission. The client deadline for physical drawings is 5 PM today.",
        ],
        "p2_contexts": [
            "All 4 printers on the finance floor are affected and the team needs to print 200-page audit packs for an external auditor arriving tomorrow at 9 AM. We need this resolved today.",
            "The sole colour printer on our marketing floor is down and we have a brand event tomorrow morning that requires 150 colour brochures. We urgently need this fixed or a same-day alternative arranged.",
            "The print server crashed overnight and 60 people cannot print anything this morning. We have a major deliverable review meeting at 11 AM that requires printed copies for 15 participants.",
            "The label printer for our courier dispatch desk is down. We cannot process any outbound parcels. We have 40 shipments queued for today's collection, which arrives in 3 hours.",
            "The badge-release printer on the secure floor is offline. Staff who use print-and-release for confidential documents cannot retrieve their prints. Security compliance is being impacted.",
            "The department's only A3 printer has failed. We have a planning presentation that requires A3 format drawings in 4 hours. Please either fix the printer or identify a same-day alternative urgently.",
        ],
        "p3_contexts": [
            "I'm the only person affected — my colleagues' prints are going through fine. I can use the printer on the next floor as a workaround. Please fix when you have time.",
            "This is intermittent — the issue happens a couple of times a week and a reboot of the printer resolves it. Not critical, but please schedule a proper fix at your convenience.",
            "The printer is usable but output quality has degraded noticeably over the past week. I can live with it for now, but it will need attention before the next big print run.",
            "The scan-to-email function is broken but I can still scan to a USB drive as a workaround. Please address this within the week when a technician is in the area.",
            "One of our three floor printers is having issues. The other two work fine. Not urgent — please schedule during the next maintenance visit.",
            "My printer driver is out of date and the new version is not installing correctly. I have a workaround but would like the driver updated when IT has bandwidth.",
        ],
        "p4_contexts": [
            "Low priority: I'd like to set the default print settings on our floor printer to double-sided to reduce paper usage. No urgency — whenever an engineer has 5 minutes.",
            "Enhancement request: could IT look into enabling the 'secure print' (PIN release) feature on our floor printer? Not urgent — for future consideration.",
            "Informational: the printer display is showing a 'maintenance kit due soon' warning. Not affecting print quality yet but wanted to flag it in advance.",
            "No rush: I'd like to request a colour printer be added to our small satellite office. We currently have to walk to the main floor for colour printing. Please raise with facilities for the next budget cycle.",
            "FYI: The paper tray has a small crack in the guide rail. It still feeds correctly but may be worth replacing at the next service. Just logging for the service record.",
            "Low priority: I'd like to add the printer on the 4th floor to my laptop as a secondary option. Not urgent — happy to drop by the helpdesk for assistance.",
        ],
        "p1_actions": [
            "CRITICAL P1: Dispatch an on-site technician immediately. Bring a replacement unit from the emergency hardware inventory. If unavailable, engage the printer vendor's emergency same-day swap service. Notify the affected team manager of the ETA.",
            "PATIENT SAFETY P1: Escalate immediately to the clinical engineering and IT on-call teams. Deploy replacement printers from the crash cart supply. Notify the pharmacy manager and clinical risk team. Do not close until all printers are operational and output is verified clinically accurate.",
            "EMERGENCY: Deploy the spare receipt printer from inventory immediately. If unavailable, engage the vendor for an emergency same-day loan. Coordinate with the restaurant manager on the interim process — manual billing if required.",
            "CRITICAL SAFETY: Replace the A&E wristband printer with the emergency spare immediately. Notify the ward manager and clinical risk team. Do not close until patient identification continuity is confirmed.",
            "P1 WAREHOUSE: Replace all failing label printers from the emergency spare pool. Engage the vendor's on-site emergency service if spares are unavailable. Prioritise the highest-volume printing station first. Notify the logistics manager of the ETA and any order delays.",
            "CRITICAL DEADLINE: Repair or replace the plotter immediately. If the fault cannot be fixed within 1 hour, arrange emergency access to an external print bureau near the firm and coordinate transport of the digital files.",
        ],
        "p2_actions": [
            "Dispatch a technician to the finance floor immediately. Bring toner and spare parts. If hardware repair is not possible today, arrange a temporary high-volume printer hire for the audit pack print run.",
            "Diagnose and repair the colour printer urgently. If a same-day repair is not possible, arrange access to the nearest colour printer on another floor or procure same-day printing from an approved external print service.",
            "Restart the print server and restore the print queues for all 60 affected users. Test at least one machine per floor before the 11 AM meeting. Bring a portable printer to the meeting room as a contingency.",
            "Replace the dispatch label printer with a spare unit. If no spare is available, manually prepare shipping labels for the most urgent parcels and escalate the hardware repair to a same-day priority service call.",
            "Repair the badge-release printer or activate an emergency alternative secure print release option. Notify staff of the interim process for retrieving confidential documents.",
            "Repair the A3 printer within 2 hours or identify a working A3 printer elsewhere in the building. If no alternative is available, arrange urgent external printing at a bureau and courier delivery of the materials.",
        ],
        "p3_actions": [
            "Clear the print queue, restart the print spooler service via services.msc, and submit a single-page test job. If the issue persists, reinstall the printer driver from the approved repository.",
            "Check toner and drum levels via the printer's web interface. Order replacement consumables if below 15%. Schedule a technician visit to inspect and clean the printing mechanism.",
            "Dispatch a facilities technician to clear the physical paper jam and inspect the feed rollers for wear. Order replacement rollers if worn.",
            "Verify the printer's IP address in the print management console. If it has changed, update the port configuration on all affected machines. Consider assigning a DHCP reservation.",
            "Switch the affected user to the nearest alternative printer as an immediate workaround. Schedule a maintenance call for the faulty printer within 3 business days.",
            "Reinstall the printer driver from the approved driver repository. Test all core print functions — simplex, duplex, A4, and colour — before closing the ticket.",
        ],
        "p4_actions": [
            "Connect to the printer's web interface and update the default duplex setting. Communicate the change to the floor via email. Log the configuration change.",
            "Evaluate the secure print / PIN release capability of the printer model. If supported, enable it via the web interface and communicate the new workflow to the team.",
            "Log the maintenance kit warning in the asset management system. Order the maintenance kit from the approved consumables supplier and schedule installation at the next technician visit.",
            "Log the colour printer request for the satellite office. Route to the facilities and IT procurement teams for budget approval in the next quarter.",
            "Log the paper tray damage in the printer's service record. Order the replacement guide rail component and schedule installation at the next scheduled service.",
            "Assist the user via remote support to add the 4th-floor printer as a secondary device. Update the user's printer profile in the print management system.",
        ],
    },

    # ─── OTHER ────────────────────────────────────────────────────────────────

    "other": {
        "subjects": [
            "conference room AV equipment", "desk phone", "building badge",
            "software license", "training platform access", "IT onboarding",
            "equipment return process", "asset inventory", "IT policy question",
            "ergonomic equipment request", "tech refresh request", "general IT enquiry",
            "IT contract renewal", "office move IT setup", "BYOD policy",
            "disaster recovery plan", "IT knowledge base article", "digital signage system",
            "reception check-in kiosk", "visitor Wi-Fi", "smart TV in boardroom",
        ],
        "symptoms": [
            "projector in the main boardroom won't display anything on screen",
            "desk phone shows 'no service' after moving to a new desk",
            "need a software license for an urgent tool evaluation",
            "building badge not working at the server room entrance",
            "IT onboarding checklist shows as incomplete for a new hire who started today",
            "equipment return process is unclear and the deadline is today",
            "the IT policy page on the intranet is out of date or missing",
            "requesting an ergonomic sit-stand desk and monitor arm",
            "laptop is overdue for tech refresh by 18 months",
            "training platform account is not linked to the correct team",
            "office move IT setup not completed before the team arrives tomorrow",
            "digital signage system showing incorrect content across all screens",
            "visitor Wi-Fi network is not working at the reception desk",
            "smart TV in the boardroom won't connect to the presentation laptop",
            "IT disaster recovery plan document is missing from the shared drive",
            "BYOD policy approval required urgently for a new remote hire",
            "reception kiosk is frozen and visitor sign-in is not working",
            "software licence renewal is overdue and users are being locked out",
        ],
        "details": [
            "My manager told me to submit a ticket for this.",
            "I'm not sure which team handles this type of request.",
            "This isn't urgent but I'd like it resolved this week.",
            "I checked the knowledge base but couldn't find a relevant article.",
            "The self-service portal doesn't have an option for this.",
            "I'm happy to provide more information if needed.",
            "This affects the entire team, not just me.",
            "I've been waiting on this for about 2 weeks with no response.",
            "The request was submitted through the old system before the migration.",
            "My manager has already approved this via email.",
        ],
        "p1_contexts": [
            "CRITICAL: The AV system in the main boardroom has failed 30 minutes before a Board of Directors meeting that includes remote participants. There is no alternative room available. We need immediate on-site AV support.",
            "EMERGENCY: The reception sign-in kiosk has frozen and cannot be rebooted. Our building is in a regulated sector and ALL visitors must be electronically signed in before entering. We are blocking legitimate visitors and failing our compliance obligation.",
            "CRITICAL: Software licence server is down and 150 users of our core engineering tool are locked out. A major client deliverable is due at 5 PM today. Every engineer in the department is completely blocked.",
            "URGENT: The IT setup for a new office floor was not completed before the team of 40 moved in this morning. There are no working phones, no printer, and half the workstations have no network connectivity. The whole team is unable to work.",
            "CRITICAL: The digital signage system in our airport terminal is displaying incorrect departure information. Passengers are going to wrong gates. This is a passenger safety and regulatory compliance issue requiring immediate correction.",
            "P1: The visitor management kiosk at our pharmaceutical facility entrance is down. All visitor access must be logged for GxP compliance. Regulatory inspectors are arriving in 45 minutes and must be processed through the system.",
        ],
        "p2_contexts": [
            "The conference room AV system has been broken for 3 days. We have 5 client-facing meetings booked in that room this week. The IT team has not responded to the previous 2 tickets. Please escalate.",
            "Our team is moving offices tomorrow and IT setup has not been confirmed. 20 workstations, 2 printers, and the phones need to be connected and tested before 8 AM. Please confirm the setup will be completed tonight.",
            "A critical software licence for a time-limited project is expiring tomorrow. Renewal was submitted 3 weeks ago with no response. If not renewed, 8 people will be locked out and the project will be delayed.",
            "A new senior executive joined yesterday and has no phone, no badge, and incomplete system access. Her first week of executive meetings begins today. This is an embarrassing onboarding failure that needs same-day resolution.",
            "Our visitor Wi-Fi has been down for 2 days. We have external consultants on-site all week who need internet access to do their work. This is impacting our ability to deliver the project.",
            "The BYOD approval for a key remote hire working with PII data is still pending after 10 days. They cannot access encrypted company data without it. Please expedite as a compliance risk is accruing.",
        ],
        "p3_contexts": [
            "The AV setup in one smaller meeting room isn't working. We have other rooms available and can work around it. Please repair during the next available technician slot.",
            "I'd like to request a tech refresh for my 4-year-old laptop. It's still functional but getting slow. I have no immediate deadline — please process within the standard tech refresh cycle.",
            "I have a general question about our BYOD policy. I'd like to understand what personal devices are supported. Please point me to the relevant knowledge base article when you have a moment.",
            "I need to return equipment from a project I've just completed. Please advise on the process. No rush — I have the equipment secured at my desk.",
            "I'd like to request a software licence for a new tool I'm evaluating. I have budget pre-approval. The evaluation is due to start in 2 weeks so no immediate urgency.",
            "The knowledge base article about our VPN client is out of date — it still references the old version. Could someone update it? Not urgent — just a documentation hygiene issue.",
        ],
        "p4_contexts": [
            "Low priority: I'd like to add a monitor arm to my desk setup when IT next has a spare. No urgency — just a comfort preference.",
            "Enhancement idea: could the IT helpdesk portal have a 'frequently asked questions' section added? It would reduce simple ticket volume. A suggestion for the next portal refresh cycle.",
            "No rush: I'd like to understand the company's IT policy on using AI assistants on work machines. Just gathering information — no action needed right now.",
            "FYI: The IT intranet page lists a phone number for the helpdesk that is no longer in service. Could this be corrected whenever someone has a moment?",
            "Low priority: I'd like to request a second monitor for my home office setup for when I work remotely. Happy to wait for the next hardware allocation cycle.",
            "Informational: I noticed the signed IT acceptable use policy form in my employee file is the 2020 version. The policy was updated in 2024. Should I re-sign the new version?",
        ],
        "p1_actions": [
            "CRITICAL P1: Dispatch an AV technician to the boardroom immediately. Bring spare HDMI cables, adapters, and a portable display unit as backup. Set up a dial-in phone bridge as a contingency for the remote participants if AV cannot be restored.",
            "EMERGENCY: Send an IT engineer to the reception kiosk immediately. If the kiosk cannot be rebooted in 5 minutes, activate the paper-based visitor log contingency procedure and notify the compliance officer. Escalate kiosk repair to the vendor's emergency line.",
            "CRITICAL: Engage the licence server vendor on their P1 support line. Restore service from the last known-good configuration. Provide emergency individual licence activations to the most critical users while the server is being restored.",
            "P1 OFFICE MOVE FAILURE: Deploy the entire IT desktop support team to the new floor immediately. Prioritise network, phone, and workstation connectivity. Activate the emergency setup protocol. Commit to all 40 stations being operational within 3 hours.",
            "CRITICAL SAFETY: Gain remote access to the digital signage management system immediately. Correct the departure information display and push an emergency refresh to all screens. Notify the operations centre of the display error and the correction.",
            "COMPLIANCE P1: Deploy a tablet or laptop at the entrance immediately as a manual visitor log. Restore the kiosk via remote access. If remote fix is not possible within 15 minutes, dispatch an engineer immediately. Notify the quality and compliance team.",
        ],
        "p2_actions": [
            "Escalate the AV repair to a senior technician. Schedule a same-day repair or source a replacement unit. Provide a confirmed resolution time to the meeting room booker and offer alternative room setups for this week's meetings.",
            "Dispatch the desktop support team to the new office floor tonight to complete setup. Confirm a checklist of all 20 workstations, phones, and printers with the IT coordinator. Send a completion confirmation to the team manager by 7 AM.",
            "Escalate the licence renewal to the procurement and software asset management team immediately. Provide emergency grace activation while renewal is processed. Confirm the renewal will be in place before the licence expires.",
            "Prioritise the executive's onboarding immediately. Assign a dedicated IT resource to complete all provisioning — phone, badge, access — within 2 hours. Notify the executive's EA once everything is confirmed working.",
            "Troubleshoot and restore visitor Wi-Fi connectivity immediately. If the access point is faulty, replace it from inventory. Provide a mobile hotspot to the consultants as an interim solution.",
            "Route the BYOD approval to the security and compliance team as a priority review. Target a 24-hour approval decision. Notify the manager and the new hire of the expected timeline.",
        ],
        "p3_actions": [
            "Log the AV repair request. Schedule a technician visit to the meeting room within the standard 3-business-day SLA. Add it to the next floor maintenance visit.",
            "Log the tech refresh request in the asset management system. Confirm eligibility per the refresh cycle policy. If eligible, queue for provisioning and notify the user of the expected lead time.",
            "Point the user to the BYOD policy article on the IT intranet. If the article is out of date, raise a documentation update request with the IT communications team.",
            "Provide the user with the equipment return instructions and the relevant form from the asset management portal. Arrange a pick-up slot for the equipment collection.",
            "Process the software licence request through the standard procurement approval flow. Confirm the evaluation licence is available and provide the user with activation instructions when ready.",
            "Assign the knowledge base article update to the IT documentation team. Target a 5-business-day turnaround. Notify the reporting user when the article is updated.",
        ],
        "p4_actions": [
            "Log the monitor arm request. Route to the office management team for procurement from the standard catalogue. Notify the user when the item is available for collection.",
            "Log the portal enhancement suggestion in the IT service improvement backlog. Add it to the next portal UX review meeting agenda.",
            "Send the user the link to the IT AI policy page on the intranet. If no policy exists yet, log a request with the IT policy team to create one.",
            "Update the helpdesk phone number on the IT intranet page. Log the change in the content management system.",
            "Log the second monitor request for the next hardware allocation cycle. Route to the IT asset manager for budget and stock approval. Notify the user when allocation is confirmed.",
            "Send the user the current 2024 acceptable use policy and the process for re-signing digitally. Forward the completed form to HR for the employee file.",
        ],
    },
}


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def _validate_templates() -> None:
    """
    Verify all templates have the required keys and minimum entries.
    Called once at generator startup to catch template errors early.
    """
    from data.schema.ticket import Category

    required_keys = {
        "subjects", "symptoms", "details",
        "p1_contexts", "p2_contexts", "p3_contexts", "p4_contexts",
        "p1_actions", "p2_actions", "p3_actions", "p4_actions",
    }
    min_entries = 6

    for cat in Category:
        if cat.value not in TEMPLATES:
            raise ValueError(
                f"Category '{cat.value}' defined in schema but missing from TEMPLATES. "
                f"Add a template entry in data/generator/templates.py."
            )
        template = TEMPLATES[cat.value]
        missing_keys = required_keys - template.keys()
        if missing_keys:
            raise ValueError(
                f"Template '{cat.value}' is missing required keys: {missing_keys}"
            )
        for key in required_keys:
            if len(template[key]) < min_entries:
                raise ValueError(
                    f"Template '{cat.value}' → '{key}' has only {len(template[key])} "
                    f"entries (minimum {min_entries}). Add more entries for variety."
                )
