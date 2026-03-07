"""
data/generator/templates.py
────────────────────────────
Per-category ticket text templates and next-action recommendations.

WHY THIS FILE EXISTS:
    Pure random text (faker.text()) produces gibberish that doesn't
    resemble real ITSM tickets. These templates give each category a
    realistic vocabulary so the synthetic data has learnable patterns.

HOW IT WORKS:
    The generator picks a category, then randomly selects a subject,
    symptom, and detail from that category's template. It combines
    them into a ticket that sounds like something a real user would
    submit to a helpdesk.

DESIGN DECISIONS:
    - Each category has 10+ subjects and 10+ symptoms to create enough
      variety that the model can't just memorise specific combinations.
    - next_actions are separate from the ticket text — they represent
      what a SUPPORT AGENT would do, not what the user wrote.
    - Templates are stored as plain Python dicts (not JSON/YAML files)
      because they're small, they benefit from syntax highlighting in
      your IDE, and they don't need to be edited by non-programmers.
"""

# ─── TYPE ALIAS ──────────────────────────────────────────────────────────────
#
# We define a type alias for readability. Each template is a dict with
# specific keys. Using a TypedDict would give us even stricter checking,
# but for templates that are only consumed by one generator, a simple
# dict type alias is sufficient. Don't over-engineer what doesn't need it.

from typing import TypedDict


class CategoryTemplate(TypedDict):
    """
    Structure for each category's template bank.

    subjects:     Things that can break ("laptop", "VPN client", "printer")
    symptoms:     What the user experiences ("won't turn on", "keeps dropping")
    details:      Extra context the user might add ("tried restarting", "happens daily")
    next_actions: What the support agent should do (used as training labels)
    """
    subjects: list[str]
    symptoms: list[str]
    details: list[str]
    next_actions: list[str]


# ─── TEMPLATES ───────────────────────────────────────────────────────────────
#
# WHY SO MANY ENTRIES PER CATEGORY?
#
# If we only had 3 subjects and 3 symptoms, the generator could only
# produce 3 × 3 = 9 unique combinations per category. With 12 subjects
# and 12 symptoms, we get 144 combinations BEFORE adding details and
# Faker variation. That's enough diversity for 2,000 tickets without
# obvious repetition.
#
# WHY ARE THE next_actions DIFFERENT FROM THE symptoms?
#
# Because symptoms describe what the USER sees ("screen is black") while
# next_actions describe what the AGENT should do ("run hardware diagnostics").
# The model needs to learn this mapping: user-language → agent-language.

TEMPLATES: dict[str, CategoryTemplate] = {

    "hardware": {
        "subjects": [
            "laptop", "desktop workstation", "docking station", "external monitor",
            "keyboard", "mouse", "headset", "webcam", "USB hub", "power supply",
            "SSD drive", "RAM module",
        ],
        "symptoms": [
            "won't turn on", "makes a clicking noise", "overheating and shutting down",
            "screen is flickering", "not detected when plugged in", "battery draining in under an hour",
            "blue screen on startup", "extremely slow after boot", "ports stopped working",
            "fan running at full speed constantly", "display has dead pixels", "touchpad not responding",
        ],
        "details": [
            "This started after the latest Windows update.",
            "I've tried restarting multiple times with no change.",
            "Other users on my floor have the same issue.",
            "The device is about 3 years old.",
            "I need this for a client presentation tomorrow.",
            "Already swapped cables — same problem.",
            "It was working fine yesterday afternoon.",
            "The warranty should still be active.",
        ],
        "next_actions": [
            "Run hardware diagnostics via Dell SupportAssist and escalate to on-site technician if failing.",
            "Schedule replacement hardware from inventory and ship to user's location.",
            "Check warranty status and create RMA ticket with vendor if within coverage.",
            "Dispatch field technician to inspect physical components and cabling.",
            "Provide loaner device immediately and schedule repair during next maintenance window.",
            "Verify BIOS and firmware are current, then run extended memory and disk tests.",
        ],
    },

    "software": {
        "subjects": [
            "Microsoft Teams", "Outlook", "Excel", "SAP", "Salesforce",
            "Adobe Acrobat", "Zoom", "Chrome browser", "company portal app",
            "VPN client", "antivirus software", "internal CRM tool",
        ],
        "symptoms": [
            "crashes immediately on launch", "freezes when opening large files",
            "stuck on the loading screen", "throwing an error on login",
            "not syncing data properly", "running extremely slowly",
            "update failed mid-install", "features are greyed out",
            "keeps asking to re-authenticate", "incompatible with the new OS version",
            "missing after the latest patch", "showing data from the wrong account",
        ],
        "details": [
            "I've cleared the cache and reinstalled but the issue persists.",
            "This started right after the IT pushed an update.",
            "My colleague on the same version has no issues.",
            "I'm on version 16.82 — not sure if that's current.",
            "Error message says: 'Unexpected application error, code 0x80070005.'",
            "This is blocking my end-of-quarter reporting.",
            "I tried running as administrator but same result.",
            "Task Manager shows it using 95% of memory.",
        ],
        "next_actions": [
            "Verify installed version against approved software catalog and push update if outdated.",
            "Clear application cache, repair installation via Settings > Apps, and test again.",
            "Check known issues board for this application version and apply documented workaround.",
            "Collect crash logs from Event Viewer and escalate to application support team.",
            "Uninstall and perform clean reinstall from the software center.",
            "Review Group Policy settings that may be restricting application features.",
        ],
    },

    "network": {
        "subjects": [
            "VPN connection", "Wi-Fi", "wired ethernet", "Microsoft Teams calls",
            "internet access", "network drive mapping", "DNS resolution",
            "proxy settings", "remote desktop connection", "cloud storage sync",
            "video conferencing", "internal website",
        ],
        "symptoms": [
            "drops every few minutes", "extremely slow — pages take 30+ seconds",
            "cannot connect at all", "works on Wi-Fi but not ethernet",
            "only affects certain websites", "disconnects during video calls",
            "shows 'connected' but no internet access", "latency spikes above 500ms",
            "times out when accessing shared drives", "won't authenticate to the network",
            "packet loss causing audio cutting out", "IP address conflict detected",
        ],
        "details": [
            "I'm working from home on a 100 Mbps connection.",
            "This only happens during peak hours (9-11 AM).",
            "Speed test shows 5 Mbps — should be 200 Mbps.",
            "Other devices on the same network work fine.",
            "Started after I changed office locations.",
            "The issue goes away when I disconnect from VPN.",
            "I'm using the company-issued router.",
            "Traceroute shows the bottleneck is at the firewall hop.",
        ],
        "next_actions": [
            "Check VPN server logs for session timeouts and reset user's VPN credentials.",
            "Run network diagnostics and verify DNS settings match corporate standard.",
            "Escalate to network operations team to check firewall rules and bandwidth allocation.",
            "Verify proxy auto-configuration (PAC) file is accessible and correctly configured.",
            "Release and renew DHCP lease, then flush DNS cache on the client machine.",
            "Schedule a call to test connectivity live while monitoring traffic on the network side.",
        ],
    },

    "security": {
        "subjects": [
            "suspicious email", "phishing attempt", "unauthorized login alert",
            "malware detection", "data breach concern", "compromised account",
            "security certificate", "MFA prompt", "firewall alert",
            "encrypted file", "USB device policy", "access log anomaly",
        ],
        "symptoms": [
            "received an email with a suspicious attachment",
            "got a login notification from an unrecognized location",
            "antivirus quarantined multiple files simultaneously",
            "browser redirecting to unfamiliar websites",
            "account locked after multiple failed login attempts",
            "MFA codes are being sent but I didn't request them",
            "security warning popup that won't go away",
            "colleague reported receiving emails from my account I didn't send",
            "sensitive document found in a public shared folder",
            "certificate expired warning on internal tools",
            "unexpected admin privilege escalation in audit logs",
            "ransomware-style message appeared on screen",
        ],
        "details": [
            "I did NOT click on anything — just reporting it.",
            "The email looked like it came from our CEO.",
            "This happened at 3 AM when I was asleep.",
            "I've changed my password as a precaution.",
            "Multiple people in my department got the same email.",
            "The alert came from our endpoint detection tool.",
            "I was traveling internationally when this happened.",
            "I'm not sure if I accidentally clicked a link last week.",
        ],
        "next_actions": [
            "Immediately isolate the affected endpoint and initiate forensic analysis.",
            "Reset all credentials for the compromised account and revoke active sessions.",
            "Forward the suspicious email to the security operations center for analysis.",
            "Run full endpoint scan and check if any data was exfiltrated via DLP logs.",
            "Escalate to the incident response team and classify severity per policy.",
            "Enable enhanced monitoring on the user's account for the next 30 days.",
        ],
    },

    "access": {
        "subjects": [
            "SharePoint site", "shared network drive", "Jira project",
            "Confluence space", "admin panel", "production database",
            "GitHub repository", "cloud AWS console", "HR portal",
            "expense reporting tool", "customer data dashboard", "build server",
        ],
        "symptoms": [
            "getting 'Access Denied' when trying to open",
            "can view but cannot edit documents",
            "access was revoked without notice",
            "need access for a new project assignment",
            "permissions changed after a role transfer",
            "temporary contractor needs access for 30 days",
            "account shows wrong department in the directory",
            "SSO login loops back to the sign-in page",
            "can access from office but not from home",
            "new hire needs day-one access to standard tools",
            "service account permissions expired",
            "manager approval is pending for over a week",
        ],
        "details": [
            "My manager already approved this in the ticketing system.",
            "I was transferred to this team last Monday.",
            "I need read-only access — not full admin.",
            "The project deadline is this Friday.",
            "I had access until the system migration last weekend.",
            "HR has confirmed my role change in Workday.",
            "I'm a contractor — my access should be time-limited.",
            "My colleague with the same role has this access already.",
        ],
        "next_actions": [
            "Verify manager approval in the access management system and provision requested permissions.",
            "Cross-reference the user's role in Active Directory with the access control matrix.",
            "Grant time-limited access (30 days) and set a calendar reminder for review.",
            "Escalate to the resource owner for approval — standard SLA is 24 hours.",
            "Check if recent Group Policy changes affected access and restore if applicable.",
            "Provision standard onboarding access package per the new hire checklist.",
        ],
    },

    "email": {
        "subjects": [
            "Outlook desktop client", "Outlook web app", "email delivery",
            "calendar invites", "distribution list", "shared mailbox",
            "email signature", "auto-reply settings", "email attachment",
            "inbox rules", "spam filter", "email archiving",
        ],
        "symptoms": [
            "not receiving emails from external senders",
            "sent emails are bouncing back with a 550 error",
            "calendar invites show the wrong timezone",
            "shared mailbox not appearing in Outlook",
            "cannot send attachments larger than 10 MB",
            "emails going directly to junk folder",
            "auto-reply not activating despite being configured",
            "signature not displaying in replies",
            "search returns no results for recent emails",
            "synchronization stuck — new emails not loading",
            "distribution list not delivering to all members",
            "email stuck in outbox and won't send",
        ],
        "details": [
            "This has been happening for 3 days now.",
            "The bounce message references our mail server.",
            "I checked my spam/junk folder — nothing there.",
            "Other people can send to the same address successfully.",
            "I'm on Microsoft 365 Business Premium.",
            "My mailbox is at 48 GB out of 50 GB quota.",
            "This works fine on my phone but not on desktop.",
            "IT made some changes to our mail flow rules last week.",
        ],
        "next_actions": [
            "Check Exchange message trace for the affected sender and identify where delivery is failing.",
            "Verify mailbox quota and archive old items if approaching the 50 GB limit.",
            "Review mail flow rules in Exchange admin center for unintended blocks.",
            "Reconfigure the Outlook profile and test with a new mail profile as fallback.",
            "Check DNS records (MX, SPF, DKIM) for the domain to rule out delivery issues.",
            "Add the affected sender to the safe senders list and whitelist in spam filter.",
        ],
    },

    "printer": {
        "subjects": [
            "office network printer", "personal desk printer", "color laser printer",
            "multifunction copier", "label printer", "print server",
            "scan-to-email function", "wireless printer", "plotter",
            "badge-release printer", "print queue", "fax machine",
        ],
        "symptoms": [
            "print jobs stuck in the queue", "printing blank pages",
            "paper jam that won't clear", "streaky or faded output",
            "not appearing in the printer list", "offline even though it's powered on",
            "prints the wrong size — everything is scaled down",
            "double-sided printing not working", "scan function produces black images",
            "toner low warning but just replaced it", "extremely slow — takes 5 minutes per page",
            "driver installation keeps failing",
        ],
        "details": [
            "I've already tried turning it off and on again.",
            "Multiple people on the floor are affected.",
            "The printer shows 'Ready' on its display panel.",
            "I need to print 200 pages for a meeting in an hour.",
            "This printer was just serviced last month.",
            "I'm on the 3rd floor — the printer is on the 2nd floor.",
            "I can print from my phone but not from my laptop.",
            "The same document prints fine on other printers.",
        ],
        "next_actions": [
            "Clear the print queue, restart the print spooler service, and test with a single-page job.",
            "Check toner and drum levels via the printer's web interface and order replacements if needed.",
            "Reinstall the printer driver from the approved driver repository.",
            "Dispatch a facilities technician to clear the physical paper jam and inspect rollers.",
            "Verify the printer's IP address hasn't changed and update the port configuration.",
            "Switch user to the nearest alternative printer and schedule maintenance for this unit.",
        ],
    },

    "other": {
        "subjects": [
            "conference room equipment", "desk phone", "building badge",
            "software license", "training platform access", "IT onboarding",
            "equipment return", "asset inventory", "IT policy question",
            "ergonomic equipment request", "tech refresh cycle", "general IT question",
        ],
        "symptoms": [
            "projector in the conference room won't display",
            "desk phone shows 'no service' after moving desks",
            "need a software license for a new tool evaluation",
            "badge not working at the entrance after hours",
            "IT onboarding checklist incomplete for new hire",
            "need to return equipment after leaving the project",
            "not sure which IT policy applies to my situation",
            "requesting ergonomic keyboard and monitor stand",
            "laptop is due for tech refresh — how do I request it",
            "training platform credentials not working",
            "need help setting up a new meeting room display",
            "general question about approved software list",
        ],
        "details": [
            "My manager said to submit a ticket for this.",
            "I'm not sure which team handles this request.",
            "This isn't urgent but I'd like it resolved this week.",
            "I checked the knowledge base but couldn't find an answer.",
            "The self-service portal doesn't have an option for this.",
            "I'm happy to provide more information if needed.",
            "This is for the entire team, not just me.",
            "I've been waiting on this for about 2 weeks.",
        ],
        "next_actions": [
            "Route to facilities management for physical equipment inspection.",
            "Check license availability in the software asset management tool and procure if needed.",
            "Consult the IT policy knowledge base and provide the user with the relevant article.",
            "Create a follow-up task for the onboarding coordinator to complete remaining items.",
            "Process the equipment return via the asset management portal and update inventory.",
            "Schedule a tech refresh consultation and verify eligibility per the replacement cycle.",
        ],
    },
}


# ─── QUICK VALIDATION ────────────────────────────────────────────────────────
#
# WHY VALIDATE TEMPLATES AT IMPORT TIME?
#
# If someone edits a template and accidentally deletes the "symptoms" key,
# we want to catch that IMMEDIATELY when the module is imported — not
# 10 minutes later when the generator crashes mid-run on ticket #847.
#
# This is called "fail fast" and it's a core engineering principle:
# the earlier you catch a bug, the cheaper it is to fix.

def _validate_templates() -> None:
    """Verify all templates have the required keys and minimum entries."""
    from data.schema.ticket import Category

    required_keys = {"subjects", "symptoms", "details", "next_actions"}
    min_entries = 6  # Minimum per list to ensure enough variety

    for cat in Category:
        if cat.value not in TEMPLATES:
            raise ValueError(
                f"Category '{cat.value}' defined in schema but missing from templates. "
                f"Add a template entry for it in data/generator/templates.py."
            )
        template = TEMPLATES[cat.value]
        missing_keys = required_keys - template.keys()
        if missing_keys:
            raise ValueError(
                f"Template '{cat.value}' is missing keys: {missing_keys}"
            )
        for key in required_keys:
            if len(template[key]) < min_entries:
                raise ValueError(
                    f"Template '{cat.value}' → '{key}' only has {len(template[key])} "
                    f"entries (minimum {min_entries}). Add more for variety."
                )


# NOTE: We don't call _validate_templates() at import time here because
# it imports from data.schema.ticket, which would create a circular import
# if ticket.py ever imports from templates.py. Instead, the generator
# calls it once at startup. This is a pragmatic trade-off: we lose
# immediate validation but avoid a circular dependency.
