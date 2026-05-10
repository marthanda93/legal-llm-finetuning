#!/usr/bin/env python3
"""Stage D — Hard Negatives & Adversarial (comprehensive coverage).

The most common failure mode of a "be helpful" LLM is OVER-EXTRACTION: it
labels procedural noise, civil disputes, and background narrative as crimes.

This stage generates negative-example training data so the model learns when
to output an EMPTY events list. Coverage is built from FOUR independent
sources so every act, every crime category, and every common FIR-shaped
non-crime pattern is touched:

    1. TEMPLATED FAMILIES (~12 families): procedural, civil_dispute,
       medical_background, ambiguous_intent, abandoned_attempt, time_barred,
       clarificatory, insurance_only, precautionary, retraction_or_denial,
       hr_workplace_only, administrative_compliance, traffic_minor_admin,
       family_separation, mistaken_identity, lost_document.

    2. TAXONOMY non_crime entries — every example + positive_indicator from
       CRIME_TAXONOMY.json's `non_crime` group, no caps.

    3. TAXONOMY BOUNDARY negatives — every `negative_indicator` for every
       one of the 227 crime categories, wrapped as "this looks like X but
       does not satisfy the legal ingredients of X" narratives.

    4. DEFINITION EXCLUSION negatives — every `what_it_excludes` snippet
       across the 21 acts in LAW_RAW_DATA/DEFINITIONS, wrapped as
       "expressly outside the meaning of <term> under <act>" narratives.
       Capped per-definition and per-act so no single act dominates.

Output:
    training/datasets/stages_raw/stage_d_hard_negatives.jsonl

Modes:
    Full overwrite (default — covers ALL sources at sane defaults):
        python -m training.build_hard_negatives

    Validate the current file (no writes):
        python -m training.build_hard_negatives --validate

    Resume — keep clean records, drop contaminated/duplicate ones,
    generate new uniques, append. Backs up the original first:
        python -m training.build_hard_negatives --resume

Source toggles (default: ALL on):
    --no-templates          skip templated families
    --no-non-crime          skip taxonomy non_crime entries
    --no-boundary           skip taxonomy boundary (negative_indicators)
    --no-excludes           skip definition what_it_excludes

Coverage caps:
    --scale N                          template families multiplier (default 30)
    --max-neg-indicators-per-category  per-category cap (default 0 = all)
    --max-excludes-per-definition      per-term cap     (default 1)
    --max-excludes-per-act             per-act cap      (default 150)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from training.common import (
    CANONICAL_LAW_NAMES,
    RAW_STAGE_DIR,
    SLIM_EXTRACTION_SYSTEM_PROMPT,
    TrainingRecord,
    canonical_law_name,
    category_to_snake,
    fingerprint,
    load_all_definitions,
    load_crime_taxonomy,
    short,
    squish,
    write_jsonl,
)


GEN_VERSION = 3  # bump when templates / pools / sources change

EMPTY_OUTPUT = {
    "fir_text_categories": [],
    "events": [],
}


# ---------------------------------------------------------------------------
# Templated narratives
#
# Every {placeholder} must be filled by `_fill()`. Templates MUST NOT mention
# any actual criminal act, even in a "for context" framing — the validator
# below will reject any narrative that slips through.
# ---------------------------------------------------------------------------
TEMPLATE_PROCEDURAL = [
    # Pure FIR-registration / paperwork narrative — no crime referenced.
    "I, {name}, {relation} {parent}, resident of {addr}, hereby state that I have come "
    "to lodge this complaint and request the police station to register a formal FIR. "
    "The investigating officer recorded my statement under section 161. List of "
    "witnesses present at the scene is attached. The site of incident was visited by "
    "the IO and a panchnama was drawn. Case property has been duly recorded in the "
    "malkhana register and a copy of the seizure memo provided to me. I request "
    "appropriate action.",

    # Background-update narrative.
    "I, {name}, am providing this background information to the police. I have been "
    "residing at {addr} for the last 25 years. My family consists of my "
    "{family_make_up}, and I am employed at a private firm. I have no prior criminal "
    "record. I am submitting documents in support of my earlier complaint. I have no "
    "specific incident to report today and only wish to update the file with the "
    "documents.",

    # Pure investigation-progress note.
    "Statement of complainant {name} recorded by the Investigating Officer. The "
    "complainant stated that test identification parade has been conducted before the "
    "magistrate. Recovery memo and seizure list have been prepared. Witness "
    "statements under Section 161 BNSS are being recorded. No fresh incident is "
    "alleged. This entry is purely procedural and reflects investigation progress.",

    # New: court-direction compliance, no crime narrative.
    "Pursuant to the order of the Hon'ble Magistrate dated {date}, I, {name}, "
    "complainant in case crime no. {crime_no}, have appeared at the police station "
    "today. My address proof and identity proof have been re-verified by the duty "
    "officer. No new incident is reported. This entry is for compliance with the "
    "earlier court direction only.",

    # New: chargesheet-filing / case status update.
    "This is a status update by the Investigating Officer in connection with case "
    "crime no. {crime_no}. The chargesheet is under preparation and pending review by "
    "the Senior Inspector. Forensic reports have been received and are being "
    "tabulated. No new factual development has occurred. The complainant {name} has "
    "been informed of the progress.",
]

TEMPLATE_CIVIL = [
    # Boundary dispute, civil court only.
    "I, {name}, resident of {addr}, state that there is an ongoing civil dispute "
    "between me and one {other_civil} over the boundary wall of our adjoining "
    "properties. The matter is already pending in the civil court. I want to record "
    "this dispute for police information. There has been no physical assault, no "
    "threat, and no use of force. The dispute is purely about title and boundary "
    "measurement.",

    # Pure non-payment under contract — explicitly disclaiming cheating.
    "I, {name}, state that I had entered into a private commercial contract with "
    "{other_civil} on {date} for supply of goods worth Rs. {amount}. Despite repeated "
    "reminders the payment has not been received. I have already filed a recovery "
    "suit in the competent civil court. I am informing the police only for record. "
    "There is no allegation of cheating or fraud — this is a pure non-payment "
    "dispute under contract.",

    # Tenancy / rent control dispute.
    "I, {name}, am the tenant of a residential premises owned by {other_civil}. The "
    "landlord has issued an eviction notice which I am contesting in the rent control "
    "court. I am submitting this complaint only to keep the police informed of the "
    "dispute. There has been no force, no trespass, no criminal intimidation. The "
    "matter is squarely civil in nature.",

    # Banking / consumer ombudsman matter — uses a real bank name.
    "I, {name}, am a customer of {bank}. I have a service-related grievance about "
    "charges levied on my account, which I have already escalated to the banking "
    "ombudsman and the consumer forum. I am informing the police for information "
    "only. This is a regulatory grievance, not a criminal complaint.",

    # New: partnership-firm dissolution (purely civil under Partnership Act).
    "I, {name}, was a partner in M/s {firm_name} along with {other_civil}. The "
    "partnership has been dissolved by mutual notice on {date}. Settlement of "
    "accounts is under way before the arbitrator nominated under our partnership "
    "deed. I have no grievance of cheating, criminal breach of trust, or "
    "misappropriation — only an accounts settlement is pending.",

    # New: copyright / IP licensing dispute pending in commercial court.
    "I, {name}, am the proprietor of {firm_name}. We have a commercial licensing "
    "dispute with {other_civil} regarding terms of a software-distribution "
    "agreement. The matter is pending before the Commercial Court. There is no "
    "allegation of infringement-with-intent under the Copyright Act — only a "
    "contractual interpretation question.",
]

TEMPLATE_MEDICAL_BACKGROUND = [
    # Routine illness, no third-party involvement.
    "I, {name}, was admitted to {hospital} on {date} for treatment of dengue fever. I "
    "underwent routine blood tests and remained admitted for five days. I have "
    "completely recovered. I am reporting this to the police only as my employer "
    "requires a copy of FIR alongside the medical certificate for leave records. "
    "There is no criminal allegation.",

    # Self-inflicted accidental injury.
    "I, {name}, suffered a minor accidental fall in my own home on {date} resulting "
    "in a fracture of my right wrist. I was treated at a private hospital. There is "
    "no third-party involvement, no negligence by anyone, and no criminal element. "
    "I am filing this only for my insurance claim records.",

    # New: routine surgical recovery.
    "I, {name}, underwent a planned cardiac stent procedure at {hospital} on {date}. "
    "The procedure was uneventful and I am recovering well at home. I have approached "
    "the police station only to obtain a routine endorsement for my insurance "
    "reimbursement. There is no incident of assault or negligence to report.",
]

TEMPLATE_AMBIGUOUS_INTENT = [
    # Heated argument that did NOT escalate. Tests "verbal disagreement != crime".
    "I, {name}, resident of {addr}, state that on {date} I had a heated verbal "
    "argument with my colleague {other_civil} over allocation of office work. We "
    "both raised our voices, but the matter was resolved within ten minutes by our "
    "supervisor. No physical contact occurred, no threat was made, and no abuse "
    "took place. I am recording this only because I am submitting a separate HR "
    "complaint and was asked to inform the police.",

    # Argument over money, refused to pay — but explicitly no demand-with-threat.
    "I, {name}, had a disagreement with {other_civil} on {date} regarding repayment "
    "of a friendly loan of Rs. {amount}. I asked them firmly to pay back, and they "
    "refused. The conversation ended on a sour note but there was no threat, no "
    "force, no intimidation. I am informing the police only as a precaution; a "
    "civil suit will be filed separately.",

    # Driver who drove home angry but caused no incident.
    "I, {name}, state that on {date} after a stressful day, I drove home from {addr} "
    "in an irritated state of mind. I was within the speed limit and did not consume "
    "any alcohol. No accident, no rash driving, and no traffic-rule violation took "
    "place. I am recording this only to clarify the timeline for an HR query.",
]

TEMPLATE_ABANDONED_ATTEMPT = [
    # Considered but did not act.
    "I, {name}, wish to clarify that on {date} I considered confronting "
    "{other_civil} after a heated dispute, but I voluntarily walked away and went "
    "back home. No confrontation, no contact, no threat actually occurred. The "
    "matter ended at the level of thought. I am submitting this clarification "
    "because someone has falsely circulated a rumour against me.",

    # Drafted a letter, did not send.
    "I, {name}, drafted a strongly worded letter against {other_civil} regarding a "
    "neighbourhood dispute, but I deliberately did NOT send it. The letter never "
    "left my house. I am voluntarily disclosing this to put on record that no "
    "communication, no threat, and no defamation was actually published.",
]

TEMPLATE_TIME_BARRED = [
    # Incident is decades old, no ongoing matter.
    "I, {name}, would like to place on record an incident that occurred more than "
    "{years} years ago at my native village. The matter has long since been settled "
    "between the families through a panchayat. There is no fresh dispute, no "
    "continuing harm, and no relief is being claimed today. I am only narrating this "
    "for personal closure as part of my ongoing court matter affidavit.",

    # Withdrawn complaint.
    "I, {name}, had filed a private complaint against {other_civil} in the year "
    "{old_year}, which I subsequently WITHDREW with permission of the court. I have "
    "no surviving grievance against the said party. I am informing the police "
    "station only for record so that no stale matter is wrongly referred to my name.",
]

TEMPLATE_CLARIFICATORY = [
    # Address-correction-only.
    "I, {name}, would like to update my address on record from {addr} to my new "
    "address at {addr2}. My earlier complaint was registered under FIR no. "
    "{crime_no}. There is no new incident or allegation. This is a purely "
    "clarificatory entry for postal correspondence.",

    # Witness availability update.
    "Reference to FIR no. {crime_no} dated {old_year}. I, {name}, am informing "
    "the IO that I will be travelling abroad from {date} for two months. Please "
    "schedule any further statement before that date. There is no new factual "
    "development to report.",
]

TEMPLATE_INSURANCE_ONLY = [
    # Insurance-claim FIR (vehicle damage with no third-party).
    "I, {name}, resident of {addr}, hereby state that on {date} my private "
    "vehicle bearing registration no. {vehicle_reg} sustained minor damage in a "
    "self-skid on a wet road. There was no collision with any other vehicle, no "
    "person was injured, and no traffic-rule violation took place. I am "
    "registering this for the sole purpose of my motor-insurance claim with "
    "{bank_or_insurer}. There is no criminal allegation against anyone.",

    # Mobile lost — needed only for SIM-block / insurance.
    "I, {name}, lost my personal mobile phone today while travelling on the "
    "{date}. The phone may have slipped from my pocket. I am registering this "
    "complaint solely so that my mobile carrier can block the SIM and so that "
    "my mobile-insurance provider can process my claim. There is no allegation "
    "of theft, snatching, or any criminal act by any person.",
]

TEMPLATE_PRECAUTIONARY = [
    # Anticipatory complaint with no actual event yet.
    "I, {name}, resident of {addr}, am submitting this anticipatory complaint "
    "for record. I have observed an unidentified person parked outside my house "
    "on two occasions, but no contact, no threat, and no untoward incident has "
    "actually occurred. I am informing the police only as a precaution so that "
    "any future development can be promptly addressed. No criminal allegation "
    "is being made at this stage.",

    # Threat-perception note (workplace).
    "I, {name}, am providing a precautionary note that I have been receiving a "
    "few cold calls from unknown numbers over the last week. The calls "
    "disconnect without any conversation. There has been no threat, no "
    "intimidation, and no demand of any kind. I am informing the police as a "
    "matter of caution; no criminal complaint is being filed today.",
]

TEMPLATE_RETRACTION = [
    # Denying a circulating false rumour.
    "I, {name}, would like to place on record that a false rumour has been "
    "circulated against me alleging various wrongdoings. I deny all the "
    "allegations in toto. No incident as alleged in the rumour ever took "
    "place. I am submitting this clarificatory complaint to put on record that "
    "I have not committed any offence, and to request the police not to act on "
    "the said unverified rumour.",

    # Voluntary withdrawal of an earlier complaint.
    "Reference to my earlier complaint dated {old_year}. I, {name}, am hereby "
    "VOLUNTARILY withdrawing the said complaint as the matter has been amicably "
    "settled between the parties through mediation. There is no surviving "
    "grievance, and no further action is being sought. The retraction is "
    "without any pressure or inducement.",
]

TEMPLATE_HR_WORKPLACE_ONLY = [
    # HR / disciplinary-only matter, no criminal element.
    "I, {name}, am an employee of {firm_name}. I have a service-related "
    "grievance regarding my last appraisal which I have already escalated to "
    "the company's internal HR committee and the disciplinary authority. I am "
    "informing the police only because my company policy requires me to keep a "
    "record. The matter is purely an HR / contractual dispute and does NOT "
    "amount to harassment or any criminal offence.",

    # Inter-personal office disagreement reported for record only.
    "I, {name}, work in the operations team of {firm_name}. I have raised a "
    "formal grievance with our internal complaints committee regarding tone "
    "and language used in a recent team meeting by my colleague. There was no "
    "physical contact, no overt threat, and no abuse of any sexual nature. The "
    "matter is being handled through internal HR channels and not as a "
    "criminal complaint.",
]

TEMPLATE_ADMINISTRATIVE_COMPLIANCE = [
    # GST / FSSAI / RTO compliance update — purely administrative.
    "I, {name}, proprietor of {firm_name}, am submitting this letter for "
    "record. We have responded to a notice from the {regulator} regarding our "
    "annual return filings. Our authorised representative has appeared, "
    "submitted the documents, and the matter is being processed administratively. "
    "There is no allegation of fraud, evasion, or any criminal offence — only "
    "a procedural compliance matter.",

    # Trade-licence renewal / shop-establishment matter.
    "I, {name}, run a small retail establishment at {addr}. Our trade-licence "
    "renewal application is pending with the local municipal authority. I am "
    "informing the police only because the regulator asked me to keep a "
    "record. There is no criminal allegation against me; this is a routine "
    "administrative compliance matter.",
]

TEMPLATE_TRAFFIC_MINOR_ADMIN = [
    # Minor parking ticket / e-challan, paid.
    "I, {name}, owner of vehicle {vehicle_reg}, received an e-challan dated "
    "{date} for a parking violation, which I have already paid via the official "
    "portal. I am informing the police station only because my employer "
    "requires me to keep a copy on record. There is no pending traffic offence "
    "and no criminal allegation. The matter is fully closed.",

    # Lapsed driving licence renewal, no driving incident.
    "I, {name}, hold a driving licence which lapsed on {date}. I have applied "
    "for renewal at the RTO and the application is under processing. During "
    "the lapsed period I did NOT drive any motor vehicle. No accident, no "
    "rash driving, no traffic-rule violation has occurred. I am informing "
    "the police only for record.",
]

TEMPLATE_FAMILY_SEPARATION = [
    # Marital separation — civil/family court only, not DV.
    "I, {name}, and my spouse have decided to live separately by mutual "
    "consent. We have approached the family court for a mutual-consent divorce "
    "petition. There is no allegation of cruelty, dowry demand, domestic "
    "violence, or any criminal offence by either side. We are informing the "
    "police only as a precaution, since both parties continue to live in the "
    "same locality.",

    # Maintenance / custody pending in family court.
    "I, {name}, am the petitioner in a maintenance application pending before "
    "the family court against my estranged spouse {other_civil}. The matter is "
    "purely civil in nature and is being adjudicated under personal-law "
    "provisions. There is no allegation of any criminal offence; I am only "
    "intimating the police for record.",
]

TEMPLATE_MISTAKEN_IDENTITY = [
    # Wrongly named in someone else's FIR.
    "I, {name}, {relation} {parent}, resident of {addr}, have been informed "
    "that my name has been mentioned by mistake in FIR no. {crime_no} of "
    "P.S. {police_station}. I was NOT present at the place of incident on "
    "the date in question — I was attending an official meeting at "
    "{firm_name}, supported by attendance records. I request that this "
    "clarification be placed on the case diary; no offence has been committed "
    "by me.",

    # Common-name confusion.
    "I, {name}, of {addr}, wish to clarify that my name and address are "
    "different from another individual with a similar name who is the actual "
    "subject of FIR no. {crime_no}. My identity proof is enclosed for "
    "verification. I am NOT the person referred to in that FIR; this entry is "
    "solely to avoid further confusion.",
]

TEMPLATE_LOST_DOCUMENT = [
    # Lost passport / Aadhaar / PAN — administrative only.
    "I, {name}, resident of {addr}, hereby report that I have lost / misplaced "
    "my personal {document_type} on or around {date}. Despite a thorough "
    "search at home and office I have been unable to trace it. I am lodging "
    "this report so that I may apply for a duplicate document and so that no "
    "third party can misuse the same. There is no allegation of theft against "
    "any specific person.",

    # Lost cheque book / bank passbook.
    "I, {name}, customer of {bank}, am reporting that I have misplaced my "
    "cheque book / passbook of account ending {acct_last4}. I have already "
    "instructed the bank to put a stop on all unused cheque numbers. I am "
    "informing the police solely so that I have a record for the bank's "
    "indemnity formalities. There is no criminal allegation.",
]


# ---------------------------------------------------------------------------
# Value pools
#
# Substantially expanded so even at large --scale the random fills produce
# diverse (not duplicate) narratives.
# ---------------------------------------------------------------------------
NAMES_MALE = [
    "Ravi Kumar", "Anand Verma", "Rakesh Singh", "Vikas Patel",
    "Suresh Iyer", "Manoj Tiwari", "Arvind Choudhary", "Deepak Jain",
    "Sanjay Pillai", "Pradeep Yadav", "Naveen Bhatt", "Mohan Rao",
    "Imran Sheikh", "Joseph D'Souza", "Harpreet Singh", "Tarun Mehta",
]
NAMES_FEMALE = [
    "Priya Sharma", "Sunita Devi", "Meera Reddy", "Kavita Nair",
    "Anjali Banerjee", "Pooja Kapoor", "Lata Krishnan", "Asha Joseph",
    "Farzana Ansari", "Roshni Pillai", "Nidhi Aggarwal", "Suman Saxena",
]
RELATION_MALE = ["son of", "S/o"]
RELATION_FEMALE = ["daughter of", "D/o", "wife of", "W/o"]

PARENTS = [
    "Late Sh. Ramesh Kumar", "Sh. Suresh Sharma", "Sh. Mohan Lal",
    "Late Sh. Hari Prasad", "Sh. Dinesh Chand", "Sh. Bhagwan Das",
    "Late Sh. Govind Rao", "Sh. Ashok Bhatia", "Sh. Pradeep Khanna",
    "Sh. Rakesh Bansal",
]
ADDRS = [
    "House No. 45, Sector 22, Chandigarh",
    "Flat 12-B, Andheri West, Mumbai",
    "Plot 78, Jubilee Hills, Hyderabad",
    "Lane 4, Sarvodaya Nagar, Lucknow",
    "MG Road, Bengaluru",
    "B-204, Vasant Kunj, New Delhi",
    "12 Park Street, Kolkata",
    "Flat 7, Shivaji Nagar, Pune",
    "27 Velachery Main Road, Chennai",
    "House 33, Civil Lines, Jaipur",
    "Quarter No. 11, NIT Faridabad",
    "Sector 21-A, Panchkula",
    "Door No. 8/2, Banjara Hills, Hyderabad",
    "Plot 9, Salt Lake Sector V, Kolkata",
    "House 16, Frazer Town, Bengaluru",
]
OTHER_CIVIL = [
    "Mr. Pankaj Gupta", "Mrs. Sushila Bansal", "M/s Bharat Traders",
    "Dr. Anil Khanna", "Mr. Rajeev Mehta", "Mrs. Radha Iyer",
    "Mr. Faisal Sayyed", "Ms. Anita D'Souza", "M/s Krishna Enterprises",
    "Mr. Vinod Tiwari", "Mr. Hardeep Bedi", "Mrs. Latha Pillai",
    "M/s Shree Ganesh Hardware", "M/s Apex Logistics Pvt Ltd",
    "Dr. Snehal Joshi", "Mr. Chetan Bhatia",
]
DATES = [
    "12th February 2026", "5th March 2026", "20th January 2026",
    "8th April 2026", "16th December 2025", "3rd June 2025",
    "27th October 2024", "14th August 2025", "9th November 2025",
    "22nd July 2024", "1st February 2026", "18th September 2024",
]
AMOUNTS = [
    "1,25,000", "85,000", "3,40,000", "55,000", "2,15,000",
    "12,500", "47,800", "4,80,000", "1,05,000", "9,75,000",
    "32,000", "78,500",
]
BANKS = [
    "HDFC Bank", "State Bank of India", "Punjab National Bank",
    "ICICI Bank", "Axis Bank", "Bank of Baroda", "Canara Bank",
    "Kotak Mahindra Bank", "Indian Overseas Bank", "Union Bank of India",
]
HOSPITALS = [
    "Apollo Hospital, Bengaluru", "AIIMS, New Delhi",
    "Lilavati Hospital, Mumbai", "Fortis Hospital, Mohali",
    "Max Super Speciality Hospital, Saket",
    "Manipal Hospital, Whitefield", "Care Hospital, Hyderabad",
    "Christian Medical College, Vellore",
    "Tata Memorial Hospital, Mumbai", "Ruby Hall Clinic, Pune",
    "PGIMER, Chandigarh", "KIMS Hospital, Kochi",
]
FIRM_NAMES = [
    "Maple & Sons", "Krishna Enterprises", "Aroma Foods Pvt Ltd",
    "Sunrise Constructions", "Bharat Logistics", "Saraswati Traders",
    "Tech Vision Software LLP", "Green Earth Recyclers",
    "Pinnacle Realty Pvt Ltd",
]
FAMILY_MAKEUPS = [
    "wife, two children and elderly parents",
    "husband, one daughter, and a younger sister",
    "spouse, three school-going children, and an elderly mother",
    "wife and an unmarried adult son",
    "husband and a teenage daughter",
    "wife, two minor children, and my widowed mother-in-law",
]
CRIME_NOS = [
    "0234/2024", "0119/2025", "0561/2024", "0080/2026",
    "0432/2023", "0007/2026", "0298/2025", "0144/2024",
]
OLD_YEARS = ["2014", "2017", "2018", "2019", "2020", "2021"]
YEARS_AGO = ["12", "15", "18", "20", "22", "25"]
VEHICLE_REGS = [
    "DL-08-AB-1234", "MH-12-CD-5678", "KA-03-EF-9012", "TN-09-GH-3456",
    "UP-32-JK-7890", "AP-09-LM-2345", "GJ-01-NP-6789", "WB-26-QR-0123",
    "RJ-14-ST-4567", "PB-08-UV-8901",
]
BANKS_OR_INSURERS = [
    "HDFC ERGO General Insurance", "ICICI Lombard", "Bajaj Allianz",
    "TATA AIG General Insurance", "New India Assurance",
    "United India Insurance", "Star Health", "Reliance General Insurance",
]
REGULATORS = [
    "GSTN", "FSSAI", "RBI", "RoC", "SEBI", "MCA", "RTO",
    "State Pollution Control Board", "Drug Controller General of India",
]
POLICE_STATIONS = [
    "Sector 17", "Andheri East", "Banjara Hills", "Connaught Place",
    "Indiranagar", "Park Street", "Vasant Vihar", "Adyar",
    "Hazratganj", "Civil Lines",
]
DOCUMENT_TYPES = [
    "passport", "Aadhaar card", "PAN card", "driving licence",
    "voter ID", "ration card", "graduation degree",
    "service identity card",
]
ACCT_LAST4 = ["1234", "5678", "9012", "3456", "7890", "0246", "1357", "8642"]


def _is_female_name(name: str) -> bool:
    return name in NAMES_FEMALE


def _fill(template: str, rng: random.Random) -> str:
    """Plug realistic Indian context into placeholders."""
    name_pool = NAMES_MALE + NAMES_FEMALE
    name = rng.choice(name_pool)
    relation = (
        rng.choice(RELATION_FEMALE) if _is_female_name(name)
        else rng.choice(RELATION_MALE)
    )
    addr = rng.choice(ADDRS)
    addr2 = rng.choice([a for a in ADDRS if a != addr])

    return template.format(
        name=name,
        relation=relation,
        parent=rng.choice(PARENTS),
        addr=addr,
        addr2=addr2,
        other_civil=rng.choice(OTHER_CIVIL),
        date=rng.choice(DATES),
        amount=rng.choice(AMOUNTS),
        bank=rng.choice(BANKS),
        hospital=rng.choice(HOSPITALS),
        firm_name=rng.choice(FIRM_NAMES),
        family_make_up=rng.choice(FAMILY_MAKEUPS),
        crime_no=rng.choice(CRIME_NOS),
        old_year=rng.choice(OLD_YEARS),
        years=rng.choice(YEARS_AGO),
        vehicle_reg=rng.choice(VEHICLE_REGS),
        bank_or_insurer=rng.choice(BANKS_OR_INSURERS),
        regulator=rng.choice(REGULATORS),
        police_station=rng.choice(POLICE_STATIONS),
        document_type=rng.choice(DOCUMENT_TYPES),
        acct_last4=rng.choice(ACCT_LAST4),
    )


# ---------------------------------------------------------------------------
# Contamination validator
#
# The whole point of a hard negative is to teach "do NOT extract events".
# That signal collapses the moment a crime keyword appears in the narrative
# (unless it is paired with an explicit negation). This validator catches
# templates / pool-fills that accidentally mention a crime.
# ---------------------------------------------------------------------------
_CRIME_KEYWORDS = [
    r"\bstolen\b", r"\bstole\b", r"\bsteal\b", r"\btheft\b",
    r"\bsnatched?\b", r"\brobbed?\b", r"\brobbery\b", r"\bdacoity\b",
    r"\bassault(?:ed|ing|s)?\b", r"\battack(?:ed|ing|s)?\b",
    r"\bmurder(?:ed|ing|s)?\b", r"\bkilled?\b", r"\bkilling\b",
    r"\brape[d]?\b", r"\bgang[- ]?rape\b", r"\bsexual\s+assault\b",
    r"\bkidnap(?:ped|ping|s)?\b", r"\babduct(?:ed|ion|ing|s)?\b",
    r"\bcheat(?:ed|ing|s)?\b", r"\bcheating\b",
    r"\bextort(?:ed|ion|ing|s)?\b", r"\bbribed?\b", r"\bbribery\b",
    r"\bharass(?:ed|ing|ment|s)?\b",
    r"\bbeaten?\b", r"\bbeat\b", r"\bhit\b", r"\bstabbed?\b",
    r"\barson\b", r"\bset\s+fire\b", r"\bset\s+ablaze\b",
    r"\btrespass(?:ed|ing|s)?\b", r"\bbreak[- ]?in\b",
    r"\bforged?\b", r"\bforgery\b",
    r"\bdrugs?\b", r"\bnarcotics?\b", r"\bcontraband\b",
    r"\billegal\s+arms?\b", r"\billegal\s+weapon\b",
]
_NEGATION_NEAR = [
    r"no\s+\w*\s*(?:assault|attack|kidnap|theft|robbery|cheat|fraud|extort|harass|threat|abuse|trespass|force|allegation|offence)",
    r"without\s+(?:any\s+)?(?:assault|attack|theft|robbery|cheat|fraud|extort|harass|threat|abuse|force|trespass|offence)",
    r"there\s+is\s+no\s+(?:specific\s+)?criminal\s+(?:allegation|element|complaint)",
    r"this\s+is\s+a\s+pure\s+non[- ]payment",
    r"matter\s+is\s+squarely\s+civil",
    r"is\s+a\s+regulatory\s+grievance",
    r"no\s+criminal\s+(?:allegation|element|complaint)",
    r"no\s+third[- ]party\s+involvement",
    r"no\s+threat\s+was\s+made",
    r"no\s+confrontation",
    r"no\s+contact",
    r"no\s+communication",
    r"no\s+(?:fresh|new)\s+incident",
    r"no\s+physical\s+contact",
    r"no\s+force\s+was\s+used",
    r"no\s+rash\s+driving",
    r"no\s+traffic[- ]rule\s+violation",
    r"no\s+accident",
    r"no\s+infringement[- ]with[- ]intent",
    r"no\s+allegation\s+of",
    r"is\s+not\s+a\s+criminal",
    r"deliberately\s+did\s+not",
    r"voluntarily\s+walked\s+away",
    r"matter\s+ended\s+at\s+the\s+level\s+of\s+thought",
    # Patterns used by boundary / definition-exclusion narratives.
    r"does\s+NOT\s+amount\s+to",
    r"does\s+not\s+amount\s+to",
    r"does\s+not\s+constitute",
    r"does\s+not\s+by\s+itself\s+constitute",
    r"expressly\s+OUTSIDE\s+the\s+meaning",
    r"expressly\s+outside\s+the\s+meaning",
    r"outside\s+the\s+scope\s+of",
    r"legal\s+(?:essentials|ingredients)\s+of\s+that\s+offence\s+are\s+not\s+satisfied",
    r"essentials\s+of\s+the\s+offence\s+are\s+not\s+satisfied",
    r"no\s+offence\s+is\s+therefore\s+made\s+out",
    r"no\s+offence\s+is\s+made\s+out",
    r"is\s+not\s+the\s+person\s+referred\s+to",
    r"falsely\s+circulated\s+a\s+rumour",
    r"deny\s+all\s+the\s+allegations",
    r"voluntarily\s+withdrawing",
]


# Act titles must be masked out before scanning, because several of them
# legitimately contain otherwise-suspicious words (e.g. "Narcotic Drugs and
# Psychotropic Substances Act", "Sexual Harassment of Women at Workplace
# Act"). The masking does NOT change the training data — it only affects
# the contamination scanner's view of the text.
_ACT_TITLE_RE = re.compile(
    "|".join(
        re.escape(n)
        for n in CANONICAL_LAW_NAMES.values()
        if n and n != "Indian Law"
    ),
    re.IGNORECASE,
)

# Strong, legally-formatted disclaimers. If any of these appears anywhere in
# the narrative we treat the entire narrative as an unambiguous "this is not
# a crime" statement — boundary / definition-exclusion narratives rely on
# this so the keyword inside a quoted exclusion-phrase isn't flagged.
_STRONG_DOCLEVEL_DISCLAIMERS = [
    r"expressly\s+OUTSIDE\s+the\s+meaning",
    r"expressly\s+outside\s+the\s+meaning",
    r"does\s+NOT\s+amount\s+to",
    r"does\s+not\s+amount\s+to",
    r"does\s+not\s+constitute",
    r"no\s+offence\s+is\s+therefore\s+made\s+out",
    r"no\s+offence\s+is\s+made\s+out",
    r"legal\s+(?:essentials|ingredients)\s+of\s+that\s+offence\s+are\s+not\s+satisfied",
]


def _mask_act_titles(text: str) -> str:
    return _ACT_TITLE_RE.sub("[ACT]", text)


def _has_doclevel_disclaimer(text: str) -> bool:
    return any(
        re.search(p, text, flags=re.IGNORECASE)
        for p in _STRONG_DOCLEVEL_DISCLAIMERS
    )


def find_contamination(text: str) -> List[Tuple[str, str]]:
    """Return list of (matched_keyword, surrounding_context) where a crime
    mention appears WITHOUT a clear negation. Empty list means clean.

    Algorithm:
      1. Strip out canonical act titles (CANONICAL_LAW_NAMES) so the scanner
         doesn't trip on words inside an Act name.
      2. If the narrative has a STRONG document-level disclaimer ("expressly
         OUTSIDE the meaning", "does not amount to", etc.), trust it.
      3. Otherwise, for every crime keyword found, accept iff the same
         60-char window OR the same sentence carries a negation pattern."""
    if not text:
        return []
    masked = _mask_act_titles(text)
    if _has_doclevel_disclaimer(masked):
        return []

    hits: List[Tuple[str, str]] = []
    for pat in _CRIME_KEYWORDS:
        for m in re.finditer(pat, masked, flags=re.IGNORECASE):
            window = masked[max(0, m.start() - 80): m.end() + 40]
            if any(re.search(n, window, flags=re.IGNORECASE) for n in _NEGATION_NEAR):
                continue
            sent_start = max(masked.rfind(".", 0, m.start()), 0)
            sent_end = masked.find(".", m.end())
            if sent_end < 0:
                sent_end = len(masked)
            sentence = masked[sent_start:sent_end]
            if any(re.search(n, sentence, flags=re.IGNORECASE) for n in _NEGATION_NEAR):
                continue
            hits.append((m.group(0), masked[max(0, m.start() - 30): m.end() + 30]))
    return hits


def is_clean_negative(text: str) -> bool:
    return not find_contamination(text)


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------
def _make_record(narrative: str, sub_kind: str) -> TrainingRecord:
    user_msg = f"FIR:\n{narrative}"
    assistant_msg = json.dumps(EMPTY_OUTPUT, ensure_ascii=False, indent=2)
    return TrainingRecord(
        stage="D",
        task="hard_negative_empty_extraction",
        messages=[
            {"role": "system", "content": SLIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        meta={"sub_kind": sub_kind, "gen_version": GEN_VERSION},
    )


def _from_taxonomy_non_crime(rng: random.Random) -> List[TrainingRecord]:
    """Build hard negatives from CRIME_TAXONOMY.json's non_crime group.

    Every example + positive_indicator is wrapped inside an FIR shell. The
    contamination validator then drops snippets that themselves mention a
    crime (e.g. 'Stolen mobile phone recovered…' which would otherwise
    poison the negative)."""
    out: List[TrainingRecord] = []
    seen: set[str] = set()
    name_pool = NAMES_MALE + NAMES_FEMALE
    for entry in load_crime_taxonomy():
        if entry.get("_group") != "non_crime":
            continue
        examples = [squish(e) for e in (entry.get("examples") or []) if e]
        positives = [squish(e) for e in (entry.get("positive_indicators") or []) if e]
        cat = entry.get("category", "")
        for snippet in examples + positives:
            if len(snippet) < 30:
                continue
            fp = fingerprint(snippet)
            if fp in seen:
                continue
            seen.add(fp)
            shell = (
                f"I, {rng.choice(name_pool)}, "
                f"resident of {rng.choice(ADDRS)}, "
                f"state as follows: {snippet} I am reporting this for record. "
                f"There is no specific criminal allegation in this matter."
            )
            if not is_clean_negative(shell):
                continue
            out.append(_make_record(shell, sub_kind=f"taxonomy_non_crime::{cat}"))
    return out


def _category_human_label(category_snake: str, fallback: str) -> str:
    """'attempt_to_murder' -> 'Attempt to Murder'."""
    if not category_snake:
        return fallback or ""
    return category_snake.replace("_", " ").title()


def _from_taxonomy_boundary(
    rng: random.Random,
    max_per_category: int = 0,
) -> List[TrainingRecord]:
    """For each crime category in the taxonomy, wrap every `negative_indicator`
    as a hard-negative narrative.

    `negative_indicator` strings are short legal descriptions of "what is NOT
    this offence" (e.g. for `murder`: "death caused without intent or
    knowledge of likely death"). We stitch each one into a full FIR-shaped
    paragraph that explicitly disclaims the offence — teaching the model
    the exact boundary between fact-pattern and offence.

    `max_per_category=0` means no cap; otherwise take the first N indicators
    per category."""
    out: List[TrainingRecord] = []
    seen: set[str] = set()
    name_pool = NAMES_MALE + NAMES_FEMALE
    for entry in load_crime_taxonomy():
        if entry.get("_group") == "non_crime":
            continue
        cat_label = (entry.get("category") or "").strip()
        if not cat_label:
            continue
        cat_snake = category_to_snake(cat_label)
        nis = [squish(s) for s in (entry.get("negative_indicators") or []) if s]
        if not nis:
            continue
        if max_per_category > 0:
            nis = nis[:max_per_category]
        human_label = _category_human_label(cat_snake, fallback=cat_label)
        for indicator in nis:
            if len(indicator) < 12:
                continue
            fp = fingerprint(cat_snake, indicator)
            if fp in seen:
                continue
            seen.add(fp)
            name = rng.choice(name_pool)
            relation = (
                rng.choice(RELATION_FEMALE) if _is_female_name(name)
                else rng.choice(RELATION_MALE)
            )
            narrative = (
                f"I, {name}, {relation} {rng.choice(PARENTS)}, resident of "
                f"{rng.choice(ADDRS)}, state the following fact pattern for the "
                f"record. On {rng.choice(DATES)}, the situation was as follows: "
                f"{indicator}. I want to make clear that this set of facts does "
                f"NOT amount to '{human_label}' under Indian criminal law, "
                f"because the legal essentials of that offence are not "
                f"satisfied here. There is therefore no criminal allegation "
                f"arising from these facts; I am only placing them on record "
                f"for clarity."
            )
            if not is_clean_negative(narrative):
                continue
            out.append(_make_record(
                narrative,
                sub_kind=f"taxonomy_boundary::{cat_snake}",
            ))
    return out


def _term_human_short(term: str) -> str:
    return short(term, 80) if term else ""


def _from_definition_exclusions(
    rng: random.Random,
    max_per_definition: int = 1,
    max_per_act: int = 150,
) -> List[TrainingRecord]:
    """For every definition in LAW_RAW_DATA/DEFINITIONS/*.json, wrap
    `what_it_excludes` snippets as boundary hard-negatives.

    Each `what_it_excludes` entry is a legally-vetted "this fact pattern
    looks like the defined term but is expressly outside its meaning" line.
    We stitch it into an FIR-shaped paragraph and label the assistant output
    as an empty events list.

    Caps:
        max_per_definition  — at most N excludes per single (act, term)
        max_per_act         — at most N TOTAL excludes per act (so BNS / MV /
                              LABOUR don't dwarf the smaller acts)"""
    out: List[TrainingRecord] = []
    seen: set[str] = set()
    name_pool = NAMES_MALE + NAMES_FEMALE
    per_act_count: Dict[str, int] = {}

    entries = load_all_definitions()
    rng.shuffle(entries)

    for entry in entries:
        law_id = entry.get("_law_id", "") or ""
        if max_per_act > 0 and per_act_count.get(law_id, 0) >= max_per_act:
            continue
        excludes = [squish(s) for s in (entry.get("what_it_excludes") or []) if s]
        if not excludes:
            continue
        if max_per_definition > 0:
            excludes = excludes[:max_per_definition]
        term = (entry.get("term") or "").strip()
        if not term:
            continue
        section_id = (entry.get("section_id") or "").strip()
        law_name = canonical_law_name(law_id, fallback=entry.get("_law_name", ""))
        section_phrase = f", Section {section_id}" if section_id else ""
        for snippet in excludes:
            if len(snippet) < 20:
                continue
            fp = fingerprint(law_id, term, snippet)
            if fp in seen:
                continue
            seen.add(fp)
            name = rng.choice(name_pool)
            relation = (
                rng.choice(RELATION_FEMALE) if _is_female_name(name)
                else rng.choice(RELATION_MALE)
            )
            narrative = (
                f"Reference to the {law_name}{section_phrase}. I, {name}, "
                f"{relation} {rng.choice(PARENTS)}, resident of "
                f"{rng.choice(ADDRS)}, am placing the following fact pattern "
                f"on record: {short(snippet, 600)} The said situation is "
                f"expressly OUTSIDE the meaning of '{_term_human_short(term)}' "
                f"as defined under the {law_name}. No offence is therefore "
                f"made out, and no criminal allegation is being raised by me "
                f"on this score."
            )
            if not is_clean_negative(narrative):
                continue
            out.append(_make_record(
                narrative,
                sub_kind=f"definition_exclusion::{law_id}",
            ))
            per_act_count[law_id] = per_act_count.get(law_id, 0) + 1
            if max_per_act > 0 and per_act_count[law_id] >= max_per_act:
                break
    return out


_TEMPLATE_FAMILIES: List[Tuple[str, List[str]]] = [
    ("procedural", TEMPLATE_PROCEDURAL),
    ("civil_dispute", TEMPLATE_CIVIL),
    ("medical_background", TEMPLATE_MEDICAL_BACKGROUND),
    ("ambiguous_intent", TEMPLATE_AMBIGUOUS_INTENT),
    ("abandoned_attempt", TEMPLATE_ABANDONED_ATTEMPT),
    ("time_barred", TEMPLATE_TIME_BARRED),
    ("clarificatory", TEMPLATE_CLARIFICATORY),
    ("insurance_only", TEMPLATE_INSURANCE_ONLY),
    ("precautionary", TEMPLATE_PRECAUTIONARY),
    ("retraction_or_denial", TEMPLATE_RETRACTION),
    ("hr_workplace_only", TEMPLATE_HR_WORKPLACE_ONLY),
    ("administrative_compliance", TEMPLATE_ADMINISTRATIVE_COMPLIANCE),
    ("traffic_minor_admin", TEMPLATE_TRAFFIC_MINOR_ADMIN),
    ("family_separation", TEMPLATE_FAMILY_SEPARATION),
    ("mistaken_identity", TEMPLATE_MISTAKEN_IDENTITY),
    ("lost_document", TEMPLATE_LOST_DOCUMENT),
]


def build_records(
    seed: int = 31,
    scale: int = 30,
    include_templates: bool = True,
    include_non_crime: bool = True,
    include_boundary: bool = True,
    include_excludes: bool = True,
    max_neg_indicators_per_category: int = 0,
    max_excludes_per_definition: int = 1,
    max_excludes_per_act: int = 150,
) -> Tuple[List[TrainingRecord], Dict[str, int]]:
    """Produce hard-negative training records from all four sources.

    Returns (records, per_source_counts)."""
    rng = random.Random(seed)
    records: List[TrainingRecord] = []
    counts: Dict[str, int] = {
        "templates": 0,
        "non_crime": 0,
        "boundary": 0,
        "excludes": 0,
    }

    if include_templates:
        before = len(records)
        for kind, templates in _TEMPLATE_FAMILIES:
            for _ in range(scale):
                for tpl in templates:
                    narrative = _fill(tpl, rng)
                    if not is_clean_negative(narrative):
                        continue
                    records.append(_make_record(narrative, sub_kind=kind))
        counts["templates"] = len(records) - before

    if include_non_crime:
        before = len(records)
        records.extend(_from_taxonomy_non_crime(rng))
        counts["non_crime"] = len(records) - before

    if include_boundary:
        before = len(records)
        records.extend(_from_taxonomy_boundary(
            rng,
            max_per_category=max_neg_indicators_per_category,
        ))
        counts["boundary"] = len(records) - before

    if include_excludes:
        before = len(records)
        records.extend(_from_definition_exclusions(
            rng,
            max_per_definition=max_excludes_per_definition,
            max_per_act=max_excludes_per_act,
        ))
        counts["excludes"] = len(records) - before

    return records, counts


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
def _validate_record(rec: Dict[str, Any]) -> Optional[str]:
    """Return None if the record is well-formed and clean, else an error
    string describing the first issue found."""
    if rec.get("stage") != "D":
        return f"stage != 'D' (got {rec.get('stage')!r})"
    if rec.get("task") != "hard_negative_empty_extraction":
        return f"task != 'hard_negative_empty_extraction' (got {rec.get('task')!r})"
    msgs = rec.get("messages") or []
    if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        return f"messages roles wrong: {[m.get('role') for m in msgs]}"
    asst = msgs[-1]["content"]
    try:
        parsed = json.loads(asst)
    except Exception as exc:  # noqa: BLE001
        return f"assistant not JSON: {exc}"
    if parsed.get("fir_text_categories") != [] or parsed.get("events") != []:
        return f"assistant not empty negative: {parsed}"
    user_text = msgs[1]["content"] or ""
    hits = find_contamination(user_text)
    if hits:
        return f"contamination: {hits[0][0]!r} in '...{hits[0][1]}...'"
    return None


def validate_file(path: Path) -> Dict[str, Any]:
    """Return a report dict for the given JSONL file."""
    if not path.exists():
        return {"path": str(path), "exists": False}

    total = 0
    bad: List[Tuple[int, str, str]] = []
    seen_users: Dict[str, int] = {}
    duplicates: List[Tuple[int, int]] = []
    sub_kind_counts: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                bad.append((line_no, "invalid_json", str(exc)))
                continue
            err = _validate_record(rec)
            if err:
                bad.append((line_no, "record_error", err))
                continue
            user_text = rec["messages"][1]["content"]
            if user_text in seen_users:
                duplicates.append((line_no, seen_users[user_text]))
            else:
                seen_users[user_text] = line_no
            sk = rec.get("meta", {}).get("sub_kind", "?")
            sub_kind_counts[sk] = sub_kind_counts.get(sk, 0) + 1

    return {
        "path": str(path),
        "exists": True,
        "total": total,
        "clean": total - len(bad) - len(duplicates),
        "bad": bad,
        "duplicates": duplicates,
        "sub_kind_counts": sub_kind_counts,
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"[validate] file:   {report['path']}")
    if not report.get("exists"):
        print("[validate] file does not exist")
        return
    print(f"[validate] total:  {report['total']}")
    print(f"[validate] clean:  {report['clean']}")
    print(f"[validate] bad:    {len(report['bad'])}")
    print(f"[validate] dups:   {len(report['duplicates'])}")
    if report["bad"]:
        print("[validate] bad records:")
        for line_no, kind, detail in report["bad"][:30]:
            print(f"  line {line_no:>4d}  [{kind}]  {detail}")
        if len(report["bad"]) > 30:
            print(f"  ... and {len(report['bad']) - 30} more")
    if report["duplicates"]:
        print("[validate] duplicates (line -> first occurrence):")
        for dup_line, first in report["duplicates"][:30]:
            print(f"  line {dup_line:>4d}  -> first at line {first}")
    print("[validate] sub_kind distribution:")
    for sk, n in sorted(report["sub_kind_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {sk:50s} {n}")


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------
def _read_existing(path: Path) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str]]]:
    """Read the existing JSONL. Return (kept_records, dropped_records).

    A record is `dropped` if it fails `_validate_record` OR is an exact
    user-content duplicate of an earlier record. Otherwise it is `kept`."""
    kept: List[Dict[str, Any]] = []
    dropped: List[Tuple[int, str]] = []
    seen_users: set[str] = set()
    if not path.exists():
        return kept, dropped
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                dropped.append((line_no, f"invalid_json: {exc}"))
                continue
            err = _validate_record(rec)
            if err:
                dropped.append((line_no, err))
                continue
            user_text = rec["messages"][1]["content"]
            if user_text in seen_users:
                dropped.append((line_no, "duplicate"))
                continue
            seen_users.add(user_text)
            kept.append(rec)
    return kept, dropped


def _backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def _write_atomic(path: Path, records: List[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(path)
    return len(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return dict(
        seed=args.seed,
        scale=args.scale,
        include_templates=not args.no_templates,
        include_non_crime=not args.no_non_crime,
        include_boundary=not args.no_boundary,
        include_excludes=not args.no_excludes,
        max_neg_indicators_per_category=args.max_neg_indicators_per_category,
        max_excludes_per_definition=args.max_excludes_per_definition,
        max_excludes_per_act=args.max_excludes_per_act,
    )


def _print_source_counts(counts: Dict[str, int]) -> None:
    print(f"[stage D] generated by source:")
    for src in ("templates", "non_crime", "boundary", "excludes"):
        print(f"  {src:12s} {counts.get(src, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build / validate / extend Stage D hard-negative training data.")
    parser.add_argument(
        "--out", type=Path,
        default=RAW_STAGE_DIR / "stage_d_hard_negatives.jsonl",
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--scale", type=int, default=30,
        help="Multiplier on each template family. With expanded pools, "
             "scale=30 yields ~1300 templated records before dedup.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate the existing file and exit. No writes.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Keep clean existing records, drop contaminated/duplicate ones, "
             "generate new uniques, and append. Backs up the original first.",
    )

    # Source toggles (all sources on by default).
    parser.add_argument("--no-templates", action="store_true",
                        help="Disable templated families source.")
    parser.add_argument("--no-non-crime", action="store_true",
                        help="Disable taxonomy non_crime source.")
    parser.add_argument("--no-boundary", action="store_true",
                        help="Disable taxonomy negative_indicators source.")
    parser.add_argument("--no-excludes", action="store_true",
                        help="Disable definition what_it_excludes source.")

    # Coverage caps.
    parser.add_argument(
        "--max-neg-indicators-per-category", type=int, default=0,
        help="Cap on negative_indicators used per crime category. "
             "0 = use all (default).",
    )
    parser.add_argument(
        "--max-excludes-per-definition", type=int, default=1,
        help="Cap on what_it_excludes used per (act, term). Default 1.",
    )
    parser.add_argument(
        "--max-excludes-per-act", type=int, default=150,
        help="Cap on TOTAL exclusion records per act, so BNS / MV / LABOUR "
             "do not dominate the smaller acts. Default 150. "
             "0 = uncapped.",
    )

    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    # ---------- VALIDATE ----------
    if args.validate:
        report = validate_file(args.out)
        print_report(report)
        return

    build_kwargs = _build_kwargs_from_args(args)

    # ---------- RESUME ----------
    if args.resume:
        kept, dropped = _read_existing(args.out)
        print(f"[stage D] resume: existing file -> kept {len(kept)}, "
              f"dropped {len(dropped)}")
        for line_no, reason in dropped[:20]:
            print(f"  drop line {line_no:>4d}: {reason}")
        if len(dropped) > 20:
            print(f"  ... and {len(dropped) - 20} more")

        backup = _backup_file(args.out)
        if backup:
            print(f"[stage D] backed up original -> {backup}")

        candidates, counts = build_records(**build_kwargs)
        print(f"[stage D] generated {len(candidates)} candidate records")
        _print_source_counts(counts)

        seen_users = {r["messages"][1]["content"] for r in kept}
        new_records: List[Dict[str, Any]] = []
        for rec in candidates:
            payload = rec.to_dict()
            user_text = payload["messages"][1]["content"]
            if user_text in seen_users:
                continue
            seen_users.add(user_text)
            new_records.append(payload)
        print(f"[stage D] {len(new_records)} of those are new (vs kept set)")

        merged = kept + new_records
        written = _write_atomic(args.out, merged)
        print(f"[stage D] wrote {written} records "
              f"(kept {len(kept)} + new {len(new_records)}) -> {args.out}")
        return

    # ---------- FULL OVERWRITE ----------
    candidates, counts = build_records(**build_kwargs)
    _print_source_counts(counts)
    seen_users: set[str] = set()
    out_records: List[Dict[str, Any]] = []
    for rec in candidates:
        payload = rec.to_dict()
        user_text = payload["messages"][1]["content"]
        if user_text in seen_users:
            continue
        seen_users.add(user_text)
        out_records.append(payload)
    backup = _backup_file(args.out)
    if backup:
        print(f"[stage D] backed up original -> {backup}")
    written = _write_atomic(args.out, out_records)
    print(f"[stage D] wrote {written} hard-negative records to {args.out}")

    if args.sample > 0 and out_records:
        rng = random.Random(args.seed)
        sample = rng.sample(out_records, min(args.sample, len(out_records)))
        for s in sample:
            print(json.dumps(s, ensure_ascii=False, indent=2))
            print("-" * 60)


if __name__ == "__main__":
    main()
