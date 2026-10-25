"""Builds data/eval/questions.jsonl for the sample Arbor Systems corpus.

Each question has a reference answer written from the corpus, the source document(s),
a type, and a split:
  - "calib": used to fit confidence weights and the abstain threshold
  - "test":  used only for the final comparison of configurations A-D

Unanswerable questions are about topics that are NOT in the corpus. Many are
"near misses" (the corpus contains closely related information), which is where
RAG systems hallucinate most often.

Run:  python data/eval/make_questions.py
"""
import json
from pathlib import Path

A = "answerable"
M = "multi_doc"
U = "unanswerable"

QUESTIONS = [
    # --- hr_leave_policy.md
    ("How many days of paid annual leave do full-time employees get per year?", "25 working days per calendar year, in addition to public holidays.", ["hr_leave_policy.md"], A),
    ("How much annual leave does an employee with 10 years of continuous service get?", "29 days (25 plus 4 additional days).", ["hr_leave_policy.md"], A),
    ("How many unused leave days can be carried over to the next year, and until when must they be used?", "Up to 5 days, which must be taken by 31 March of the following year or they are forfeited.", ["hr_leave_policy.md"], A),
    ("How far in advance must I request a week of annual leave?", "At least 10 working days in advance, because it is 3 or more consecutive days.", ["hr_leave_policy.md"], A),
    ("From which day of sickness is a medical certificate required?", "From the 4th day of absence; up to 3 consecutive days can be self-certified.", ["hr_leave_policy.md"], A),
    ("How many days of sick leave are paid at full salary?", "Up to 15 days per calendar year; after that statutory sick pay applies.", ["hr_leave_policy.md"], A),
    ("How many days of bereavement leave do I get if my father dies?", "3 paid days, because a parent is immediate family.", ["hr_leave_policy.md"], A),
    ("When can Finance team members not take annual leave?", "During the first 5 working days of January, because of year-end closing.", ["hr_leave_policy.md"], A),
    # --- remote_work_policy.md
    ("How many days per week must employees work from the office?", "At least 2 days per week; Tuesday and Thursday are the anchor days.", ["remote_work_policy.md"], A),
    ("What are the core working hours?", "10:00 to 15:00 CET.", ["remote_work_policy.md"], A),
    ("How much is the home office equipment budget?", "A one-time budget of 400 EUR.", ["remote_work_policy.md"], A),
    ("How many days per year can I work from another country?", "Up to 20 working days per calendar year, with HR approval at least 15 calendar days in advance.", ["remote_work_policy.md"], A),
    ("Who must approve a fully remote work arrangement?", "The relevant Vice President, in writing; the arrangement is reviewed every 6 months.", ["remote_work_policy.md"], A),
    ("By when do I need to book a desk in the office?", "Through DeskFinder by 18:00 on the previous working day.", ["remote_work_policy.md"], A),
    # --- travel_expense_policy.md
    ("Within how many days must expenses be submitted?", "Within 30 days of the date the expense was incurred, through Expensly.", ["travel_expense_policy.md"], A),
    ("What is the hotel limit per night in London?", "220 EUR per night, because London is a high-cost city.", ["travel_expense_policy.md"], A),
    ("What is the meal per diem for international travel?", "70 EUR per day.", ["travel_expense_policy.md"], A),
    ("Which flight class can I book for a 4-hour flight?", "Economy class, because the flight is shorter than 6 hours.", ["travel_expense_policy.md"], A),
    ("Who must approve a business class flight?", "The Chief Financial Officer (CFO), in writing.", ["travel_expense_policy.md"], A),
    ("What is the mileage rate for using a private car?", "0.30 EUR per kilometre.", ["travel_expense_policy.md"], A),
    ("What can I do if I lose a receipt for a 20 EUR taxi ride?", "Submit a missing receipt declaration, allowed for amounts up to 25 EUR and at most 3 times per year.", ["travel_expense_policy.md"], A),
    ("When does a business trip need pre-approval?", "When its estimated total cost is above 500 EUR; the line manager pre-approves it in Expensly before booking.", ["travel_expense_policy.md"], A),
    # --- it_security_policy.md
    ("What is the minimum password length?", "14 characters.", ["it_security_policy.md"], A),
    ("How often do passwords have to be changed?", "There is no fixed rotation; passwords must be changed immediately if a compromise is suspected.", ["it_security_policy.md"], A),
    ("Who must use a hardware security key?", "System administrators and Finance team members (YubiKey).", ["it_security_policy.md"], A),
    ("How quickly must a lost laptop be reported?", "Within 1 hour of discovering the loss, to the IT Service Desk (extension 4400).", ["it_security_policy.md"], A),
    ("By what date must the annual security awareness training be completed?", "By 30 November each year.", ["it_security_policy.md"], A),
    ("Can I use a USB stick to copy customer personal data?", "No. USB storage is blocked by default and Restricted data, such as customer personal data, must never be copied to USB storage.", ["it_security_policy.md"], A),
    ("How do I report a phishing email?", "Use the Report Phish button in Outlook, or forward it to the Information Security team mailbox.", ["it_security_policy.md"], A),
    # --- data_retention_policy.md
    ("How long are customer support tickets kept?", "3 years after the ticket is closed.", ["data_retention_policy.md"], A),
    ("How long is CCTV footage retained?", "30 days.", ["data_retention_policy.md"], A),
    ("How long do we keep job applicant data?", "6 months after the recruitment process closes, or 2 years with talent-pool consent.", ["data_retention_policy.md"], A),
    ("Within how many days must the company respond to a data subject request?", "Within 30 days.", ["data_retention_policy.md"], A),
    ("Within how many hours must the supervisory authority be notified of a data breach?", "Within 72 hours of becoming aware of the breach, where required.", ["data_retention_policy.md"], A),
    ("How long are security logs retained?", "1 year.", ["data_retention_policy.md"], A),
    # --- incident_response_runbook.md
    ("What is the acknowledgement target for a SEV1 incident?", "15 minutes.", ["incident_response_runbook.md"], A),
    ("When is the secondary on-call engineer paged?", "If the primary does not acknowledge the page within 10 minutes.", ["incident_response_runbook.md"], A),
    ("How often must the status page be updated during a SEV1?", "At least every 30 minutes until the incident is resolved.", ["incident_response_runbook.md"], A),
    ("Within how many days must a postmortem be written?", "Within 5 business days of resolution, for every SEV1 and SEV2 incident.", ["incident_response_runbook.md"], A),
    ("How much are engineers paid for a week of on-call duty?", "150 EUR per week of on-call duty.", ["incident_response_runbook.md"], A),
    # --- onboarding_guide.md
    ("What time should I arrive on my first day?", "At 9:00, at the reception of the Győr headquarters.", ["onboarding_guide.md"], A),
    ("How long is the probation period?", "3 months.", ["onboarding_guide.md"], A),
    ("What is the annual cafeteria benefit budget?", "450,000 HUF per year.", ["onboarding_guide.md"], A),
    ("Which mandatory trainings do new employees complete?", "Security awareness, data protection, and the code of conduct, during the first 2 weeks.", ["onboarding_guide.md"], A),
    ("How long does an onboarding buddy support a new employee?", "For the first 3 months.", ["onboarding_guide.md"], A),
    # --- procurement_policy.md
    ("Who approves a purchase of 25,000 EUR?", "The Chief Financial Officer (CFO), with 3 quotes.", ["procurement_policy.md"], A),
    ("How many quotes are needed for a 5,000 EUR purchase?", "2 quotes, with approval from the department head and Finance.", ["procurement_policy.md"], A),
    ("What happens to supplier invoices without a purchase order number?", "They are rejected by Accounts Payable.", ["procurement_policy.md"], A),
    ("What is the standard supplier payment term?", "30 days from the invoice date.", ["procurement_policy.md"], A),
    ("What is the maximum value of a gift I can accept from a supplier?", "50 EUR; gifts worth more than 50 EUR must not be accepted.", ["procurement_policy.md"], A),
    # --- arbordesk_product_faq.md
    ("How much does the ArborDesk Professional plan cost?", "29 EUR per agent per month.", ["arbordesk_product_faq.md"], A),
    ("Is SSO available on the ArborDesk Professional plan?", "No. SAML SSO is available only on the Enterprise plan.", ["arbordesk_product_faq.md"], A),
    ("What is the maximum attachment size in ArborDesk?", "25 MB per ticket.", ["arbordesk_product_faq.md"], A),
    ("Where is ArborDesk customer data hosted?", "In EU data centres in Frankfurt, Germany.", ["arbordesk_product_faq.md"], A),
    ("What uptime does the ArborDesk Enterprise plan guarantee?", "99.9% per month, with service credits if missed.", ["arbordesk_product_faq.md"], A),
    ("How long after cancellation is ArborDesk customer data deleted?", "60 days after the end of the subscription.", ["arbordesk_product_faq.md"], A),
    # --- performance_review_process.md
    ("When are salary increases effective?", "On 1 April.", ["performance_review_process.md"], A),
    ("What is the target bonus for people managers?", "15% of base salary.", ["performance_review_process.md"], A),
    ("How long must I be in my current role before being considered for promotion?", "At least 12 months.", ["performance_review_process.md"], A),
    ("How long does a performance improvement plan last?", "60 days.", ["performance_review_process.md"], A),
    ("How many peers should I ask for 360-degree feedback?", "At least 3 peers.", ["performance_review_process.md"], A),
    # --- multi-document questions
    ("If I travel to Paris for a client meeting, what are my hotel limit and daily meal allowance?", "220 EUR per night for the hotel and 70 EUR per day for meals.", ["travel_expense_policy.md"], M),
    ("I am a new Finance team member. Which second authentication factor must I use, and when must I finish security training?", "A hardware security key (YubiKey); security awareness training within 30 days of joining.", ["it_security_policy.md", "onboarding_guide.md"], M),
    ("Can I work fully hybrid during my probation period, and how long is it?", "No. Probation lasts 3 months, and during probation employees work from the office at least 4 days per week; hybrid eligibility starts after probation.", ["remote_work_policy.md", "onboarding_guide.md"], M),
    ("How are home office purchases reimbursed, and within how many days must I submit them?", "Through the Expensly expense tool, within 30 days of the expense.", ["remote_work_policy.md", "travel_expense_policy.md"], M),
    ("We want to buy a 3,000 EUR SaaS tool. Which approvals and reviews are needed?", "Department head and Finance approval with 2 quotes, plus a review by the Information Security team, required for all software and SaaS purchases.", ["procurement_policy.md"], M),
    # --- unanswerable (not in corpus); many are near misses
    ("How many weeks of paid parental leave do employees get?", None, [], U),
    ("What is the company pension contribution percentage?", None, [], U),
    ("What is the API rate limit on the ArborDesk Starter plan?", None, ["arbordesk_product_faq.md"], U),
    ("What is the target bonus for directors?", None, ["performance_review_process.md"], U),
    ("How long are emails retained?", None, ["data_retention_policy.md"], U),
    ("Who pays the fee if a booked business flight is cancelled by the employee?", None, ["travel_expense_policy.md"], U),
    ("How much does the ArborDesk Enterprise plan cost per agent?", None, ["arbordesk_product_faq.md"], U),
    ("Does ArborDesk offer an on-premise installation?", None, [], U),
    ("What is the company car policy?", None, [], U),
    ("How many days of paid study leave can employees take?", None, ["hr_leave_policy.md"], U),
    ("What is the overtime pay rate for weekend work?", None, [], U),
    ("What is the acknowledgement target for a SEV4 incident?", None, ["incident_response_runbook.md"], U),
    ("Who is the CEO of Arbor Systems?", None, [], U),
    ("How much is the monthly gym membership subsidy?", None, [], U),
    ("Is ArborDesk HIPAA compliant?", None, [], U),
    ("What is the dress code in the Budapest office?", None, [], U),
    ("How many free parking spaces are available at the Győr headquarters?", None, [], U),
    ("What is the approval limit for purchases by team leads?", None, ["procurement_policy.md"], U),
    ("How long is the free trial of the ArborDesk Enterprise plan?", None, ["arbordesk_product_faq.md"], U),
    ("What is the per diem for domestic travel longer than 10 days?", None, ["travel_expense_policy.md"], U),
]


def main() -> None:
    out = Path(__file__).with_name("questions.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for i, (q, ref, docs, qtype) in enumerate(QUESTIONS, start=1):
            row = {
                "id": f"q{i:03d}",
                "question": q,
                "reference": ref,
                "answerable": qtype != U,
                "type": qtype,
                "source_docs": docs,
                # roughly one third of the questions (every 3rd) go to calibration
                "split": "calib" if i % 3 == 0 else "test",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_u = sum(1 for *_, t in QUESTIONS if t == U)
    print(f"Wrote {len(QUESTIONS)} questions ({len(QUESTIONS) - n_u} answerable, {n_u} unanswerable) to {out}")


if __name__ == "__main__":
    main()
