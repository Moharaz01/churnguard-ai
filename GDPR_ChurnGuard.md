# 🔒 GDPR Compliance Document
## ChurnGuard AI — Customer Churn Prediction System
### Prepared in accordance with UK GDPR and the Data Protection Act 2018

**Document Version:** 1.0  
**Prepared By:** Portfolio Project — Data Science  
**Date:** 2025  
**Supervisory Authority:** Information Commissioner's Office (ICO), United Kingdom  
**Status:** Portfolio / Educational — Not yet deployed with real personal data

---

## 1. Overview and Purpose

This document sets out the data protection and privacy framework for the ChurnGuard AI system — an end-to-end machine learning pipeline for predicting customer churn in the telecommunications sector.

This document serves two purposes:
1. **For portfolio review:** To demonstrate professional understanding of UK GDPR obligations for ML systems
2. **For production deployment:** To serve as the starting point for a formal compliance assessment before live deployment with real customer data

---

## 2. Data Used in This Project

### Current State (Portfolio / Demo)

| Category | Detail | GDPR Relevance |
|----------|--------|----------------|
| Dataset type | Fully synthetic — algorithmically generated | No natural persons represented; UK GDPR does not apply to synthetic data |
| Sample size | 5,000 synthetic records | N/A |
| Personal identifiers | None — no names, addresses, phone numbers, or IDs | N/A |
| Special category data | None | N/A |
| Source | Generated via Python NumPy with statistical distributions | No real data processing |

**Conclusion (current state):** Because this project uses exclusively synthetic data, UK GDPR does not apply at this stage. No data subjects are affected. No consent is required. No lawful basis needs to be established.

---

## 3. Production Deployment — Obligations That Would Apply

If this system were deployed to process real customer data, the following UK GDPR obligations would apply:

### 3.1 Lawful Basis for Processing (Article 6)

Processing customer data to predict churn requires a lawful basis. The most appropriate options are:

**Option A — Legitimate Interests (Article 6(1)(f)):**
Churn prediction serves a legitimate business interest (revenue protection, customer relationship management). This must be balanced against customer rights through a Legitimate Interests Assessment (LIA).

LIA considerations:
- **Purpose test:** Is the business interest legitimate? Yes — preventing revenue loss is a recognised legitimate interest
- **Necessity test:** Is ML the least intrusive means? Possibly — simpler rule-based systems might achieve similar results; this must be assessed
- **Balancing test:** Do customer interests override the business interest? Unlikely for low-risk churn scoring, but the use of predictions (e.g., restricting services) could change this balance

**Option B — Contract Performance (Article 6(1)(b)):**
If the customer contract includes a clause about personalised service management, this may provide a basis. However, this is a narrow basis and contracts would need to explicitly include ML-based personalisation.

**Option C — Consent (Article 6(1)(a)):**
Customers explicitly consent to their data being used for churn modelling. This is rarely practical for existing customers and can be withdrawn at any time.

**Recommended basis:** Legitimate Interests, documented via a formal LIA.

### 3.2 Special Category Data (Article 9)

Churn data may inadvertently include special category data:
- **Disability-related charges:** If accessibility features are purchased, this may indicate disability
- **Senior citizen flag:** Age-related data may intersect with health conditions
- **Religious or cultural patterns:** Usage patterns around religious holidays

Where special category data is involved, Article 9 requires either explicit consent or one of the specific exemptions (e.g., 9(2)(b) employment, 9(2)(j) scientific research).

**Mitigation:** Review all features used by the model and assess whether any indirectly reveal special category attributes. Consider removing or aggregating problematic features.

### 3.3 Automated Decision-Making and Profiling (Article 22)

**Does this system trigger Article 22?**

Article 22 applies when:
- Processing is solely automated (no meaningful human involvement)
- It produces decisions that have **legal or similarly significant effects** on individuals

Churn prediction **as a scoring tool** (output reviewed by a human before any action) does **not** trigger Article 22.

Churn prediction **as an automated trigger** for actions such as:
- Automatically changing a customer's tariff
- Automatically restricting services
- Automatically sending targeted commercial offers that differentiate between customers

...may trigger Article 22 depending on the significance of the action.

**This system's design avoids Article 22 issues by:**
1. Presenting predictions as decision-support scores, not autonomous decisions
2. Including a human review step before any customer-facing action
3. Providing feature importance explanations (satisfying the right to explanation)
4. Including an audit trail of all predictions with timestamps

**If actions become automated in production**, the organisation must:
- Inform customers that automated profiling is occurring (Article 13/14 privacy notice)
- Provide a simple mechanism for customers to request human review
- Ensure meaningful human review is genuinely available (not just a rubber stamp)
- Document the safeguards in the DPIA

### 3.4 Data Protection Impact Assessment (Article 35)

A DPIA is mandatory before deployment because this system involves:
- Large-scale systematic processing of personal data
- Automated profiling of individuals
- New technology applied in a new context

**DPIA must cover:**
1. A systematic description of the processing operations and purposes
2. An assessment of the necessity and proportionality of the processing
3. An assessment of the risks to data subjects
4. Measures to address those risks

Use the ICO's DPIA template available at: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/data-protection-impact-assessments-dpias/

### 3.5 Data Subject Rights

Under UK GDPR, customers have the following rights that the organisation must be able to fulfil:

| Right | Article | Implication for Churn System |
|-------|---------|------------------------------|
| Right to information | Art. 13/14 | Privacy notice must mention ML profiling |
| Right of access | Art. 15 | Customers can request their churn score and the data used |
| Right to rectification | Art. 16 | If customer data is wrong, model must be re-run with corrected data |
| Right to erasure | Art. 17 | Churn scores must be deleted when customer account is closed |
| Right to restriction | Art. 18 | Customer can request their data not be used for profiling |
| Right to object | Art. 21 | Customer can object to profiling under legitimate interests basis |
| Rights re automated decisions | Art. 22 | Right to human review if significant automated decisions are made |

**Implementation requirements:**
- A mechanism to retrieve and export all data held about a specific customer (Subject Access Request)
- A deletion workflow that removes churn scores alongside the customer record
- A suppression flag to exclude objecting customers from the ML scoring process

### 3.6 Data Retention

Churn prediction scores are personal data and must not be retained indefinitely.

**Recommended retention schedule:**
- Active customer churn scores: Retained for the duration of the customer relationship + 30 days (to allow final account review)
- Closed account churn history: Retain for 12 months (business analytics), then anonymise or delete
- Model training data: Retain for as long as the model is in use; delete when model is decommissioned
- Audit logs of predictions: 7 years (aligns with financial record-keeping requirements)

### 3.7 Data Minimisation

Only features that are directly predictive of churn should be included in the model. Features should be reviewed against the question: "Can we achieve similar predictive performance without this feature?"

Features that should be **excluded unless strictly necessary:**
- Full name, address, phone number, email (no predictive value for churn)
- Date of birth (age grouping is sufficient — birth date is more precise than needed)
- Payment card details (payment method category is sufficient)
- Individual transaction history (aggregate spend metrics are sufficient)

### 3.8 Security Measures (Article 32)

| Measure | Requirement | Implementation |
|---------|-------------|----------------|
| Encryption at rest | All personal data stored on disk must be encrypted | AES-256 for databases and file storage |
| Encryption in transit | Data moving between systems must be encrypted | TLS 1.2+ for all API calls |
| Access control | Only authorised personnel can access churn scores | Role-based access control (RBAC) |
| Logging | Access to personal data must be logged | Immutable audit log with user ID and timestamp |
| Pseudonymisation | Where possible, replace identifiers with pseudonymous IDs | Use customer_id rather than name/email in ML pipeline |
| Penetration testing | Systems holding personal data should be tested | Annual pen test by qualified third party |

---

## 4. Records of Processing Activities (Article 30)

As required by Article 30, the following records document the churn prediction processing activity:

| Field | Value |
|-------|-------|
| Name of processing activity | Customer Churn Prediction ML System |
| Controller | [Organisation Name] |
| Data Protection Officer | [DPO Name and Contact] |
| Purpose | Predict likelihood of customer churn to enable proactive retention |
| Lawful basis | Legitimate Interests (Article 6(1)(f)) |
| Categories of data subjects | Current and recently lapsed customers |
| Categories of personal data | Account information, service subscriptions, billing data, usage data, support interactions |
| Special category data | None (subject to feature review) |
| Recipients | Internal: Customer Success, Marketing, Finance. No external recipients. |
| Third-country transfers | None (UK-hosted infrastructure only) |
| Retention period | As per retention schedule above |
| Security measures | As per Article 32 measures above |

---

## 5. Contact and Escalation

Any data protection queries relating to this system should be directed to the Data Protection Officer.

For portfolio review purposes, queries should be directed to the portfolio owner.

*This document was prepared to demonstrate professional understanding of UK GDPR obligations for ML systems. It should be reviewed and updated by a qualified Data Protection professional before any live deployment.*

---

**Regulatory References:**
- UK General Data Protection Regulation (UK GDPR) — retained EU law
- Data Protection Act 2018 (DPA 2018)
- ICO Guidance on AI and Data Protection: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/
- ICO Guidance on Automated Decision Making: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/individual-rights/automated-decision-making-and-profiling/
