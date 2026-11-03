# EXP-001 Ground-Truth Verification Report (v2)

## 0. Scope and status

- Experiment: **EXP-001** (fixed-character chunking baseline; corrected run).
- Source document: `data/satcom-ngp.pdf`.
- Verified dataset: `experiments/exp001_questions.csv` (approved 15-question set, `EXP001-Q01`-`EXP001-Q15`).
- Ground-truth manifest: `experiments/exp001_ground_truth.json`.
- Verification method: exact character-offset check `page_text[start_char:end_char] == exact_supporting_excerpt`.
- Heading spans are never used as ground truth; only supporting passage content is verified.

**PASS**: all generated ground-truth evidence entries verified exactly against the extracted page text.

## 1. Summary

| Question | Expected answer | Evidence clause(s) | Source page(s) | Status | Excerpt supports expected answer |
|---|---|---|---|---|---|
| EXP001-Q01 | Which satellite systems receive preferential treatment for operations from Indian soil? | ARTICLE-1 | 1 | VALID | True |
| EXP001-Q02 | On what basis should INSAT capacity be made available to the commercial sector? | 2.3.1 | 2 | VALID | True |
| EXP001-Q03 | Who determines policies regarding the INSAT system? | 2.3.2 | 2 | VALID | True |
| EXP001-Q04 | What are the broad classifications of INSAT user sectors? | 2.4.1 | 3 | VALID | True |
| EXP001-Q05 | Who is responsible for obtaining licenses to offer services in a territory? | 2.5.6 | 3 | VALID | True |
| EXP001-Q06 | What agreement governs the use of INSAT capacity by non-governmental parties? | 2.6.5 | 4 | VALID | True |
| EXP001-Q07 | How is an Indian Satellite defined in the policy? | 3.2 | 5 | VALID | True |
| EXP001-Q08 | What authorisations or licenses are required to establish an Indian Satellite System? | 3.4.1, 3.4.2, 3.4.3 | 5, 6 | VALID | True |
| EXP001-Q09 | Where should the Satellite Control Centre (SCC) be located? | 3.6.4 | 7 | VALID | True |
| EXP001-Q10 | What is the procedure when multiple applicants request the same orbit-spectrum in new or unplanned bands? | 3.6.8 | 8 | VALID | True |
| EXP001-Q11 | What financial information must an applicant provide? | 3.7.1.5 | 9 | VALID | True |
| EXP001-Q12 | How are international satellite communication services defined? | 4.2 | 13 | VALID | True |
| EXP001-Q13 | What are the categories of domestic satellite communication services? | 4.3.1, 4.3.2, 4.3.3, 4.3.4 | 13, 14 | VALID | True |
| EXP001-Q14 | What happens when Indian satellites have the capacity and capability to provide a domestic service proposed through a foreign satellite? | 4.4.3 | 14 | VALID | True |
| EXP001-Q15 | What must be ensured before an operating license is issued for a foreign satellite system? | 4.4.6 | 15 | VALID | True |

## Question EXP001-Q01

**Question:** Which satellite systems receive preferential treatment for operations from Indian soil?

**Expected answer:** Proposals using Indian satellites are given preferential treatment over foreign satellites.

### Evidence unit EXP001-Q01-E01

- **Question ID:** EXP001-Q01
- **Evidence ID:** EXP001-Q01-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 1
- **Clause ID:** ARTICLE-1
- **Start char:** 338
- **End char:** 638
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
Government of India has decided that the Satellite Communications  Policy should be 
implemented in such a manner that while ope rations from Indian soil may be allowed 
with both Indian and foreign Satellites, proposal s envisaging use of I ndian Satellites will 
be accorded preferential treatment.
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q02

**Question:** On what basis should INSAT capacity be made available to the commercial sector?

**Expected answer:** The INSAT capacity should be made available to the commercial sector on the basis of sound business lines, i.e. on a for-profit basis.

### Evidence unit EXP001-Q02-E01

- **Question ID:** EXP001-Q02
- **Evidence ID:** EXP001-Q02-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 2
- **Clause ID:** 2.3.1
- **Start char:** 1719
- **End char:** 1985
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
As a baseline making the INSAT capacity  available to the Commercial Sector 
should be based on sound business lines, i.e.., this activity should be on a ‘for profit’ 
basis and at the same time consistent with the Government policies in the concerned user 
sectors.
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q03

**Question:** Who determines policies regarding the INSAT system?

**Expected answer:** The INSAT Coordination Committee (ICC).

### Evidence unit EXP001-Q03-E01

- **Question ID:** EXP001-Q03
- **Evidence ID:** EXP001-Q03-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 2
- **Clause ID:** 2.3.2
- **Start char:** 1995
- **End char:** 2173
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
All the policies regarding INSAT syst em shall be determined by the INSAT 
Coordination Committee(ICC)  keeping in view Policy Frame-work for Satellite 
Communications in India. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q04

**Question:** What are the broad classifications of INSAT user sectors?

**Expected answer:** Telecommunications, broadcasting, education and developmental communications, and security communications.

### Evidence unit EXP001-Q04-E01

- **Question ID:** EXP001-Q04
- **Evidence ID:** EXP001-Q04-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 3
- **Clause ID:** 2.4.1
- **Start char:** 41
- **End char:** 239
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
The user sectors can be broadly classified as: 
 
(a) Telecommunications 
(b) Broadcasting 
(c) Education and developmental communications 
(d) Security Communications for Defence Ministry/Services 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q05

**Question:** Who is responsible for obtaining licenses to offer services in a territory?

**Expected answer:** The Party which has taken the capacity on lease.

### Evidence unit EXP001-Q05-E01

- **Question ID:** EXP001-Q05
- **Evidence ID:** EXP001-Q05-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 3
- **Clause ID:** 2.5.6
- **Start char:** 1954
- **End char:** 2157
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
The responsibility for obtaining the necessary licenses to offer a service in a 
particular territory (in India or in other coun tries) shall be that of the Party which has 
taken the capacity on lease. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q06

**Question:** What agreement governs the use of INSAT capacity by non-governmental parties?

**Expected answer:** A formal lease agreement signed between the Department of Space/INSAT and the Party.

### Evidence unit EXP001-Q06-E01

- **Question ID:** EXP001-Q06
- **Evidence ID:** EXP001-Q06-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 4
- **Clause ID:** 2.6.5
- **Start char:** 2059
- **End char:** 2300
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
The use of INSAT capacity by non-govern mental Parties will be based on a 
formal lease agreement signed between the Department of Space/INSAT and the Party 
which will spell out the technical, financial, contractual and management clauses. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q07

**Question:** How is an Indian Satellite defined in the policy?

**Expected answer:** The satellites which are informed, coordinated, registered and notified by the Indian Administration following the International Telecommunications Union (ITU) Radio Regulations, for the actions of which Government of India shall be internationally responsible.

### Evidence unit EXP001-Q07-E01

- **Question ID:** EXP001-Q07
- **Evidence ID:** EXP001-Q07-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 5
- **Clause ID:** 3.2
- **Start char:** 1252
- **End char:** 1699
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
Indian Satellites” shall be defined as  the satellites which are part of a 
satellite network or system which are informed, Co-ordinated, Registered and 
Notified by the Indian Administration following the International 
Telecommunications Union (ITU) Radio Regu lations (RR) and for the actions of 
which Government of India shall be intern ationally responsible.  Indian Satellites 
are a subset of the broader concept of Indian Space Objects.  
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q08

**Question:** What authorisations or licenses are required to establish an Indian Satellite System?

**Expected answer:** Three authorizations/licenses are required: an authorisation from the Department of Space, an authorisation by the Wireless Planning and Coordination Wing, and operating licenses for the services.

### Evidence unit EXP001-Q08-E01

- **Question ID:** EXP001-Q08
- **Evidence ID:** EXP001-Q08-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 5
- **Clause ID:** 3.4.1
- **Start char:** 2218
- **End char:** 2357
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
Authorisation from Department of Space to own and operate an Indian registered 
satellite system, including the spacecraft control centre. 
```

- **Excerpt supports the expected answer:** Yes

### Evidence unit EXP001-Q08-E02

- **Question ID:** EXP001-Q08
- **Evidence ID:** EXP001-Q08-E02
- **Source:** satcom-ngp.pdf
- **Source page:** 6
- **Clause ID:** 3.4.2
- **Start char:** 394
- **End char:** 612
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
Authorisation by the Wi reless Planning and Coordination Wing (WPC) of the 
Ministry of Communications, being the Indian Administration, to operate a Space Station 
in accordance with the extant ITU Radio Regulations. 
```

- **Excerpt supports the expected answer:** Yes

### Evidence unit EXP001-Q08-E03

- **Question ID:** EXP001-Q08
- **Evidence ID:** EXP001-Q08-E03
- **Source:** satcom-ngp.pdf
- **Source page:** 6
- **Clause ID:** 3.4.3
- **Start char:** 621
- **End char:** 1136
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
Operating licenses for the se rvices to be provided by the system/network. In so far 
as India is concerned, these will be dealt wi th in accordance with the regulations in the 
particular sector as defined in Article 1. Fo r instance, for broadcasting the Broadcast Act 
will apply .and for telecommunications, the Telegraph Act will apply. The Administrative 
Ministry for Telecommunications is the De partment of Telecomm unications and for the 
Broadcasting, it is the Ministry of Information and Broadcasting. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q09

**Question:** Where should the Satellite Control Centre (SCC) be located?

**Expected answer:** Within the territory of India.

### Evidence unit EXP001-Q09-E01

- **Question ID:** EXP001-Q09
- **Evidence ID:** EXP001-Q09-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 7
- **Clause ID:** 3.6.4
- **Start char:** 1505
- **End char:** 1880
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
The Satellite Control Centre/s (SCC), i. e., the facility that monitors correct 
functioning of the key technica l parameters and controls the movements of the satellite 
with the ability to partially or complete ly close down the network, should be located 
within the territory of the Union of India, or at a place authorised by CAISS based on the 
technical requirements. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q10

**Question:** What is the procedure when multiple applicants request the same orbit-spectrum in new or unplanned bands?

**Expected answer:** First come first served, with preference to government-owned systems; if unresolved, the Administration may file a common API and decide on the applicant.

### Evidence unit EXP001-Q10-E01

- **Question ID:** EXP001-Q10
- **Evidence ID:** EXP001-Q10-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 8
- **Clause ID:** 3.6.8
- **Start char:** 6
- **End char:** 726
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
For 3.6.7(a) and 3.6.7(b) other conditions  being fulfilled, authorisation and 
notification/co-ordination/registration shall be done on a first come first served basis. In 
case of a conflict, Government owned system shall have prio rity. In case two or more 
non-governmental systems ask for the same or bit spectrum the Administration shall hold 
discussions with the Applicants and try to fulfill all the requirements. In case such a 
solution is not possible, the Administartion may file a common Advanced Publication of 
Information (API) and take a decision on the choice of the Applicant at the stage of AP 
S4 filing (Appendix 3) of ITU Radio Regulatio ns or later depending upon the progress of 
coordination. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q11

**Question:** What financial information must an applicant provide?

**Expected answer:** Evidence that the Applicant has current financial ability to meet the cost of construction and launch, plus a broad business plan showing intended sources of funding and estimated revenues.

### Evidence unit EXP001-Q11-E01

- **Question ID:** EXP001-Q11
- **Evidence ID:** EXP001-Q11-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 9
- **Clause ID:** 3.7.1.5
- **Start char:** 931
- **End char:** 1256
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
Evidence that the Applicant has current financial ability to meet the cost of 
construction and launch of the proposed sate llite(s) and that it has the financial 
resources to operate the system. A broad business plan of th e Applicant showing 
the intended sources of funding and estimated revenues shall also be provided. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q12

**Question:** How are international satellite communication services defined?

**Expected answer:** International communications include PSTN, GMPCS, Value Added Services, Programme Exchange, where one end of the link terminates in a foreign country; the Maritime Mobile Satellite Service and the Aeronautical Mobile Satellite Service may be treated as international communications.

### Evidence unit EXP001-Q12-E01

- **Question ID:** EXP001-Q12
- **Evidence ID:** EXP001-Q12-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 13
- **Clause ID:** 4.2
- **Start char:** 247
- **End char:** 1329
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
In so far as International communication is concerned the existing practice of 
using International Inter-Governmental Systems shall continue.  Private sector off-shoots, 
which retain their international character and include Indian equity will also be eligible. 
However, wherever Indian Satellites provide capacity and capability for international 
communications, the use of the same sha ll be allowed subject to International 
Agreements. The International communications  provider shall actively promote this use 
in accordance with the applicable intern ational agreements. The International 
Communications Services will include PSTN, GMPCS, Value Added Services, 
Programme Exchange etc., where one end of th e link terminates in a foreign country i.e., 
the ‘call’ either originates or terminates  in a foreign country. The Maritime Mobile 
Satellite Service (MMSS) and Aeronautical Mobile Satellite Servie (AMSS) may be 
treated as International communications, even though there may be an element of 
domestic traffic, in respect for operations with foreign satellites. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q13

**Question:** What are the categories of domestic satellite communication services?

**Expected answer:** Broadcasting; Public Switched Telephone Network (domestic); Value Added Services (domestic); Land Mobile Satellite Services (domestic and coastal waters).

### Evidence unit EXP001-Q13-E01

- **Question ID:** EXP001-Q13
- **Evidence ID:** EXP001-Q13-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 13
- **Clause ID:** 4.3.1
- **Start char:** 1605
- **End char:** 1987
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
Broadcasting (moving images (TV) a nd associated sound; and audio (radio)) 
 
(h) Free to air satellite broadcasting – digital or analogue 
(i) Broadcasting meant for reception by subscribers 
 
The receivers may be fixed, portable or mobile as the case may be. Audio-video 
teleconferncing, data transfer (one way or duplex) and computer networking are not part 
of this category. 
```

- **Excerpt supports the expected answer:** Yes

### Evidence unit EXP001-Q13-E02

- **Question ID:** EXP001-Q13
- **Evidence ID:** EXP001-Q13-E02
- **Source:** satcom-ngp.pdf
- **Source page:** 13
- **Clause ID:** 4.3.2
- **Start char:** 2044
- **End char:** 2271
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
This will have two sub-divisions namely, the network provided by the Public and Private 
sectors (presently DOT and Basic Telecomm unications Operators respectively). Also 
included will be any switched service including ISDN. 
```

- **Excerpt supports the expected answer:** Yes

### Evidence unit EXP001-Q13-E03

- **Question ID:** EXP001-Q13
- **Evidence ID:** EXP001-Q13-E03
- **Source:** satcom-ngp.pdf
- **Source page:** 14
- **Clause ID:** 4.3.3
- **Start char:** 0
- **End char:** 284
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
This includes VSATs for closed user groups , Private VSAT servi ce with shared hub, 
Cellular and Paging services  interconnection, Video Teleconferencing, Audio/Video 
Programme exchange, data broadcasting, Conput er networking, Intern et services, Multi-
media communications, etc. 
```

- **Excerpt supports the expected answer:** Yes

### Evidence unit EXP001-Q13-E04

- **Question ID:** EXP001-Q13
- **Evidence ID:** EXP001-Q13-E04
- **Source:** satcom-ngp.pdf
- **Source page:** 14
- **Clause ID:** 4.3.4
- **Start char:** 293
- **End char:** 476
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
Land Mobile Satellite Services (Domestic and Coastal Waters): 
 
This includes communications to and from moving vehicles, hand-held personal 
terminals and portable small terminals. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q14

**Question:** What happens when Indian satellites have the capacity and capability to provide a domestic service proposed through a foreign satellite?

**Expected answer:** The services shall be authorised only on Indian satellites; if capacity or capability is lacking, operations through foreign satellites may be authorised with permission to shift to Indian satellites when capacity becomes available.

### Evidence unit EXP001-Q14-E01

- **Question ID:** EXP001-Q14
- **Evidence ID:** EXP001-Q14-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 14
- **Clause ID:** 4.4.3
- **Start char:** 1457
- **End char:** 1995
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
For services listed from 4.4.2 to 4.4.4 wherever capacity and capability exists in 
the Indian Satellites, the services shall be auth orised only on Indian Satellites. If for lack 
of capacity or capability reason, operations and foreign satellites are authorised, the 
permission or license should stipulate that th e operations should shift to Indian satellites 
as soon as the capacity and capability is availa ble. To that extent the initial permission 
should be issued only for a limited period and must be reviewed before renewal. 
```

- **Excerpt supports the expected answer:** Yes

## Question EXP001-Q15

**Question:** What must be ensured before an operating license is issued for a foreign satellite system?

**Expected answer:** The system must be internationally coordinated for frequency and orbit and security concerns must be fully addressed.

### Evidence unit EXP001-Q15-E01

- **Question ID:** EXP001-Q15
- **Evidence ID:** EXP001-Q15-E01
- **Source:** satcom-ngp.pdf
- **Source page:** 15
- **Clause ID:** 4.4.6
- **Start char:** 11
- **End char:** 209
- **Verification status:** valid
- **Exact supporting excerpt:**

```text
The process of giving operating license  must ensure that the system is 
internationally coordinated for frequency and orb it. It should also ensure that the security 
concerns are fully addressed. 
```

- **Excerpt supports the expected answer:** Yes

## 2. Conclusion

All approved questions for EXP-001 passed exact source-offset verification against `data/satcom-ngp.pdf`.
The corrected retrieval evaluation may proceed. Final corrected metrics will be reported only after the corrected run completes and is inspected.
