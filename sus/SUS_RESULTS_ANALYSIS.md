# §5.10 — Usability study results

**n = 10 · Sat 29 Aug 2026 · SUS (Brooke, 1996) · five task scenarios**

Scores for P1–P6 were read from the returned forms by tick position; three of
the six were verified against rendered pages by eye and matched exactly.
P7–P10 are as recorded on the observation sheets. Four ambiguities in the
records were resolved by the facilitator on 3 Sep and are noted where they
change an interpretation.

---

## The headline

**Mean SUS 83.8, and all ten participants scored above the 68-point baseline
committed to in §3.2.**

| | |
|---|---|
| n | 10 |
| Mean | **83.8** |
| Median | 83.8 |
| Range | 70.0 – 95.0 |
| SD | 9.3 |
| 95% CI | 77.1 – 90.4 |
| Above the 68 baseline | **10 / 10** |

On the Sauro & Lewis curved grading scale, 83.8 sits in grade A — roughly the
top 10% of systems measured with this instrument. The lower bound of the
confidence interval (77.1) is still comfortably above 68, so the claim "the
system exceeds the usability baseline" survives the sample size rather than
depending on it.

**Report the CI, not just the mean.** At n=10 a bare 83.8 invites the question
you should answer before it is asked.

### Per participant

| P | SUS | Group | Mode |
|---|---|---|---|
| P7 | 95.0 | technical | in person |
| P9 | 95.0 | technical | remote |
| P2 | 92.5 | technical | in person |
| P10 | 90.0 | technical | in person |
| P4 | 87.5 | technical | in person |
| P6 | 80.0 | technical | remote |
| P8 | 77.5 | technical | in person |
| P1 | 75.0 | technical | in person |
| P5 | 75.0 | non-technical | in person |
| P3 | 70.0 | non-technical | in person |

### Per item (raw 1–5, before inversion)

| Item | Mean | Spread |
|---|---|---|
| 4 — would need technical support | **1.1** | 1–2 |
| 10 — a lot to learn first | 1.3 | 1–4 |
| 2 — unnecessarily complex | 1.6 | 1–2 |
| 6 — too much inconsistency | 1.7 | 1–3 |
| 8 — cumbersome | 1.8 | 1–3 |
| 1 — would use frequently | **3.5** | 2–5 |
| 5 — functions well integrated | 3.9 | 3–5 |
| 7 — most would learn quickly | 4.5 | 4–5 |
| 9 — felt confident | 4.5 | 4–5 |
| 3 — easy to use | 4.6 | 4–5 |

Item 4 is the strongest in the instrument at 1.1, and it speaks directly to
§3.1's claim that a non-expert can operate the system — with the caveat in the
next section.

Item 1 is the weakest at 3.5, two participants at 2. That item measures *wanting
to use it*, not usability; a study corpus about a fictional library gives nobody
a reason to return. One sentence explains it rather than leaving it hanging.

---

## Task completion

The measure promised in §3.3, and one that cannot be recovered after the fact.

| Task | Scheduled | Not attempted | Attempted | Unaided | With prompting | Not completed |
|---|---|---|---|---|---|---|
| 1 — add a document | 10 | 0 | 10 | 9 | 1 | 0 |
| 2 — ask by typing | 10 | 0 | 10 | 10 | 0 | 0 |
| 3 — ask by speaking | 10 | **1** | 9 | 9 | 0 | 0 |
| 4 — ask something not covered | 10 | 0 | 10 | 10 | 0 | 0 |
| 5 — find the source | 10 | 0 | 10 | 9 | 1 | 0 |
| **Total** | **50** | **1** | **49** | **47** | **2** | **0** |

**47 of 49 attempted tasks completed unaided (96%). No task was attempted and
failed.**

The table carries both denominators on purpose. Fifty task instances were
scheduled; one was never run, so the completion rate is quoted out of 49 rather
than 50.

**That one is a protocol deviation, not a failure.** P6 was remote, and remote
screen-share carries keyboard and mouse but not a microphone, so the voice task
could not be run as written. Counting it under "not completed" would attribute a
study-design constraint to the interface, and would invite the question of which
participant failed the voice task — to which the answer is that nobody did,
because nobody was asked. Keeping it in its own column shows the deviation
without either hiding it or miscoding it.

The two prompted cases were P8 on upload (hesitated over which file formats were
accepted) and P10 on finding the source (forgot to look).

---

## The sample is eight technical to two non-technical

| Group | n | Mean | Range |
|---|---|---|---|
| Technical | 8 | 86.6 | 75.0 – 95.0 |
| Non-technical | 2 | **72.5** | 70.0 – 75.0 |

**This is the honest headline limitation and it belongs in §5.11.**

§3.1 claims the system is operable by a non-expert. Eight technical participants
cannot test that claim; they can only fail to falsify it. The two non-technical
participants are the bottom two scores in the entire sample, **14.1 points below
the technical mean**.

At n=2 that is a signal, not a finding, and it must be written as one. But it is
the signal that bears directly on the claim the chapter exists to support, and
every non-technical observation in this study points the same way — see the
layout defect below, and note that the one participant who would not trust the
system is also non-technical.

This was foreseen twice: the recruitment pack said ten classmates could not test
a non-expert claim, and the PPR risk table rated under-recruitment as *Low*
likelihood. Say both. Grading your own forecasting reads as competence, not as
failure.

Remote (n=2, mean 87.5) versus in person (n=8, mean 82.8) is not worth
interpreting at these sample sizes. State the split and move on.

---

## The layout defect — three participants, one component

This is the study's most concrete result, and it emerged from three independent
participants describing the same thing:

> P1 — *"The add files and sources were above the ask, it wouldve been better if it was below"*
> P3 — *"Wasnt able to find the chat box"*
> P5 — *"Couldnt find the chatbox"*

The source and upload panels sit **above** the question input, so the input is
below the fold. P3 and P5 both scrolled down and found it themselves — hence
Task 2 is correctly recorded as unaided for both — but both named it as the most
confusing part of the session.

**Both are the non-technical participants.** Six technical participants did not
raise it. That is what a usability study is supposed to surface: a defect
invisible to people who already expect a chat box, and the first thing hit by
people who do not.

The fix is reordering two panels. Worth stating plainly in §5.10 as the study's
primary actionable outcome, and worth carrying into §6.4.

Related, from a third angle:

> P10 — *"Finding the exact passage in the source panel required a bit of scrolling"*
> P3 — *"the sources are all over the place"*

Four of ten participants commented on the source panel's placement or
navigability. No other component drew more than one comment.

---

## The source panel is always visible — what the measure actually records

**Resolved 3 Sep: the source panel is permanently open and cannot be closed.**

So "did they open the source panel unprompted" was never answerable as written.
What the sheets actually record is whether the participant *consulted* it
without being told to: **8 of 10 did** (P8 and P10 did not; P10 was the
prompted case on Task 5).

Report it that way, and say why: *the source panel is permanently visible, so
this records engagement rather than discovery.* Two sentences, and it converts a
measure that would not survive scrutiny into one that does.

It also supports a design argument the chapter can make: keeping provenance
permanently on screen rather than behind a control meant four in five
participants read it unprompted. FR6 exists to make grounding inspectable, and
the study shows it is inspected. That is a stronger claim than "the feature
exists", and it is the one the evidence supports.

---

## The refusal — and the limitation buried in it

Task 4 was designed as the most valuable data point of the day: does declining
to answer read as *honest* or as *broken*?

**Six of ten reactions are usable. All six came from technical participants.**

> P4 — *"Nice work, its a good thing that it tells you that rather than making up fake info"*
> P6 — *"Expected it to hallucinate, but glad it hard stopped."*
> P7 — *"Oh, neat. It threw a fallback response instead of guessing."*
> P8 — *"Ah, it doesn't have the context for parking."*
> P9 — *"Good, it successfully rejected an out-of-domain prompt."*
> P10 — *"It doesn't know. That's way better than hallucinating."*

Every one frames the refusal against hallucination — a failure mode they already
knew to expect. **A participant who arrives knowing that language models
fabricate is primed to read a refusal as correct behaviour.** That is precisely
the population least able to test whether refusal reads as honest to someone who
does not.

The other four are not usable for this purpose:

- P1 *"Worked really good"*, P2 *"Hmmmm nice"*, P5 *"Good work"* — approving, but
  they do not say what was approved of.
- P3 *"Ohhh the speech recognition is good"* — captured against the wrong task;
  that is a Task 3 reaction in the Task 4 field.

**Neither non-technical participant produced a usable reaction to the refusal.**
So the defensible claim is:

> Among participants who already understood hallucination, the refusal was
> uniformly read as correct behaviour rather than as a fault. The study did not
> establish how the refusal reads to a non-expert.

That is weaker than the result hoped for and a far stronger sentence than
overstating it. It also writes the future work: the refusal question needs
asking of people who do not already know what a language model does wrong.

---

## Trust — nine yes, one no

Nine of ten would trust the system with their own documents, and the reasons
cluster on transparency:

> P1 — *"it even gives the source"*
> P2 — *"a mini google for my personal documents and i dont have to worry about privacy"*
> P7 — *"the source attribution is explicit so I don't have to guess where it pulled from"*
> P9 — *"it avoids the standard LLM trap of making things up"*
> P10 — *"since the retrieval is transparent, I can easily verify its work"*

Two qualified their yes on the privacy architecture — P8 *"as long as the backend
processing doesn't leave my local machine"*, P6 *"assuming I can self-host the
vector store locally"*. **Both conditions the system already meets.** Worth
stating: the guarantee participants spontaneously asked for is the one FR7
provides, which is evidence the local-only design answers a concern users
actually hold rather than one assumed for them.

The single no is **P5, non-technical**:

> *"Nope, id rather read it myself than give it my computer"*

Do not bury this. A non-technical participant rejecting the premise — not the
interface, the idea — is the most interesting sentence in the dataset, and it
comes from the group the chapter is least able to speak for.

---

## Other actionable findings

| Finding | Source | Fix |
|---|---|---|
| No feedback when upload completes | P6 | Completion state on the upload control |
| Accepted formats not visible until picker opens | P8 | State `.txt / .md / .pdf` on the control |
| Would prefer a familiar chat layout | P4 | Cosmetic; note only |

### The cold start, observed rather than measured

> P8 — *"During the cold start it hung for a second, wasn't sure if it crashed."*

The best single sentence in the study for Chapter 5, because it connects §5.8's
measured ~12-second cold start to what a user actually experiences: not "slow",
but *"I think it crashed."*

Use it to justify the warm-up fix. A latency number argues for an optimisation;
a user reporting they thought the system had crashed argues for it far better,
and it is evidence gathered rather than reasoning asserted.

---

## How to write this up

Chapter 5 has roughly 50 words of headroom, so §5.10 is a table and four
sentences, not an essay.

- **Results in the table.** Mean, CI, n, group split, task completion. Method
  detail goes in the caption.
- **One sentence of method.** SUS after five task scenarios, n=10.
- **Two of interpretation.** Above baseline on every participant; the 14-point
  technical / non-technical gap and the layout defect that explains it.
- **One threat to validity.** Sample composition — 8 technical to 2
  non-technical against a claim about non-experts.
- Raw per-item responses and per-task completion go in an appendix table
  (`sus_scores.csv`), which also satisfies the promise to report the split.

The qualitative material is worth more per word than the score. The score says
the system cleared a bar. The layout defect, the refusal limitation, P8's
cold-start quote and P5's refusal say what the study actually learned.
