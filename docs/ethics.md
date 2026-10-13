# Ethics Statement — PJ-03

## Nature of the data

The datasets used in this project (HateXplain, HASOC, Hinglish offensive
tweets, HateCheck) contain **offensive, hateful and distressing content**,
including slurs and attacks on identity groups. Anyone working with the raw
data should be aware of this and take breaks; exposure is cumulative.

## Intended use

- **Research use only.** This code and these models are for the PJ-03
  academic project. They are not a production moderation system.
- The system is designed to **support human moderators, not replace them**.
  Model output (including rationales) must always be reviewed by a person
  before any moderation action. Automated hate-speech detection makes false
  positives (silencing legitimate speech, often from the very communities
  targeted) and false negatives (missing harmful content).

## Annotation subjectivity

Hate speech labels are **subjective** and context-dependent. Annotators
disagree, both within and across datasets — the label distributions in
HateXplain (majority voting over 3 annotators) exist precisely because of
this. Any reported metric inherits these annotator limitations; we will
report agreement statistics where available and avoid over-claiming.

## Bias risks

- Models learn biases present in the data: identity terms ("muslim",
  "women", "dalit", …) can become shortcuts for the hateful label even when
  used non-hatefully. This is exactly why the bias audit (Phase 4: subgroup
  AUC, BPSN/BNSP) and the bias penalty term `R_bias` in the composite reward
  exist.
- Hinglish/code-mixed data is under-resourced; performance will differ by
  language, which could translate into uneven moderation quality across
  communities. We report per-language metrics explicitly.
- The bias-reward lexicon itself encodes choices about what counts as an
  "identity term"; it will be documented and reviewed, and its limitations
  stated in the paper.

## Responsible disclosure

We will not publish raw hateful examples beyond what the dataset licences
and venues permit, and any qualitative examples shown in the paper/slides
will be checked and, where appropriate, minimally masked.

## Human decisions still open

- Final dataset choices and licence compliance (see `docs/dataset_notes.md`).
- The 3-way → binary label mapping (e.g. is "offensive but not hateful" = 1?).
- The identity-term lexicon for the bias reward.
