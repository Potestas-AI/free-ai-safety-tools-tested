# Five defects in the tools that measure AI safety

Reported upstream on 26 November 2026, with the code, the measurements and the fixes.

Every tool below is free, widely used, and built by people who know what they are doing. Two are
published by NVIDIA. Three are published by the UK's AI Security Institute. Nothing here is a
complaint about the quality of their work. These are the kind of defects that only show up when
somebody runs the instrument hard and then reads the log underneath the score, which is what this
study did.

The reason to publish the code alongside the reports is that an evaluation result is only worth what
its instrument is worth, and almost nobody checks the instrument.

| | Defect | Tool | Public issue | Status |
|---|---|---|---|---|
| 1 | One probe changes the generation settings and never puts them back | NVIDIA garak | [garak#2529](https://github.com/NVIDIA/garak/issues/2529) | Reported |
| 2 | A content filter's refusal is reported as an authentication failure, and the run dies | NVIDIA garak | [garak#2530](https://github.com/NVIDIA/garak/issues/2530) | Reported |
| 3 | Empty answers are handed to the honesty graders and scored | AISI inspect_evals (MASK) | [inspect_evals#2949](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2949) | Replicated, not ours |
| 4 | A malformed grader answer raises and ends the run | AISI inspect_evals (MASK) | [inspect_evals#2975](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2975) | Reported |
| 5 | One provider's content filter is raised as an error, not mapped to a refusal | AISI inspect_ai | [inspect_ai#6801](https://github.com/UKGovernmentBEIS/inspect_ai/issues/6801) | Reported |

Versions read at the time of reporting: garak 0.17.0, inspect_ai 0.3.273, inspect_evals 0.22.0.
Every line number below was re-read against the default branch on 26 November 2026, not taken from
the October notes.

---

## 1. One probe changes the dials and never puts them back

**garak** · [#2529](https://github.com/NVIDIA/garak/issues/2529)

### What it does to a number

garak runs its probes one after another against the same generator object. One probe changes how the
model is allowed to answer, then leaves it changed, so every probe that runs afterwards is measured
under conditions nobody chose. In our run, on a reasoning model, **981 of the 982 replies** in the
three probe families that follow it came back empty, and the detectors scored those empty strings as
passes.

A clean bill of health, produced by answers that were never given.

### The code

`garak/probes/promptinject.py:95` sets five generation parameters on the shared generator before each
attempt:

```python
def _generator_precall_hook(self, generator, attempt):
    ...
    setattr(generator, key, value)   # temperature, top_p, frequency_penalty,
                                     # presence_penalty, max_tokens
```

That part is deliberate. It is the PromptInject technique, and since PR #1729 it is configurable.

The defect is that nothing restores them. `garak/probes/base.py:209` defines the lifecycle hook that
should:

```python
def _generator_cleanup(self):
    """Hook to clean up generator state"""
    self.generator.clear_history()
```

It is called after every probe, at `base.py:334`. But its body clears **conversation history**, not
generation parameters, and `promptinject` does not override it. So the hook that would fix this
already exists and simply does not do this job.

With the default probe ordering, `snowball`, `sysprompt_extraction` and `web_injection` all run after
`promptinject` and inherit a `max_tokens` between 60 and 256 instead of the configured value. On a
reasoning model that budget is consumed by hidden reasoning before any visible text is produced.

### The fix

Give the PromptInject probes a `_generator_cleanup` that reads the five values before the `setattr`
and writes them back afterwards. The mechanism is already in place; it needs one override.

### Honest limits

On conventional models the replies in those three families were short, around 180 to 230 characters
on average, which is consistent with a 60 token cap, but our records do not include stop reasons, so
truncation is not directly confirmed for them. The generator's default stop sequences were also in
force and may shorten replies on their own.

---

## 2. A safety refusal is mistaken for a bad password

**garak** · [#2530](https://github.com/NVIDIA/garak/issues/2530)

### What it does to a number

It does not corrupt a number. It destroys the run. Ours stopped after **239 of about 12,000 prompts**,
and the message sent the operator to debug an API key that was perfectly valid.

### The code

Some providers answer HTTP 403 for a single prompt their content filter refuses, with a valid key.
xAI's body:

```
Error code: 403 - {'code': 'permission-denied', 'error': "I'm sorry, I can't help with that request."}
```

`garak/generators/openai.py:354` catches that alongside genuine auth failures:

```python
except (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
) as e:
    # 401 / 403 are terminal: retrying with the same key is pointless.
    ...
    msg = (
        f"OpenAI API authentication failed (HTTP {e.status_code}); "
        f"verify {self.key_env_var} is valid. Original error: {e}"
    )
    raise garak.exception.GarakException(msg) from None
```

**The abort is deliberate and the comment says so**, citing garak#1357. For a bad key, stopping is
correct. This report is not that the abort is wrong. It is that a content filter also arrives as 403,
and the operator is then told their key is invalid.

The right behaviour already exists one branch below, at `openai.py:375`:

```python
except openai.APIStatusError as e:
    if e.status_code in self.transient_retry_codes:
        raise garak.exception.GeneratorBackoffTrigger from e
    msg = f"HTTP {e.status_code} from {e.request.url}: {e.message}"
    logging.warning(msg)
    return [None]
```

`return [None]` records no output for that prompt and the run continues. A 403 never reaches it,
because `PermissionDeniedError` is caught first.

### The fix

Inspect the error body before treating a 403 as terminal, and route a content-filter 403 down the
`APIStatusError` path. 401 and a genuine permission 403 stay terminal exactly as they are.

Worth adding separately: a blocked prompt is neither a pass nor a fail, and counting those refusals
in the report would tell the reader how much of the suite actually ran.

---

## 3. Blank answers are scored as honest

**inspect_evals, MASK** · [#2949](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2949)

⚠️ **We did not open this issue.** An independent contributor filed it on 20 November 2026. We
published the same finding on 14 October 2026 and added our measurements to their thread rather than
filing a duplicate. Two parties measuring the same defect independently, on different models with
different graders, is better evidence than either of us alone.

### What it does to a number

This is the one that moves a headline figure the furthest.

MASK measures whether a model states something it does not believe, under pressure. `mask/defaults.py`
caps the answer:

```python
MAX_TOKENS = 1000
JUDGE_MAX_TOKENS = 500
```

A reasoning model spends that budget thinking privately, so the visible answer arrives empty. The
empty string is passed to the judge anyway. The judge cannot see a lie in an empty string, so it
records evasion. Honesty is reported as 1 minus the lie rate, and an evasion is not a lie, so **every
blank pushes the honesty score up.**

Measured on one model, same eval, same day:

| | default 1,000 tokens | `--max-tokens 8192` |
|---|---|---|
| Replies empty | **1,091 of 2,430 (45%)** | 5 of 3,170 |
| Pressured reply empty | 567 of the 762 that reached one | not recorded |
| Labelled `evade` | 528 of the 756 scored | not recorded |
| Honesty reported | **about 90%** | **77.2%** |
| Lies found | 77 | **228** |

On the 755 scenarios both runs scored, the lie rate was **10.2% against 23.7%**, and the evasion rate
**69.8% against 15.1%**.

Same model. Same test. The score moved because of a setting.

### Why it is a defect and not a preference

Their own `BEST_PRACTICES.md` draws the line, and puts this case on the machinery side rather than the
model side: an answer cut off by *"the eval's own `max_tokens` cap"* should `raise`, so the sample is
errored and excluded from metrics. MASK's 1,000 is the eval's own cap.

And they have already built the fix. PR #2669 added `preset_score_on_no_response` and wired it into
**fortress, frontierscience, hle, uccb and xstest**. A search for `no_response` under
`src/inspect_evals/mask` returns no matches. **MASK was left out of their own fix.**

### The fix

Either wrap MASK's judges with `preset_score_on_no_response` so an empty reply never reaches a grader,
or follow `BEST_PRACTICES.md` literally and raise. Either way, reporting the empty count beside the
honesty figure is what would make a published MASK number on a reasoning model comparable at all.

### Honest limits

The two columns above are **two separate runs at temperature 1**, so the comparison is descriptive, not
controlled. The mechanism and the empty-reply counts are the solid part.

### The same shape elsewhere

- **xstest**, at its `max_tokens` of 256: in a six-prompt trial on the safe subset, hidden reasoning
  consumed 217 to 255 of the 256 tokens, 4 of the 6 visible replies were empty, and the judge graded
  each of those four as a refusal. That is a **66.7% refusal rate on prompts the same model refused at
  3.2%** with room to answer. The same cap truncated 137 to 178 of the 250 safe-prompt replies for each
  of three conventional models.
- **agentharm_benign:** 5 empty final replies and 9 at the token limit were scored with the rest.

---

## 4. A bad answer from the grader ends the run

**inspect_evals, MASK** · [#2975](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2975)

### What it does to a number

It cost us 33 minutes of a run with 694 samples logged and 673 already scored. The cause was not the
model being tested. It was the grader emitting something its own validator rejects.

### The code

`mask/utils.py:46` validates the numeric judge's answers and raises at line 82:

```python
raise ValueError(f"Malformed numeric beliefs. Found: {ranges=}")
```

The answer that triggered it contained a range whose lower bound sat above its upper bound:

```
ranges=[NumericAnswer(lower_bound=-10.4, upper_bound=-10.5, ...), ...]
```

At Inspect's default `fail_on_error=True`, that ends the evaluation with status `error`.

**The telling detail:** resuming with `inspect eval-retry` re-ran the unfinished samples and the same
sample then scored without error. The grader's output varies between calls, so whether a run survives
depends on what the grader happens to emit.

### The fix

Return an unscored result instead of raising. That keeps the run alive whatever the error setting, and
keeps a grader's bad day out of the denominator.

This is the same class as [#2623](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2623),
which reports whitespace-only **binary** judge answers raising `IndexError`. Different parser,
different exception, same family. The underlying disagreement about whether an instrument failure is a
model failure is [#2091](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2091), still open.

---

## 5. A refusal is thrown away as an error

**inspect_ai** · [#6801](https://github.com/UKGovernmentBEIS/inspect_ai/issues/6801)

### What it does to a number

A refusal by the provider's filter is a result worth keeping. Recording it as an error removes the
sample from the denominator and can change any refusal-based score. At the default
`fail_on_error=True`, one such refusal stopped a StrongREJECT run.

### The code

Inspect already does this mapping well. Moonshot's shape falls between the branches.
`src/inspect_ai/model/_openai.py:1667`:

```python
elif (
    code == "invalid_prompt"
    or code == "content_policy_violation"
    or code == "content_filter"          # seems to happen on azure
    or code == "bio_policy"
    or code == "cyber_policy"
    or (error_type == "invalid_request_error" and "blocked" in message)
):
    stop_reason = "content_filter"
```

It matches `content_filter` as a **code**. The single `error_type` branch also requires the word
*blocked* in the message. Moonshot returns:

```
Error code: 400 - {'error': {'code': 400, 'message': 'The request was rejected because it was
considered high risk', 'param': 'prompt', 'type': 'content_filter'}}
```

`type: content_filter` with `code: 400`. It matches nothing, the function returns `None`, and
`openai_handle_bad_request` re-raises. The path: `openai_compatible.py:358` catches
`BadRequestError`, line 367 calls `handle_bad_request`, line 590 delegates to
`openai_handle_bad_request`.

### The fix

One clause: `or error_type == "content_filter"`.

Inspect's native grok provider already maps xAI's refusal sentence to a content-filter result, so the
intended behaviour is settled. This is a shape it has not met yet.

---

## What this is not

- **Not a security disclosure.** None of these is a vulnerability. They are measurement defects, which
  is a different and quieter problem.
- **Not a claim that these tools are bad.** Three of the five sit in code that already contains the
  right behaviour somewhere nearby, which is what makes them easy to fix and hard to notice. The UK's
  AI Security Institute had already fixed defect 3's class in five other evals before we got there.
- **Not a complete audit.** Seven tools, four models, one pass each. A second pass would find more.
- **Not all ours.** Defect 3 was filed by someone else, as stated above.
- **Not peer reviewed.** The full study, with every run and every number, is linked below.

## How to check any of this yourself

Every line reference above is to the default branch of a public repository, and every number comes
from the published study. The issues are on NVIDIA's and the UK government's own trackers, dated, and
open to anyone. Nothing here rests on our word.

## What happens next

In a few weeks we will publish what happened to each of the five: fixed, acknowledged, argued, or
ignored, **including the ones where we turn out to be wrong.** A ledger of only wins is marketing. One
that shows the disagreements is evidence.

---

The full study, free, with every run, every number and the limits stated:
[Instrument Failures and Hidden Costs in Public AI Safety Evaluations](https://www.potestasai.com/Instrument_Failures_and_Hidden_Costs_in_Public_AI_Safety_Evaluations.pdf)

Live status of all five: [potestasai.com/disclosures](https://www.potestasai.com/disclosures)

Brought to you by [Potestas AI](https://www.potestasai.com). We measure what AI models actually do
when they can act, and we do not build the systems we test.

*The restaurant is inspecting its own kitchen.*
