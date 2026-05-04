# 종합설계프로젝트 발표 슬라이드 디자인 프롬프트
## Cycle-aware AI Model for BOSCH Process Virtual Metrology

## Purpose

종합설계프로젝트 발표를 위한 깔끔하고 일관된 슬라이드 덱을 제작한다.

슬라이드는 발표 대본이 확정된 뒤 제작하며, 발표자가 자연스럽게 설명할 수 있도록 도와주는 시각 자료로 구성한다.

발표의 핵심은 다음 흐름을 명확히 전달하는 것이다.

- 왜 이 문제가 중요한가?
- 기존 방식의 한계는 무엇인가?
- 왜 cycle 정보를 고려해야 하는가?
- 우리 모델은 어떤 방식으로 접근하는가?
- 기대 효과와 활용 가능성은 무엇인가?

슬라이드는 논문이나 보고서를 그대로 요약하는 용도가 아니다.
3~7분 발표에서 청중이 핵심 아이디어를 빠르게 이해하도록 돕는 것이 목적이다.

---

## Core Principle

Slides are not a report summary.

Slides are visual support for the spoken presentation.

Each slide should represent one clear moment in the script.

Good slides should help the presenter explain:

- the process problem
- the data structure
- the AI/ML concept
- the modeling strategy
- the expected engineering value

The script explains the details.
The slide supports the explanation.

---

## Overall Style

Use a clean, minimal, technical, and professional style.

The design should feel:

- simple
- structured
- research-oriented
- engineering-friendly
- easy to read
- consistent across the whole deck

Avoid:

- decorative design without purpose
- too many colors
- crowded diagrams
- full script sentences
- overly academic paragraphs
- inconsistent slide layouts

---

## Color Direction

Use one consistent color system across the entire deck.

Recommended style:

- Dark navy or deep blue for cover slides and main emphasis
- Soft blue or muted cyan as one accent color
- White or very light gray background for content slides
- Dark gray or black for main text
- Light gray for captions, sources, and secondary labels

Important:

- Do not change the theme color for each section.
- Distinguish sections by numbering and layout, not by changing the whole color palette.
- Key takeaway boxes should use the same style throughout the deck.

---

## Font and Text Style

Use one clean sans-serif font consistently.

Prefer:

- short phrases
- clear labels
- simple wording
- readable spacing
- technical terms only when needed

Avoid:

- long paragraphs
- full script sentences
- dense bullet lists
- unnecessary textbook definitions
- overly formal wording

Slide text should be shorter than the script.

---

## Recommended Deck Structure

Use the following flow for the 종합설계 발표.

1. Title slide
2. Motivation
3. Problem definition
4. Dataset overview
5. Why cycle information matters
6. Existing approach or baseline
7. Proposed modeling approach
8. Expected result or evaluation plan
9. Expected benefits
10. Final takeaway

If the presentation time is short, combine related slides.

Recommended short version:

1. Motivation
2. Dataset & cycle structure
3. Proposed model
4. Expected effects
5. Conclusion

---

## Title Slide

The title slide should include:

- project title
- team name
- members
- course name
- professor name, if needed
- presentation date

Example:

Cycle-aware Virtual Metrology Model  
for BOSCH Etching Process

종합설계프로젝트  
Team XX  
Name / Name / Name / Name

---

## Content Slide Header

Each content slide should have a small label at the top.

Use a consistent format such as:

01 · MOTIVATION  
02 · DATASET STRUCTURE  
03 · CYCLE-AWARE MODELING  
04 · EXPECTED EFFECTS

Then include:

- a clear slide title
- a short subtitle or framing sentence
- body content below

The slide title should express the main point of that slide.

Example:

03 · CYCLE-AWARE MODELING

Why cycle information should not be ignored

Each BOSCH cycle affects the final etch profile,
so the model should learn process changes over time.

---

## Script Alignment

Slides must follow the script order exactly.

Before finalizing, check:

- Does each slide match one part of the script?
- Does the slide order match the speaking order?
- Are there slide points that the script does not explain?
- Are there script points missing from the slides?
- Does the slide help the presenter continue naturally?

Do not add textbook material that is not used in the script.

---

## Motivation Slide

The motivation slide should explain why this project matters.

Recommended structure:

Problem → Limitation → Need

Example:

PROBLEM  
BOSCH 공정은 반복적인 etch/passivation cycle로 진행된다.

LIMITATION  
기존 VM 모델은 전체 공정 결과만 보고 예측하는 경우가 많다.

NEED  
Cycle별 변화 정보를 반영하면 공정 상태를 더 정밀하게 예측할 수 있다.

Use a simple process diagram if possible.

Recommended visual:

Input recipe / sensor data  
→ BOSCH cycle sequence  
→ final etch profile / target value

---

## Dataset Overview Slide

Dataset slides should be simple and visual.

Show:

- input data
- output target
- cycle structure
- preprocessing direction

Recommended layout:

LEFT: Data source  
CENTER: Cycle-based sequence  
RIGHT: Prediction target

Example:

INPUT  
process recipe / sensor log / cycle data

SEQUENCE  
cycle 1 → cycle 2 → cycle 3 → ... → cycle N

OUTPUT  
etch depth / profile / CD / process quality metric

Avoid showing too many raw variables at once.
Group variables by meaning instead.

---

## Cycle Importance Slide

This slide should clearly answer:

Why do we need to consider cycle?

Recommended structure:

Without cycle-aware modeling  
- treats the process as one static input
- may miss temporal changes
- weaker explanation of accumulated effects

With cycle-aware modeling  
- reflects repeated BOSCH cycle behavior
- captures time-dependent process variation
- improves interpretability and prediction potential

Recommended visual:

Static model:
Whole process data → Model → Prediction

Cycle-aware model:
Cycle 1 → Cycle 2 → Cycle 3 → Sequence model → Prediction

Key message:

KEY TAKEAWAY  
BOSCH 공정은 반복 cycle의 누적 결과이므로, cycle 정보를 반영하는 모델이 공정 특성을 더 잘 설명할 수 있다.

---

## Baseline Slide

If a baseline model is included, explain it simply.

Examples:

- XGBoost baseline
- MLP baseline
- CNN/LSTM/Transformer comparison
- Non-cycle feature model

Recommended layout:

BASELINE  
Uses aggregated process features

LIMITATION  
Does not explicitly model cycle order

ROLE IN PROJECT  
Used as a comparison point for the proposed cycle-aware model

Avoid making the baseline look useless.
Explain that it is a reasonable starting point but has limitations.

---

## Proposed Model Slide

The proposed model slide should show the model flow visually.

Recommended structure:

Input → Feature encoding → Cycle-aware modeling → Prediction

Example:

Cycle-level process features  
→ feature embedding  
→ sequence model / attention / temporal model  
→ predicted process result

If using Time-LLM, LSTM, Transformer, or FiLM, show the model at a high level.
Do not over-explain the internal math unless the presentation requires it.

Possible labels:

- Cycle feature extraction
- Temporal dependency learning
- Prediction head
- Output target

---

## Model Concept Explanation

When explaining AI concepts, keep them intuitive.

Use short explanations:

XGBoost  
A strong tabular-data baseline that combines many decision trees.

LSTM  
A sequence model that can learn how previous cycles affect later results.

Transformer  
A sequence model that can learn which cycle information is more important.

FiLM  
A conditioning method that changes feature representation depending on additional process context.

Time-LLM  
A method that adapts language-model-style sequence learning to time-series data.

Do not include all of these unless they are actually used in the script.

---

## Expected Effects Slide

The expected effects slide should connect technical improvement to engineering value.

Recommended structure:

1. Prediction improvement  
Cycle-level information may improve VM prediction accuracy.

2. Better process understanding  
The model can help identify which cycle regions affect the final result.

3. Reduced measurement burden  
More reliable VM can reduce dependence on expensive or delayed metrology.

4. Process optimization support  
The model can support recipe tuning and abnormal trend detection.

Use one consistent key takeaway box.

Example:

KEY TAKEAWAY  
Cycle-aware VM은 단순 예측 정확도 향상을 넘어, 공정 이해와 최적화에 활용될 수 있다.

---

## Evaluation Slide

If evaluation is included, show the plan clearly.

Include:

- baseline model
- proposed model
- metrics
- validation method

Recommended layout:

MODEL COMPARISON  
Baseline model vs. Cycle-aware model

METRICS  
MAE / RMSE / R² / prediction error

VALIDATION  
train / validation / test split  
or wafer-level / lot-level split if applicable

INTERPRETATION  
Lower error means better virtual metrology performance.

Do not only show numbers.
Always explain what the result means operationally.

---

## Results Slide

If actual results are available, use simple charts.

Recommended charts:

- predicted vs. actual scatter plot
- error comparison bar chart
- RMSE/MAE table
- cycle attention or feature importance visualization

Avoid:

- too many graphs on one slide
- raw training logs without interpretation
- small unreadable tables

Each result slide should answer:

What improved?  
Compared to what?  
Why does it matter?

---

## Key Takeaway Boxes

Use one consistent key takeaway style throughout the deck.

Preferred style:

- light gray or pale blue background
- thin accent line or simple border
- dark text
- short sentence
- minimal design

Example:

KEY TAKEAWAY  
Cycle 정보를 반영하면 BOSCH 공정의 누적 변화와 최종 결과 사이의 관계를 더 잘 학습할 수 있다.

Avoid:

- different takeaway colors for each section
- long takeaway paragraphs
- introducing new information in the takeaway box

---

## Diagram Style

Use simple flow diagrams rather than complex architecture diagrams.

Recommended diagram types:

- process flow
- model pipeline
- comparison diagram
- input/output structure
- before/after concept diagram

Use consistent shapes:

- rounded rectangles for process steps
- arrows for flow
- small labels for explanation
- one accent color for important parts

Avoid using too many arrows or overly detailed neural network diagrams.

---

## Technical Term Handling

Use technical terms naturally, but do not overload the slide.

Important terms may include:

- BOSCH process
- etch/passivation cycle
- virtual metrology
- process variation
- time-series data
- cycle-aware modeling
- feature extraction
- baseline model
- prediction accuracy
- model interpretability
- process optimization

Definitions should be short and connected to the project.

Example:

Virtual Metrology  
실제 측정 없이 공정 데이터로 결과를 예측하는 방법

Cycle-aware Modeling  
각 cycle의 순서와 변화를 함께 학습하는 방식

---

## Image and Source Handling

Use images only when they help explain the project.

Possible visuals:

- BOSCH process schematic
- etch/passivation cycle diagram
- dataset structure diagram
- model pipeline diagram
- result graph
- evaluation comparison

If using external images, add a small source note.

Preferred format:

Image: BOSCH process schematic | Source: paper/company/website name

Rules:

- keep source text small
- place it near the image or at the bottom
- use light gray text
- do not let source text compete with the main content

If an image is needed but unavailable, leave a clear placeholder.

Example:

[Insert BOSCH process diagram here]  
[Insert cycle sequence visualization here]

---

## Text Density Rule

Each slide should be readable quickly.

Use:

- short phrases
- clear grouping
- enough whitespace
- no more text than needed

Avoid:

- filling every empty space
- repeating the same sentence
- long bullets that wrap too much
- small text that is hard to read

If the slide feels crowded, reduce text.
Do not shrink everything just to keep all wording.

---

## Design Consistency Checklist

Before finalizing the deck, check:

1. Does every slide follow the same color system?
2. Are section labels consistent?
3. Does the slide order match the script?
4. Is the motivation clear?
5. Is the dataset structure easy to understand?
6. Is the need for cycle-aware modeling obvious?
7. Is the proposed model shown as a clear pipeline?
8. Are baseline and proposed model compared fairly?
9. Are expected effects connected to engineering value?
10. Are key takeaway boxes consistent?
11. Are charts and diagrams readable?
12. Is there any unnecessary textbook explanation?
13. Is each slide clean enough to present from?

---

## Final Design Philosophy

Problem first.  
Cycle structure second.  
Model third.  
Engineering value last.

The slide deck should help the audience understand:

What is the process problem?  
Why is the existing approach limited?  
Why does cycle information matter?  
How does our model use it?  
What value can it create?

Keep the design unified, minimal, and presentation-friendly.