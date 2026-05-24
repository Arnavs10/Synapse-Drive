# Synapse Drive — ADAS Fuzzy Intelligence System Dashboard — Final Website Explanation

**Project:** Hybrid Fuzzy-LSTM ADAS (Advanced Driver Assistance System)  
**Dashboard Name:** Synapse Drive  
**Purpose:** Interactive web-based demonstration of the complete hybrid Mamdani Fuzzy + LSTM collision risk assessment system

---

The website is a **3-page interactive dashboard** that demonstrates the entire hybrid ADAS system in action. It has three tabs: **Dashboard** (main interactive page), **Analysis** (comparison and visualization), and **About** (system documentation).

---

## PAGE 1: DASHBOARD (Main Interactive Page)

This is the primary page where users interact with the system in real time. It has the following sections:

---

### 1. HEADER BAR (Top of page)

**What it shows:**
- **"Synapse Drive"** — the project name / brand
- **"ADAS Fuzzy Intelligence System"** — subtitle describing what the system is
- **Live Metric Summary** — shows the current prediction values at a glance: `48.3% BRAKE | 51.7% RISK | 49.4% HYBRID | MODERATE` (updates each time a prediction is computed)
- **"Fuzzy Online"** indicator (green dot) — shows that the Mamdani Fuzzy Inference System is loaded and operational
- **Timestamp** — current time

**Why it matters for viva:**
> This header provides at-a-glance situational awareness — a real ADAS dashboard would display similar real-time metrics so the driver/system operator can instantly see the current risk status.

---

### 2. FOUR METRIC CARDS (Below header)

Four large colored cards showing the primary outputs:

| Card | Value | Description |
|------|-------|-------------|
| **Brake Force** | 48.3% | Output from the Mamdani Fuzzy system — how hard brakes should be applied |
| **Fuzzy Risk** | 51.7% | Collision risk computed by the fuzzy inference system (FIS Output) |
| **Driver Alert** | 88.8% | Alert level for the driver from fuzzy system |
| **Hybrid Risk** | 49.4% | Combined risk from both Fuzzy (60%) and LSTM (40%) — the FINAL decision metric, labeled "60% Fuzzy + 40% LSTM" |

**Why it matters for viva:**
> These cards show all four key outputs: three from the Mamdani FIS (Brake, Risk, Alert) and one from the Hybrid Engine. The hybrid risk card explicitly shows the fusion weights being used (60/40 in this case, because distance=80m is > 20m).

---

### 3. SENSOR INPUT PANEL (Left side)

**What it shows:**
- **Distance slider** (0-100m) — currently set to 80m
- **Relative Speed slider** (0-100 km/h) — currently set to 80 km/h
- **"Compute Prediction" button** — triggers the full hybrid prediction pipeline

**How it works:**
When you adjust the sliders and click "Compute Prediction":
1. The Mamdani FIS computes brake, risk, and alert values
2. The LSTM generates a sequence and predicts collision probability
3. The hybrid engine fuses both with adaptive weights
4. TTC override is checked
5. All displays update with new values

**Why it matters for viva:**
> This is the user interface for the system. In a real ADAS, these inputs would come from sensors (LiDAR for distance, radar for speed). Here, the sliders simulate sensor readings for demonstration purposes.

---

### 4. QUICK SCENARIOS PANEL (Below sensor input)

Six preset scenario buttons for quick testing:

| Scenario | Distance | Speed | Expected Category |
|----------|----------|-------|-------------------|
| **Emergency** | 5m | 90 km/h | Critical (HIGH) |
| **Dangerous** | 20m | 70 km/h | High |
| **Normal** | 50m | 50 km/h | Moderate |
| **Highway** | 80m | 80 km/h | Moderate |
| **Calm** | 50m | 20 km/h | Low |
| **Safe** | 95m | 10 km/h | Low (Safe) |

**Why it matters for viva:**
> These presets let you quickly demonstrate the system's behavior across the full range of scenarios without manually adjusting sliders. They cover everything from life-threatening emergencies to perfectly safe situations, showing that the system responds appropriately to each.

---

### 5. FUZZY OUTPUTS PANEL (Center)

**What it shows:**
- **Five horizontal progress bars:**
  - **BRAKE** (green bar) — 48.3% — fuzzy brake force output
  - **RISK** (red bar) — 51.7% — fuzzy collision risk output
  - **ALERT** (orange bar) — 88.8% — fuzzy driver alert output
  - **LSTM** (dark blue bar) — 46.0% — LSTM collision probability (converted to %)
  - **HYBRID** (blue/teal bar) — 49.4% — final fused hybrid risk

- **MODERATE Risk** badge (top right) — the categorical risk status (LOW/MODERATE/HIGH)

- **TTC Value** — 3.6s — Time-To-Collision in seconds

- **Spider/Radar Chart** — a pentagon/diamond-shaped radar chart showing Brake, Risk, Alert, and Hybrid values as a "Fuzzy polygon." This gives a visual shape to the current prediction — a larger polygon = higher overall danger.

**Why it matters for viva:**
> This panel is the core output display. It shows ALL outputs from all three systems (Fuzzy, LSTM, Hybrid) in one view. The radar chart is particularly useful because its shape instantly communicates the overall threat level. The TTC value adds a physics-based metric for context.

---

### 6. MEMBERSHIP FUNCTIONS PANEL (Right side)

**What it shows:**
Three small real-time plots:
- **Distance (M)** — shows the Near/Medium/Far membership functions with the current input value (80m) marked, showing how much it belongs to each category
- **Rel Speed (KM/H)** — shows the Low/Medium/High speed membership functions with current value (80 km/h) marked
- **Brake Output** — shows the Low/Medium/High brake output membership functions

Each plot has a legend: `Near/Low • Medium • Far/High`

**Why it matters for viva:**
> This is a live visualization of the fuzzy inference process. You can see exactly which membership functions are activated for the current inputs and by how much. For example, at 80m distance, the input has high membership in "far" and partial membership in "medium." This is the core of how fuzzy logic works — you can point to this panel and explain fuzzification in real time.

---

### 7. EVENT LOG (Right side, below MFs)

**What it shows:**
A chronological log of system events:
- `[BOOT] Fuzzy inference engine ready — Mamdani FIS`
- `[BOOT] Trapezoidal boundary MFs loaded — 9 rules compiled`
- `[BOOT] Physics fallback enabled for edge cases`
- `[BOOT] Adaptive LSTM hybrid predictor online (60/40→70/30)`

**Why it matters for viva:**
> This log shows the system initialization sequence — confirming that all four components are loaded: (1) Mamdani FIS, (2) Trapezoidal MFs + 9 rules, (3) Physics fallback, (4) Adaptive LSTM hybrid engine. It also shows the adaptive weight configuration.

---

### 8. RULE FIRING TABLE (Bottom left)

**What it shows:**
A table listing all 9 Mamdani rules:

| # | IF DISTANCE | IF SPEED | BRAKE | RISK | STRENGTH |
|---|-------------|----------|-------|------|----------|
| R1 | near | high | high | high | — |
| R2 | near | medium | high | high | — |
| R3 | near | low | medium | medium | — |
| R4 | medium | high | medium | high | — |
| R5 | medium | medium | medium | medium | — |
| R6 | medium | low | low | medium | — |
| **R7** | **far** | **high** | **medium** | **medium** | **1.000** |
| R8 | far | medium | low | low | — |
| R9 | far | low | low | low | — |

The **STRENGTH** column shows the activation level of each rule for the current input. In the screenshot, Rule R7 (far + high speed) has strength 1.000 — meaning it's FULLY activated (distance=80m is fully "far" and speed=80 km/h is fully "high").

**Why it matters for viva:**
> This is the most important explainability feature. You can literally SEE which rules fired and with what strength. This is what makes fuzzy logic interpretable — you can tell the examiner "Rule R7 fired with full strength because 80m is fully 'far' and 80 km/h is fully 'high', producing medium brake and medium risk."

---

### 9. PREDICTION HISTORY CHART (Bottom right)

**What it shows:**
A line chart tracking **Brake** (blue), **Risk** (red), and **Hybrid** (orange) values across multiple prediction runs (#1 through #11 in the screenshot).

**Why it matters for viva:**
> This shows how the system responds to different scenarios over time. You can demonstrate the system by clicking different Quick Scenario buttons and watching how the curves change. This simulates a time-series of driving conditions.

---

## PAGE 2: ANALYSIS

This page provides deeper analytical views:

---

### 1. SCENARIO COMPARISON (Left)

**What it shows:**
A **grouped bar chart** comparing Brake (blue), Risk (red), and Alert (purple) values across all six scenarios (Emergency, Dangerous, Normal, Highway, Calm, Safe).

**Why it matters for viva:**
> This chart validates the system's behavior at a glance. You can see that Emergency has the highest bars across all three outputs, while Safe has the lowest. The gradual decrease from Emergency to Safe confirms that the system responds proportionally to danger.

---

### 2. SCENARIO TABLE (Right)

**What it shows:**
A data table with columns: **Scenario | Dist | Speed | Brake | Risk | Alert | Category**

| Scenario | Dist | Speed | Brake | Risk | Alert | Category |
|----------|------|-------|-------|------|-------|----------|
| Emergency | 5m | 90km/h | 83.7% | 85.7% | 88.8% | **HIGH** |
| Dangerous | 20m | 70km/h | 66.3% | 84.1% | 49.0% | **HIGH** |
| Normal | 50m | 50km/h | 48.3% | 51.7% | 53.3% | **MODERATE** |
| Highway | 80m | 80km/h | 48.3% | 51.7% | 88.8% | **MODERATE** |
| Calm | 50m | 20km/h | 11.7% | 51.7% | 11.3% | **LOW** |
| Safe | 95m | 10km/h | 11.7% | 15.2% | 11.3% | **LOW** |

**Why it matters for viva:**
> This table provides exact numerical values for all scenarios, confirming the system's correctness. Key observations you can point out:
> - Emergency (5m, 90km/h) → all outputs are HIGH (~84-89%) ✓
> - Safe (95m, 10km/h) → all outputs are LOW (~11-15%) ✓
> - Highway has high ALERT (88.8%) even though risk is moderate — because high speed at any distance triggers a warning

---

### 3. INFERENCE SURFACE (Bottom)

**What it shows:**
A **3D heatmap/surface** showing how Brake Force (or Risk, togglable with buttons) varies across the entire Distance x Speed space. The gradient goes from blue (low) to orange/red (high).

- **Blue region** (bottom-left): far distance + low speed = low brake/risk
- **Orange/Red region** (top-right): near distance + high speed = high brake/risk

Toggle buttons: **Brake** | **Risk** — switch between viewing brake force surface and risk surface.

**Why it matters for viva:**
> This is a 2D projection of the 3D fuzzy inference surface. It visually proves that the fuzzy system produces smooth, continuous outputs with no discontinuities. The gradient transition from blue to red shows how the system smoothly intensifies its response as danger increases.

---

## PAGE 3: ABOUT

This page serves as **documentation** of the system architecture:

---

### 1. THREE INFO CARDS (Top)

**Card 1: FUZZY INPUTS**
- Distance: 0-100 m
- Rel. Speed: 0-100 km/h
- TTC: derived feature (s)
- MF: Trapezoidal (boundary)
- MF: Triangular (interior)

**Card 2: FUZZY OUTPUTS**
- Brake Force (%)
- Collision Risk (%)
- Driver Alert (%)
- Defuzz: Centroid method
- Fallback: Physics-based

**Card 3: HYBRID ENGINE**
- LSTM: 2-layer + BatchNorm
- Features: d, s, TTC (3-dim)
- Seq length: 10 timesteps
- Near: 70% Fuzzy + 30% LSTM
- Far: 60% Fuzzy + 40% LSTM

**Why it matters for viva:**
> These three cards summarize the entire architecture in one view. If your examiner asks "tell me about the system architecture," you can point to these three cards and explain: (1) inputs use trapezoidal/triangular MFs, (2) outputs use centroid defuzzification with physics fallback, (3) hybrid engine uses 2-layer LSTM with adaptive weighting (70/30 near, 60/40 far).

---

### 2. MAMDANI RULE BASE — 9 RULES (Bottom)

**What it shows:**
A complete table of all 9 Mamdani rules with full details:

| RULE | IF DISTANCE | IF SPEED | THEN BRAKE | THEN RISK | THEN ALERT | INTERPRETATION |
|------|-------------|----------|------------|-----------|------------|----------------|
| R01 | near | high | high | high | high | Emergency brake |
| R02 | near | medium | high | high | medium | Hard brake |
| R03 | near | low | medium | medium | medium | Cautious |
| R04 | medium | high | medium | high | high | Moderate brake + alert |
| R05 | medium | medium | medium | medium | medium | Normal traffic |
| R06 | medium | low | low | medium | low | Light brake |
| R07 | far | high | medium | medium | high | Speed warning |
| R08 | far | medium | low | low | low | Safe — cruise |
| R09 | far | low | low | low | low | Very safe |

**Why it matters for viva:**
> This is the complete rule base with human-readable interpretations. This is one of the most likely viva questions — "explain your fuzzy rules." You can point to this table and walk through the logic: near+high = emergency, far+low = very safe, and everything in between follows physical intuition.

---

## SUMMARY: How to Explain the Website in Your Viva

**If asked "What does your website do?":**
> "Our website is an interactive dashboard called Synapse Drive that demonstrates the complete hybrid Fuzzy-LSTM ADAS system. It has three pages:
> 1. **Dashboard** — users adjust distance and speed sliders, click Compute, and see all outputs in real-time: fuzzy brake/risk/alert, LSTM probability, hybrid risk with category, TTC, a radar chart, live membership function plots, which rules fired and with what strength, and a prediction history chart.
> 2. **Analysis** — shows scenario comparisons across 6 predefined driving situations and a 3D inference surface showing how outputs vary smoothly across the entire input space.
> 3. **About** — documents the system architecture: fuzzy inputs/outputs, LSTM hybrid engine parameters, and the complete 9-rule Mamdani rule base with interpretations."

**If asked "What is the most important feature?":**
> "The Rule Firing table on the Dashboard. It shows exactly WHICH fuzzy rules activated and with WHAT strength for any given input. This is the key advantage of using Mamdani fuzzy logic — the decisions are transparent and explainable, which is critical for safety-certified ADAS systems."
