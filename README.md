# Research Lab

AI-powered computational research loop for engineering experiments.

Research Lab turns a research question into a bounded experimental campaign:

```text
Research Question
       ↓
AI Research Agent
       ↓
Research / Evidence
       ↓
Evidence Gate
       ↓
Hypothesis + Objective
       ↓
Experiment Planner
       ↓
MCP
       ↓
Flowlab / Gmsh
       ↓
Measured Results
       ↓
Validation + Interpretation
       ↓
Next Experiment ↺

What it does
The current implementation focuses on mesh optimization experiments using STEP/STP geometry and Flowlab's Gmsh-based meshing platform.
The agent can:
- Research engineering context and supporting evidence
- Establish an explicit experimental objective
- Plan bounded mesh-resolution experiments
- Run experiments through MCP
- Collect and normalize mesh-quality metrics
- Compare experimental results
- Select the next experiment using deterministic search
- Stop when the objective is satisfied or the experiment budget is exhausted
- Generate a structured research report
The LLM acts as a research supervisor. Experimental continuation is controlled by the deterministic planner rather than by free-form model decisions.
Architecture
                    ┌──────────────────┐
                    │ Research Question│
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │   AI Researcher  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  Evidence Gate   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Experiment       │
                    │ Planner          │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │       MCP        │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Flowlab / Gmsh   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Measured Results │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Validation /     │
                    │ Interpretation   │
                    └────────┬─────────┘
                             │
                             └──────────→ Next Experiment

Scientific Controls
Research Lab separates:
- Evidence — external information used to motivate hypotheses
- Measurements — actual results returned by Flowlab
- Agent interpretation — model-generated reasoning
- Demo assumptions — explicitly labelled user-supplied assumptions
The planner enforces:
- Experiment budgets
- Search bounds
- Duplicate prevention
- Planned-parameter matching
- Terminal stop conditions
- Structured handling of failed experiments
Infrastructure and API errors are not treated as engineering measurements.
Provider Independence
Research
The research layer is provider-independent and supports:
- Bright Data
- Direct Web access
- No research provider
LLM
The LLM layer is provider-independent and supports OpenAI-compatible providers such as Groq, with Anthropic support retained.
The computational research pipeline is not tied to a single model vendor.
Flowlab Integration
Research Lab communicates with Flowlab through its REST API.
It does not import Flowlab internals or directly access its database, files, Celery workers, or container runtime.
Flowlab owns engineering state. Research Lab owns research and campaign state.
Running Locally
Create the environment:
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

Configure .env with the required provider and Flowlab settings.
Start the web UI:
python -m research_lab.ui --port 8765

Run the test suite:
pytest -q

Current Demo
The current demo template is:
Mesh Optimization
A STEP/STP geometry is used to run a bounded mesh-resolution campaign. The system measures mesh quality and element count, learns from previous experiments, and selects the next experiment until the deterministic planner reaches a stopping condition.
Project Status
🚧 Active hackathon project
The architecture is designed to support additional computational research templates and experimental instruments beyond mesh optimization.
