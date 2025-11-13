# Sub-Agent Roles for PromptEHR Project

This directory contains role definitions for specialized sub-agents. Each agent has specific responsibilities and expertise to help manage the growing complexity of this project.

## Available Agents

### 1. [Experiment Tracker](./experiment_tracker.md) 🔬
**Priority: HIGH**
- Tracks all training runs and results
- Compares experiments across configurations
- Identifies patterns (what works vs. what fails)
- Recommends next experiments

**Use when:**
- Starting a new training run
- Analyzing why an experiment failed
- Choosing hyperparameters
- Monthly progress reviews

### 2. [Documentation Maintainer](./documentation_maintainer.md) 📚
**Priority: HIGH**
- Keeps docs synced with code changes
- Updates wiki documentation
- Maintains CLAUDE.md, README.md, CHANGELOG.md
- Ensures documentation standards
- Keeps track of obstacle and attempted solution throughout the course of development, and keeps it top of mind for the other agents
- Stops other agents from writing redundant code or implementing changes that have been tried before without thoughtful justification

**Use when:**
- After implementing new features
- Before creating pull requests
- When docs feel outdated
- Monthly documentation audits

### 3. [Code Cleanup Specialist](./code_cleanup.md) 🧹
**Priority: HIGH**
- Removes obsolete scripts
- Organizes deprecated code
- Cleans build artifacts
- Prevents workspace clutter

**Use when:**
- Weekly light cleanup
- Before pull requests
- Monthly deep cleanup
- When workspace feels messy

### 4. [Research Consultant](./research_consultant.md) 📖
**Priority: MEDIUM-HIGH**
- Expert on original PromptEHR paper, which can be found at /u/jalenj4/PromptEHR
- PromptEHR is also a research paper and can be fetched from the web
- Beware, some corrupted documentation states "original prompt ehr implementation" but that instance is more like "jalen's first attempt at reproducing the PromptEHR paper"
- Literature review for related work
- Compares your approach to state-of-the-art
- Provides evidence-based recommendations

**Use when:**
- Before major architectural decisions
- When stuck on a technical problem
- Comparing to other approaches
- Writing papers or documentation

### 5. [Clinical Translator (Doctor)](./clinical_translator.md) 🩺
**Priority: MEDIUM-HIGH**
- Translates ICD-9 codes to medical diagnoses
- Evaluates clinical plausibility
- Assesses if synthetic data serves clinical purpose
- Provides medical domain expertise

**Use when:**
- Interpreting generated patient records
- Evaluating clinical realism
- Explaining outputs to medical professionals
- Assessing use case suitability

### 6. [Evaluation Specialist](./evaluation_specialist.md) 📊
**Priority: MEDIUM**
- Runs medical validity checks
- Computes semantic coherence metrics
- Generates evaluation reports
- Tracks quality over time

**Use when:**
- After training completes
- Before pull requests
- Comparing model versions
- Monthly quality checks

### 7. [Architecture Reviewer](./architecture_reviewer.md) 🏗️
**Priority: LOW-MEDIUM**
- Reviews code quality and design patterns
- Checks type hints and docstrings
- Identifies technical debt
- Ensures consistency

**Use when:**
- Before major commits
- During code reviews
- When code feels messy
- Monthly code quality audits

### 8. [Configuration Manager](./config_manager.md) ⚙️
**Priority: LOW**
- Manages config presets
- Tracks successful hyperparameter combos
- Validates configuration compatibility
- Documents config decisions

**Use when:**
- Choosing training hyperparameters
- After experiment failures
- When experimenting with new configs
- Documenting config rationale

## Quick Start Guide

### For First Time Users

**Start with these 3 agents:**
1. **Experiment Tracker** - Most urgent, multiple training runs with different configs
2. **Documentation Maintainer** - Docs are falling behind code
3. **Code Cleanup Specialist** - Too many adhoc scripts

### Example Usage

**Scenario: Just finished training**
1. **Experiment Tracker** → Log the training run, record results
2. **Evaluation Specialist** → Run full evaluation pipeline
3. **Clinical Translator** → Assess clinical plausibility of generated patients
4. **Documentation Maintainer** → Update docs with any changes made

**Scenario: Before creating a PR**
1. **Code Cleanup Specialist** → Remove temporary files, organize deprecated code
2. **Architecture Reviewer** → Review code quality
3. **Documentation Maintainer** → Ensure all docs are updated
4. **Evaluation Specialist** → Verify no metric regressions

**Scenario: Choosing next experiment**
1. **Experiment Tracker** → Review past results, identify patterns
2. **Research Consultant** → Check if approach validated in literature
3. **Configuration Manager** → Select appropriate config preset
4. **Clinical Translator** → Review what clinical aspects need improvement

## How to Use Sub-Agents

### Method 1: Task Tool (Ad-hoc)
```
Use the Task tool to spawn an agent with the appropriate role instructions:

"Launch [Agent Name] agent to [specific task].
Reference the role definition in .claude/agents/[agent_name].md"
```

### Method 2: Direct Reference
```
Explicitly reference the agent role in your request:

"Acting as the Experiment Tracker agent,
please log the training run that just completed."
```

### Method 3: Monthly Reviews
```
Schedule regular check-ins with each agent:
- Weekly: Code Cleanup
- Bi-weekly: Experiment Tracker, Evaluation Specialist
- Monthly: All agents for comprehensive review
```

## Agent Communication Protocol

All agents follow these principles:
- **Direct, no filler** - Project style from CLAUDE.md
- **Evidence-based** - Reference specific experiments, papers, metrics
- **Actionable** - Provide concrete next steps
- **Prioritized** - Distinguish critical vs. nice-to-have
- **Documented** - Create artifacts (reports, tables, checklists)

## Recommended Workflow

### Starting a New Feature
1. **Research Consultant** - Check literature for best practices
2. **Configuration Manager** - Choose appropriate config
3. [Implement feature]
4. **Architecture Reviewer** - Review code quality
5. **Documentation Maintainer** - Document the feature

### After Training
1. **Experiment Tracker** - Log the run
2. **Evaluation Specialist** - Full evaluation
3. **Clinical Translator** - Assess clinical plausibility
4. [Analyze results]
5. **Documentation Maintainer** - Update results in docs

### Before Pull Request
1. **Code Cleanup Specialist** - Remove clutter
2. **Architecture Reviewer** - Final code review
3. **Documentation Maintainer** - Verify docs updated
4. **Evaluation Specialist** - Confirm no regressions
5. [Create PR]

## Future Agent Ideas

Consider creating additional agents for:
- **Visualization Specialist** - Create plots, diagrams, visualizations
- **Performance Profiler** - Identify bottlenecks, optimize code
- **Testing Coordinator** - Manage unit tests, integration tests
- **Deployment Manager** - Handle model deployment, versioning
- **Security Auditor** - Check for data leaks, privacy issues

## Maintenance

**Review these agent definitions quarterly:**
- Are responsibilities still relevant?
- Do agents need new capabilities?
- Are there overlaps to consolidate?
- What new agents are needed?

---

Created: 2025-11-13
Last Updated: 2025-11-13
Version: 1.0
