@echo off
python agent.py --ticker SRVCABLE --documents research_pack --variant naive --output run1_naive.md
python agent.py --ticker SRVCABLE --documents research_pack --variant entity-guard --output run2_entity_guard.md
python agent.py --ticker SRVCABLE --documents research_pack --variant conflict-aware --output run3_conflict_aware.md
