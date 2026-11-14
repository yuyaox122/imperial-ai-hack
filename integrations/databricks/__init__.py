"""Databricks for Sensorium: agent runs and report cards into Delta tables, run records into a Volume, and one
MLflow run per verified bridge. `python -m integrations.databricks --help` for the commands.

Live runs: `agent.jobs.add_listener(integrations.databricks.sink.Listener())` once at server start.
"""
