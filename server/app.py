"""HTTP API and static interface. Run: uvicorn server.app:app --port 8000

All JSON. Errors: {"detail": "<what went wrong and what to do>"} with a 4xx or 5xx status.

  GET  /api/models
       -> [ModelInfo dict]                          name, label, native, open_weights, available, note
  GET  /api/datasets
       -> [{"name", "title", "modality", "labels", "n"}]
  GET  /api/senses?model=<name>
       -> [{"bridge", "kind", "modality", "dataset", "card": ReportCard dict}]
  DELETE /api/senses?model=<name>&modality=<modality>
       -> {"removed": bool}
  POST /api/fit                                     body: {"model", "dataset", "modality": "audio", "scope": "closed"|"open"}
       -> {"job_id": str}                           409 if a run is already in progress
  GET  /api/jobs/<job_id>
       -> JobState dict                             id, request, status, events[], cards[], installed, error
  GET  /api/samples?dataset=<name>
       -> [{"id", "label", "name"}]                 held-out example files the interface can offer
  GET  /api/samples/<id>
       -> the audio file
  POST /api/chat                                    multipart form: model, message, files (0..n uploads), sample_ids (0..n)
       -> {"reply": str, "used": [{"bridge", "kind", "source", "detail"}]}
          `used` lists how each file reached the model; a file whose modality has no installed sense is
          reported with kind "none" and is not shown to the model.
  GET  /                                            web/index.html; other files under web/ are served at /static/

The server holds no model itself: it calls sense.models, sense.registry, sense.datasets and agent.jobs.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Sense harness")
