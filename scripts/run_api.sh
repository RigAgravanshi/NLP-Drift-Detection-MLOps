#!/bin/bash
export PYTHONPATH=.
uvicorn src.api.main:app --reload