"""Paths"""
import os

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_root"))
BASE = DATA_ROOT
WORK = os.path.join(BASE, "work")
ADD = os.path.join(BASE, "20261013,4 AI hDF P5,15 migration(additional_data)")
os.makedirs(WORK, exist_ok=True)
