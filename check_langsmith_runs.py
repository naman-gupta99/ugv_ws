from dotenv import load_dotenv
from pathlib import Path
import os
from datetime import datetime, timedelta, timezone

env_path = Path('/home/ws/ugv_ws/src/ugv_main/ugv_tools/ugv_tools/agent/.env')
load_dotenv(env_path)

from langsmith import Client

project = os.getenv('LANGSMITH_PROJECT')
print('Project:', project)
client = Client()
window_start = datetime.now(timezone.utc) - timedelta(minutes=30)
print('Querying runs since', window_start)
# LangSmith enforces a maximum limit of 100 per request
runs = list(client.list_runs(project_name=project, run_type='llm', start_time=window_start, limit=100))
print('Total recent LLM runs:', len(runs))
for r in runs[:20]:
    extra = getattr(r, 'extra', None)
    meta = (extra or {}).get('metadata') if extra else None
    print('id=', getattr(r,'id',None), 'start=', getattr(r,'start_time',None), 'metadata=', meta)
