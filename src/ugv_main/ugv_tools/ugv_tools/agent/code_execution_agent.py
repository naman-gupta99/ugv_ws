import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple
 
from langgraph.errors import GraphRecursionError

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig

from .audit_toolset import audit_state_instance
from .models import Models
from .hints import hints


def _extract_python_code(content: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return content.strip()


class CodeState(TypedDict):
    """State for code generation langgraph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    generated_code: str
    model_name: str
    current_model_index: int


def execute_code_audit(metrics, print_agent_metrics, llm_model_name=None, hint=""):
    """
    Execute code audit using multiple LLM models with langgraph integration.
    
    Args:
        metrics: Metrics dictionary to track performance
        print_agent_metrics: Callback to print metrics
        llm_model_name: Optional specific model to use; if None, tries all models
        hint: Strategy hint to provide to the LLM (e.g., "easy", "medium", "hard")
    """
    print(f"=== Starting Code Execution Audit Bot Mission (Multi-LLM) with hint='{hint}' ===")
    metrics["start_time"] = time.time()

    max_steps = int(os.getenv("UGV_CODE_AGENT_MAX_STEPS", "50"))
    code_dir = os.getenv("UGV_CODE_AGENT_OUTPUT_DIR", "tests")
    os.makedirs(code_dir, exist_ok=True)

    step_count = 0
    used_model_name = ""
    langsmith_thread_id = ""

    def _ensure_langsmith_env():
        if os.getenv("LANGSMITH_TRACING", "false").lower() != "true":
            return
        if not os.getenv("LANGCHAIN_TRACING_V2"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
        if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
        if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

    def _build_langsmith_callbacks():
        if os.getenv("LANGSMITH_TRACING", "false").lower() != "true":
            return None
        project = os.getenv("LANGSMITH_PROJECT")
        if not project:
            return None
        try:
            from langchain_core.tracers.langchain import LangChainTracer

            return [LangChainTracer(project_name=project)]
        except Exception as exc:
            print(f"[Code Agent][LangSmithDiag] tracer init failed: {exc}")
            return None

    def _record_move(move_func):
        nonlocal step_count
        if step_count >= max_steps:
            raise RuntimeError(f"Generated code exceeded max movement steps ({max_steps}).")
        step_count += 1
        metrics["tools_transitions"] += 1
        move_func()
        return audit_state_instance.get_state()

    def move_ahead():
        return _record_move(audit_state_instance.move_ahead)

    def move_back():
        return _record_move(audit_state_instance.move_back)

    def move_left():
        return _record_move(audit_state_instance.move_left)

    def move_right():
        return _record_move(audit_state_instance.move_right)

    def get_state():
        return audit_state_instance.get_state()

    def is_mission_complete():
        return audit_state_instance.is_mission_complete()

    try:
        current_state = audit_state_instance.get_state()
        
        # Get hint if provided
        hint_text = hints.get(hint, "") if hint else ""

        prompt = f"""
You are controlling a rover on an integer grid.

Write ONE Python program that completes the mission from the current state.
The program will be executed immediately after you write it.

CURRENT STATE:
{current_state}

MISSION:
- Visit every integer coordinate inside target_area, inclusive.
- Stop moving once is_mission_complete() returns True.

AVAILABLE FUNCTIONS:
- move_ahead() -> state   # y += 1
- move_back() -> state    # y -= 1
- move_left() -> state    # x -= 1
- move_right() -> state   # x += 1
- get_state() -> dict
- is_mission_complete() -> bool

RULES:
- Return only executable Python code.
- Do not use markdown.
- Do not import modules.
- Do not define or call unavailable APIs.
- Prefer deterministic loops over step-by-step hardcoding.
- The code should finish in at most {max_steps} movement calls.
- Plan your path efficiently to cover the entire rectangle.

IMPORTANT: x moves (left/right) are expensive — minimize them. y moves (up/down) are cheap — maximize them.

{f"STRATEGY HINT:{hint_text}" if hint_text else ""}
"""

        _ensure_langsmith_env()
        model_name = llm_model_name or os.getenv("UGV_CODE_AGENT_MODEL", Models.DEFAULT_MODEL)
        print(f"[Code Agent] Will attempt code generation with model: {model_name}")

        allowed_builtins = {
            "abs": abs,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "set": set,
            "sorted": sorted,
        }
        execution_api = {
            "move_ahead": move_ahead,
            "move_back": move_back,
            "move_left": move_left,
            "move_right": move_right,
            "get_state": get_state,
            "is_mission_complete": is_mission_complete,
        }

        models = Models()
        try:
            model = models.get_model(model_name)
        except Exception as e:
            print(f"[Code Agent] could not get model '{model_name}': {e}")
            # Fallback: try the first available concrete LLM name
            llm_names = models.llm_model_names()
            if llm_names:
                fallback = llm_names[0]
                print(f"[Code Agent] falling back to LLM model: {fallback}")
                try:
                    model = models.get_model(fallback)
                except Exception:
                    # As a last resort, call internal constructor to get a concrete LLM
                    model = models._get_google_model(fallback)
                model_name = fallback
            else:
                raise

        # Prepare tracing callbacks and a unique thread id early so agent_node
        # can ensure the model invocation receives metadata and callbacks.
        thread_id = f"code_agent_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        callbacks = _build_langsmith_callbacks()

        def agent_node(state: CodeState, config: RunnableConfig):
            metrics["agent_turns"] += 1
            # Ensure the model receives a RunnableConfig with our metadata and callbacks
            model_config = dict(config) if config else {}
            model_config.setdefault("configurable", {})
            model_config["configurable"]["thread_id"] = thread_id
            meta = model_config.setdefault("metadata", {})
            meta.update({
                "ugv_agent_thread_id": thread_id,
                "ugv_model_name": model_name,
                "ugv_agent_type": "code_execution",
            })
            if callbacks:
                model_config.setdefault("callbacks", callbacks)

            response = model.invoke([HumanMessage(content=prompt)], config=model_config)
            return {"messages": [response]}

        builder = StateGraph(CodeState)
        builder.add_node("agent", agent_node)
        builder.set_entry_point("agent")
        builder.add_edge("agent", END)

        memory = InMemorySaver()
        graph = builder.compile(checkpointer=memory)

        thread_id = f"code_agent_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        callbacks = _build_langsmith_callbacks()
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "ugv_agent_thread_id": thread_id,
                "ugv_model_name": model_name,
                "ugv_agent_type": "code_execution",
            },
        }
        if callbacks:
            config["callbacks"] = callbacks

        initial_state = {
            "messages": [],
            "generated_code": "",
            "model_name": model_name,
            "current_model_index": 0,
        }
        # Run the graph in a worker thread and enforce a 7-minute timeout to avoid
        # indefinite blocking. Using the synchronous `invoke` path in a separate
        # thread preserves the tracing behavior that was working for other agents.
        langsmith_thread_id = thread_id
        try:
            result = graph.invoke(initial_state, config=config)
        except Exception as e:
            print(f"Error invoking graph: {e}")
            result = None

        # Diagnostic: when LangSmith tracing is enabled, query recent LLM runs
        # and report whether any runs contain our thread id metadata. This helps
        # determine whether traces are being created and whether metadata is set.
        try:
            if os.getenv('LANGSMITH_TRACING', 'false').lower() == 'true' and os.getenv('LANGSMITH_PROJECT'):
                from langsmith import Client

                client = Client()
                window_start = datetime.now(timezone.utc) - timedelta(minutes=15)
                runs = list(client.list_runs(project_name=os.getenv('LANGSMITH_PROJECT'), run_type='llm', start_time=window_start, limit=200))
                matching = [r for r in runs if (r.extra or {}).get('metadata', {}).get('ugv_agent_thread_id') == thread_id]
                print(f"[Code Agent][LangSmithDiag] found {len(runs)} recent llm runs; {len(matching)} match ugv_agent_thread_id={thread_id}")
                for r in matching[:5]:
                    try:
                        print(f"[LangSmithDiag] id={getattr(r, 'id', None)} start={getattr(r, 'start_time', None)} total_tokens={getattr(r, 'total_tokens', None)} extra={getattr(r, 'extra', None)}")
                    except Exception:
                        print("[LangSmithDiag] could not print run details")
        except Exception as exc:
            print(f"[Code Agent][LangSmithDiag] diagnostic error: {exc}")

        last_message = result["messages"][-1] if result.get("messages") else None
        if last_message:
            generated_code = _extract_python_code(getattr(last_message, "content", str(last_message)))
        else:
            generated_code = ""

        if not generated_code:
            print(f"[Code Agent] No code generated by {model_name}")
            return

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(code_dir, f"code_agent_{model_name}_{timestamp}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(generated_code)
            f.write("\n")
        print(f"[Code Agent] Generated code from {model_name} saved to {file_path}")

        try:
            exec_globals = {"__builtins__": allowed_builtins, **execution_api}
            exec_locals = dict(execution_api)
            exec(generated_code, exec_globals, exec_locals)

            if audit_state_instance.is_mission_complete():
                metrics["missions_completed"] += 1
                print(f"[Code Agent] SUCCESS: Mission completed with {model_name}")
                used_model_name = model_name
                langsmith_thread_id = thread_id
            else:
                print(f"[Code Agent] {model_name} executed but mission not complete.")
        except Exception as exec_error:
            print(f"[Code Agent] Code execution failed with {model_name}: {exec_error}")
            audit_state_instance.reset()
    except Exception as e:
        print(f"Error running code execution audit bot: {e}")
    finally:
        metrics["end_time"] = time.time()
        metrics["duration_sec"] = (
            round(metrics["end_time"] - metrics["start_time"], 3)
            if metrics["start_time"]
            else None
        )
        # Store which model was used in metrics
        metrics["used_model"] = used_model_name
        metrics["hint_used"] = hint
        metrics["langsmith_thread_id"] = langsmith_thread_id
        audit_state_instance.plot_path()
        audit_state_instance.print_metrics()
        print_agent_metrics()
