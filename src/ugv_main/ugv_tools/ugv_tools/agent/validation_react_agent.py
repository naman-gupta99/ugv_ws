import json
import os
from datetime import datetime
import time
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langchain_core.messages import (
    messages_to_dict,
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from typing import TypedDict, Annotated, Sequence

from .audit_toolset import *
from .models import Models

def system_message(content: str, flag: bool = False) -> BaseMessage:
    if flag:
        return SystemMessage(content=content)
    return HumanMessage(content=content)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    feedback: str


class ValidationReactAgent:
    def __init__(self):
        
        
        self.prompt = """
        You are a rover navigation AI. Your mission is to visit **every integer grid point** inside the rectangular target area.
        
        
WORLD & GOAL:
- The rover moves on an integer (x, y) grid.
- The target area is inclusive and given by:
  target_area = {{'x_min': <>, 'x_max': <>, 'y_min': <>, 'y_max': <>}}
  Visit every (x, y) where x_min ≤ x ≤ x_max and y_min ≤ y ≤ y_max.
- mission_complete is True only after all required points have been visited.
        
        """
        
        
        self.tools = [move_ahead, move_back, move_left, move_right]
        self.model = Models().get_model("llama-3.3-70b").bind_tools(self.tools)

        # Add memory to maintain conversation state
        self.memory = InMemorySaver()

        self.agent = self.build_agent()

        self.config = {
            "configurable": {
                "thread_id": f"audit_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "recursion_limit": 200,
        }

        # Metrics store
        self.metrics = {
            # Turn/iteration counts
            "agent_turns": 0,
            "validation_turns": 0,
            "tools_transitions": 0,   # times we routed to tools (attempted tool execution)
            "graph_loops": 0,         # times the loop cycled (counted at validation)
            "graph_recursion_errors": 0,

            # Tool usage quality
            "tool_calls_present": 0,
            "tool_calls_missing_fields": 0,
            "tool_calls_nonexistent": 0,
            "tool_calls_with_args": 0,
            "tool_calls_invalid_json": 0,
            "mentioned_tool_but_not_called": 0,
            "no_tool_calls_made": 0,

            # Outcomes
            "missions_completed": 0,

            # Time
            "start_time": None,
            "end_time": None,
            "duration_sec": None,
        }

    def build_agent(self):

        def agent_node(state: AgentState):
            """The main agent node that processes messages and generates responses."""
            # Metrics: count agent turns
            self.metrics["agent_turns"] += 1
            messages = state["messages"]

            # Call the LLM
            try:
                response = self.model.invoke(messages)
            except Exception as e:
                # Return error message if LLM call fails
                response = AIMessage(content=f"Error calling model: {str(e)}")

            # Clear feedback after using it
            return {"messages": [response], "feedback": ""}

        def validation_node(state: AgentState):
            """The validation node that checks the agent's last message for correctness."""
            # Metrics: count validation turns and loops
            self.metrics["validation_turns"] += 1
            self.metrics["graph_loops"] += 1
            messages = state["messages"]
            last_message = messages[-1]

            try:
                if audit_state_instance.is_mission_complete():
                    self.metrics["missions_completed"] += 1
                    return {"messages": system_message("Mission complete! Congratulations on completing all objectives."), "feedback": "Mission complete"}
                # Check if there are tool calls
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    self.metrics["tool_calls_present"] += 1
                    # Validate tool calls
                    for tool_call in last_message.tool_calls:
                        # Check if tool call has required fields
                        if not tool_call.get("name") or not tool_call.get("id"):
                            self.metrics["tool_calls_missing_fields"] += 1
                            return {"messages": system_message("Tool call is missing required fields (name or id). Please provide a complete tool call."), "feedback": ""}

                        # Check if the tool exists
                        tool_names = [tool.name for tool in self.tools]
                        if tool_call["name"] not in tool_names:
                            self.metrics["tool_calls_nonexistent"] += 1
                            return {"messages": system_message(f"Tool '{tool_call['name']}' does not exist. Available tools: {tool_names}"), "feedback": ""}

                        # Check if arguments are valid JSON and empty (movement tools take no args)
                        try:
                            args = tool_call.get("args", {})
                            if args:  # Movement tools should have no arguments
                                self.metrics["tool_calls_with_args"] += 1
                                return {"messages": system_message(f"Tool '{tool_call['name']}' should not have arguments. Call it as {tool_call['name']}() with no parameters."), "feedback": ""}
                            json.dumps(args)  # Validate JSON format
                        except Exception:
                            self.metrics["tool_calls_invalid_json"] += 1
                            return {"messages": system_message("Tool call arguments are not valid JSON format."), "feedback": ""}
                    return {}
                else:
                    tool_names = [tool.name for tool in self.tools]
                    if any(
                        tool_name in last_message.content.lower()
                        for tool_name in tool_names
                    ):
                        self.metrics["mentioned_tool_but_not_called"] += 1
                        return {"messages": system_message("You mentioned a tool name but did not call it correctly. Please use the proper tool calling format."), "feedback": ""}
                    self.metrics["no_tool_calls_made"] += 1
                    return {"messages": system_message("No tool calls were made. Please use the available tools to proceed."), "feedback": ""}

            except Exception as e:
                return {"messages": system_message(f"Error during validation: {str(e)}. Please try again."), "feedback": ""}

        def should_continue(state: AgentState):
            """Determine the next node based on feedback and tool calls."""
            if state.get("feedback"):
                # If feedback indicates mission complete, end the process
                if "Mission complete" in state.get("feedback", ""):
                    audit_state_instance.plot_path()
                    return END
                return "agent"

            return "tools"

        builder = StateGraph(AgentState)

        builder.add_node("agent", agent_node)
        builder.add_node("validation", validation_node)
        builder.add_node("tools", ToolNode(self.tools))

        builder.set_entry_point("agent")
        builder.add_edge("agent", "validation")
        def _route_after_validation(state: AgentState):
            # Custom router to track transitions
            decision = should_continue(state)
            if decision == "tools":
                self.metrics["tools_transitions"] += 1
            return decision

        builder.add_conditional_edges(
            "validation",
            _route_after_validation,
            {
                "agent": "agent",
                "tools": "tools",
                END: END,
            },
        )
        builder.add_edge("tools", "agent")

        return builder.compile(checkpointer=self.memory)

    def execute(self):
        try:
            print("=== Starting Continuous Audit Bot Mission ===")
            # Start timer
            self.metrics["start_time"] = time.time()

            # Initial comprehensive message
            try:
                current_state = audit_state_instance.get_state()
            except Exception as e:
                print(f"Warning: Could not get initial state: {e}")
                current_state = "State unavailable"

            comprehensive_message = f"""
{self.prompt}

CURRENT STATE (JSON):
{current_state}

AVAILABLE TOOLS (use these exact function names with NO parameters):
- move_ahead()  - moves rover up (y+1)
- move_back()   - moves rover down (y-1)
- move_left()   - moves rover left (x-1)
- move_right()  - moves rover right (x+1)

CRITICAL INSTRUCTIONS:
1) ALWAYS use proper tool-calling format — do NOT write tools as plain text.
   Call exactly one tool per turn, e.g., move_right()
2) These functions take NO parameters.
3) Plan your path efficiently to cover the entire rectangle.
   IMPORTANT: x moves (left/right) are expensive — minimize them. y moves (up/down) are cheap — maximize them.
   Hint: A column-by-column serpentine sweep keeps x changes to a minimum:
     - If not at (x_min, y_min), first navigate there.
     - For x from x_min to x_max (inclusive):
         - Sweep bottom→top on even columns ((x - x_min) % 2 == 0): move up (y+1) until y == y_max.
         - Sweep top→bottom on odd columns: move down (y-1) until y == y_min.
         - Only after fully sweeping the current column, step right once (x+1) to the next column.
     - Never move right/left in the middle of a column sweep; finish the entire column first.
4) After each move, read the returned state to track progress.
5) STOP when mission_complete == true. If complete, do not call any tool.
6) Make sure to reason about the next best move based on current position and remaining points.

Start by analyzing the current state, choose the immediate next step of an efficient coverage plan, and make your first move.
Your response MUST include exactly one tool call in the correct format (unless mission_complete is already true).
"""

            print("Sending initial instructions...")
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=comprehensive_message)]},
                config=self.config,
            )

            if response.get("messages"):
                final_message = response["messages"][-1]
                print(f"Final response: {final_message.content}")
            else:
                print("No final message received")

            # Save conversation with error handling
            try:
                tests_dir = "tests"
                if not os.path.exists(tests_dir):
                    os.makedirs(tests_dir)

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                file_path = os.path.join(tests_dir, f"continuous_test_{timestamp}.json")

                serializable_response = {
                    "messages": messages_to_dict(response["messages"])
                }

                with open(file_path, "w") as f:
                    json.dump(serializable_response, f, indent=2)

                print(f"Complete conversation saved to {file_path}")
            except Exception as e:
                print(f"Warning: Could not save conversation: {e}")

        except GraphRecursionError:
            self.metrics["graph_recursion_errors"] += 1
            print("Agent stopped due to max iterations in a single invoke.")
        except Exception as e:
            print(f"Error running continuous audit bot: {e}")
        finally:
            # End timer
            self.metrics["end_time"] = time.time()
            self.metrics["duration_sec"] = round(self.metrics["end_time"] - self.metrics["start_time"], 3) if self.metrics["start_time"] else None
            audit_state_instance.plot_path()
            audit_state_instance.print_metrics()
            # Print agent metrics
            self.print_agent_metrics()

    def get_graph(self):
        return self.agent.get_graph()
    
    def print_agent_metrics(self):
        """Print a concise summary of agent run metrics, including graph recursion and tool usage quality."""
        m = self.metrics
        print("\n=== Agent Run Metrics ===")
        # Timing
        if m.get("duration_sec") is not None:
            print(f"Runtime: {m['duration_sec']} sec")
        # Looping / recursion
        print(f"Graph loops (validation passes): {m['graph_loops']}")
        print(f"Agent turns: {m['agent_turns']} | Validation turns: {m['validation_turns']}")
        print(f"Transitions to tools (tool attempts): {m['tools_transitions']}")
        print(f"Graph recursion errors: {m['graph_recursion_errors']}")

        # Tool call quality / format issues
        invalid_total = (
            m['tool_calls_missing_fields']
            + m['tool_calls_nonexistent']
            + m['tool_calls_with_args']
            + m['tool_calls_invalid_json']
        )
        print(f"Tool calls present: {m['tool_calls_present']}")
        print(f"  - Invalid tool call attempts: {invalid_total}")
        if invalid_total:
            print(f"    • Missing fields: {m['tool_calls_missing_fields']}")
            print(f"    • Nonexistent tool: {m['tool_calls_nonexistent']}")
            print(f"    • Arguments not allowed: {m['tool_calls_with_args']}")
            print(f"    • Invalid JSON args: {m['tool_calls_invalid_json']}")
        print(f"Mentioned tool but not called: {m['mentioned_tool_but_not_called']}")
        print(f"No tool calls made: {m['no_tool_calls_made']}")

        # Outcome
        print(f"Missions completed (this run): {m['missions_completed']}")

