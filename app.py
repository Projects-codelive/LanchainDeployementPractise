import math
import re
import numexpr
import streamlit as st
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# 1. Set up the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver & Data Search Assistant", page_icon="🧮")
st.title("🧮 Text To Math Problem Solver (LangGraph Edition)")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password") 

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot powered by LangGraph! Ask me any word problem or factual question."}
    ]
    st.rerun()

if not groq_api_key:
    st.info("Please add your Groq API Key to continue")
    st.stop()

# Initialize Groq LLM with explicit reasoning_format="parsed"
llm = ChatGroq(
    groq_api_key=groq_api_key.strip(),
    model="qwen/qwen3.8-27b",
    temperature=0,
    reasoning_format="parsed",
)


# ---------------------------------------------------------
# Helper Functions: Extract <think> Logic & Fix $ Rendering
# ---------------------------------------------------------

def split_think_and_answer(raw_text: str) -> tuple[str, str]:
    """Separate <think>...</think> internal reasoning from the final visible answer."""
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    think_matches = re.findall(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    thinking_text = "\n\n".join(m.strip() for m in think_matches if m.strip())
    clean_answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return thinking_text, clean_answer


def format_for_streamlit(text: str) -> str:
    """Escape currency dollar signs ($100 -> \\$100) so Streamlit doesn't corrupt text as LaTeX."""
    return re.sub(r"\$(?=\d)", r"\\$", str(text))


# ---------------------------------------------------------
# 2. Define Tools (@tool Decorator)
# ---------------------------------------------------------

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia to find factual information, dates, heights, populations, or general knowledge."""
    return wikipedia_wrapper.run(query)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression using numexpr.
    You MUST call this tool for all arithmetic operations.
    Examples of valid inputs: '4 * 2.50', '6 * 0.40', '(10 + 2.4) * 24 * 30', '(8928 - 500) * 1.18'.
    Pass only numbers and standard math operators (+, -, *, /, **)."""
    try:
        cleaned_expr = expression.strip().replace("^", "**")
        result = numexpr.evaluate(
            cleaned_expr,
            global_dict={},
            local_dict={"pi": math.pi, "e": math.e}
        )
        val = result.item() if hasattr(result, "item") else result
        return f"{val}"
    except Exception as e:
        return f"Calculation error: {e}. Please provide a valid numerical expression."


tools = [calculator, wikipedia_search]

# ---------------------------------------------------------
# 3. Create the LangGraph ReAct Agent
# ---------------------------------------------------------

system_instructions = (
    "You are an agentic Math Problem Solver and Data Search Assistant.\n"
    "CRITICAL RULES:\n"
    "1. NEVER calculate numbers in your head. You MUST call the `calculator` tool to compute the math expressions.\n"
    "2. If a question requires factual knowledge (like dates or measurements), call `wikipedia_search` first.\n"
    "3. Wait for the `calculator` tool output, and then generate your final Response strictly based on the tool outputs.\n"
    "4. Format your final Response clearly with point-wise steps and the final result."
)

assistant_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_instructions,
)

# ---------------------------------------------------------
# 4. Initialize Session State
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot powered by LangGraph! Ask me any word problem or factual question."}
    ]

# ---------------------------------------------------------
# 5. Question Input & Live Agent Execution (Top)
# ---------------------------------------------------------

question = st.text_area(
    "Enter your question:",
    "A startup rents 4 cloud GPU servers at $2.50 per hour each and 6 CPU servers at $0.40 per hour each. They run all servers 24 hours a day for 30 days. If their cloud provider gives them a $500 promotional credit and then applies an 18% tax on the remaining bill, what is their final total cost?"
)

if st.button("Find my answer"):
    if question.strip():
        st.session_state["messages"].append({"role": "user", "content": question})

        final_answer = ""

        # Thinking box ONLY shows internal thoughts, tool calls, and tool outputs
        with st.status("🤔 Agent Thinking & Calling Tools...", expanded=True) as status:
            step_num = 0

            # Pass the current question as a fresh turn so old history doesn't stop tool calling
            for chunk in assistant_agent.stream(
                {"messages": [HumanMessage(content=question)]},
                stream_mode="updates"
            ):
                for node_name, node_data in chunk.items():
                    if not isinstance(node_data, dict):
                        continue

                    for msg in node_data.get("messages", []):
                        # 1. Agent Node: Extract internal thoughts and Tool Calls
                        if isinstance(msg, AIMessage):
                            raw_content = msg.content if isinstance(msg.content, str) else ""
                            think_logic, clean_text = split_think_and_answer(raw_content)

                            reasoning_kwarg = (
                                msg.additional_kwargs.get("reasoning_content", "")
                                or msg.response_metadata.get("reasoning_content", "")
                            )
                            combined_thinking = think_logic or reasoning_kwarg

                            # Show internal reasoning in Thinking box
                            if combined_thinking:
                                step_num += 1
                                st.markdown(f"**🧠 Step {step_num}: Agent Internal Thought**")
                                st.info(format_for_streamlit(combined_thinking))

                            # Show Tool Call Request in Thinking box
                            if getattr(msg, "tool_calls", None):
                                for tc in msg.tool_calls:
                                    step_num += 1
                                    st.markdown(f"**🛠️ Step {step_num}: Calling Tool `{tc['name']}`**")
                                    st.code(f"Tool Arguments: {tc['args']}", language="json")

                            # Save the clean post-tool response for the Response box ONLY
                            if clean_text and not getattr(msg, "tool_calls", None):
                                final_answer = clean_text

                        # 2. Tools Node: Show exact Tool Output returned to the Agent
                        elif isinstance(msg, ToolMessage):
                            step_num += 1
                            st.markdown(f"**✅ Step {step_num}: Tool Output from `{msg.name}`**")
                            st.code(f"Result: {msg.content}", language="text")

            status.update(label=f"✅ Completed in {step_num} reasoning/tool steps!", state="complete", expanded=True)

        if not final_answer:
            final_answer = "Could not generate an answer. Please try again."

        formatted_answer = format_for_streamlit(final_answer)
        st.session_state["messages"].append({"role": "assistant", "content": formatted_answer})

        # Response box ONLY shows the final synthesized explanation built from tool outputs
        st.write("### Response:")
        st.success(formatted_answer)
    else:
        st.warning("Please enter a question.")

# ---------------------------------------------------------
# 6. Show Chat History at Last (Bottom of the Page)
# ---------------------------------------------------------

st.divider()
st.subheader("💬 Chat History")

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(str(msg["content"]))