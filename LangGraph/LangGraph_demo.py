from dotenv import load_dotenv
import os
import yaml
import csv
from datetime import datetime
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ====================== 工具函数 ======================
def load_yaml_config(config_path="config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 {config_path} 不存在，请检查路径！")
    except Exception as e:
        raise RuntimeError(f"yaml配置读取失败: {str(e)}")

def write_log_to_csv(log_data: dict, csv_path: str):
    headers = [
        "time", "user_query", "total_call_times",
        "input_tokens", "output_tokens", "total_tokens",
        "cost_usd", "final_answer"
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

# ====================== State 状态定义 ======================
class State(TypedDict):
    user_query: str
    llm_answer: str
    retry_count: int
    max_retry: int
    min_answer_length: int
    answer_history: List[str]       # 本轮图内部重试的多版回答
    conversation_memory: List[dict] # ✅全部对话记忆，交给checkpoint保存
    input_tokens: int  #
    output_tokens: int  #

# ====================== Node 节点 ======================
def call_llm(state: State):
    history_text = ""
    if state["answer_history"]:
        history_text = "【本轮重试历史回答记录】\n"
        for idx, ans in enumerate(state["answer_history"], 1):
            history_text += f"{idx}. {ans}\n"
    messages = list(state["conversation_memory"])
    messages.append({"role": "user", "content": state["user_query"]})
    sys_prompt = f"""
{history_text}
请给出回答，回答不少于{state['min_answer_length']}个字。
"""
    messages.append({"role": "system", "content": sys_prompt})
    print(f"\n===== 第 {state['retry_count']+1} 次调用模型 =====")
    resp = llm.invoke(messages)
    full_content = resp.content
    # 模拟流式逐字输出
    for char in full_content:
        print(char, end="", flush=True)
    print("\n")
    new_ans_history = state["answer_history"].copy()
    new_ans_history.append(full_content)
    usage = resp.usage_metadata or {}
    new_memory = state["conversation_memory"].copy()
    new_memory.append({"role": "user", "content": state["user_query"]})
    new_memory.append({"role": "assistant", "content": full_content})

    # ✅累加token，重试循环多次调用LLM时统计准确
    add_in = usage.get("input_tokens", 0)
    add_out = usage.get("output_tokens", 0)
    total_input = state["input_tokens"] + add_in
    total_output = state["output_tokens"] + add_out

    return {
        "llm_answer": full_content,
        "retry_count": state["retry_count"] + 1,
        "answer_history": new_ans_history,
        "conversation_memory": new_memory,
        "input_tokens": total_input,
        "output_tokens": total_output
    }

# ====================== 条件路由函数 ======================
def route_check_answer(state: State):
    answer = state["llm_answer"]
    current_retry = state["retry_count"]
    max_retry = state["max_retry"]
    min_len = state["min_answer_length"]
    is_ok = len(answer) >= min_len

    if is_ok:
        print("✅ 代码校验：回答合格，结束本轮问答")
        return "end_flag"
    elif current_retry < max_retry:
        print(f"❌ 代码校验：回答太短，需要重试。当前次数{current_retry}/{max_retry}")
        return "retry_flag"
    else:
        print(f"⚠️ 达到最大重试次数{max_retry}，直接结束本轮问答")
        return "end_flag"

# ====================== 构建图 + 绑定Checkpoint内存存储 ======================
def build_graph(cfg):
    builder = StateGraph(State)
    builder.add_node("call_llm", call_llm)
    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges(
        "call_llm",
        route_check_answer,
        {
            "retry_flag": "call_llm",
            "end_flag": END
        }
    )
    # 内存检查点，会话状态自动保存；生产替换为 SQLite / Redis
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    try:
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 在 .env 文件中未找到！")
        cfg = load_yaml_config("config.yaml")

        llm = ChatOpenAI(
            api_key=api_key,
            base_url=cfg["llm"]["base_url"],
            model=cfg["llm"]["model_name"],
            temperature=cfg["llm"]["temperature"],
            streaming=cfg["llm"]["streaming"]
        )

        graph = build_graph(cfg)
        # thread_id代表会话，同一个thread_id维持一套对话记忆；更换id=新开会话
        thread_config = {"configurable": {"thread_id": "chat_session_001"}}

        print("==== LangGraph Checkpoint会话Demo ====")
        print("输入问题对话，输入 exit / quit 退出\n")

        while True:
            user_input = input("👤 请输入问题：").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("👋 程序退出")
                break
            if not user_input:
                print("⚠️输入不能为空，请重新输入！\n")
                continue

            # 获取线程快照，判断是否是全新会话
            snapshot = graph.get_state(thread_config)
            if snapshot.values:
                # 已有历史会话，不要传入conversation_memory！防止覆盖记忆
                input_state = {
                    "user_query": user_input,
                    "retry_count": 0,
                    "max_retry": cfg["retry"]["max_retry"],
                    "min_answer_length": cfg["retry"]["min_answer_length"],
                    "answer_history": [],
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            else:
                # 全新会话，初始化空对话记忆
                input_state = {
                    "user_query": user_input,
                    "retry_count": 0,
                    "max_retry": cfg["retry"]["max_retry"],
                    "min_answer_length": cfg["retry"]["min_answer_length"],
                    "answer_history": [],
                    "conversation_memory": [],
                    "input_tokens": 0,
                    "output_tokens": 0
                }

            final_state = graph.invoke(input_state, config=thread_config)
            input_tokens = final_state["input_tokens"]
            output_tokens = final_state["output_tokens"]
            total_tokens = input_tokens + output_tokens
            input_cost = input_tokens / 1_000_000 * cfg["price"]["input_token_price"]
            output_cost = output_tokens / 1_000_000 * cfg["price"]["output_token_price"]
            total_cost = input_cost + output_cost

            print("\n======= 本轮汇总结果 =======")
            print(f"本轮内部调用次数：{final_state['retry_count']}")
            print(f"本轮输入token：{input_tokens}")
            print(f"本轮输出token：{output_tokens}")
            print(f"本轮总token消耗：{total_tokens}")
            print(f"本轮预估费用(USD): {total_cost:.6f}")
            log_record = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_query": final_state["user_query"],
                "total_call_times": final_state["retry_count"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(total_cost, 6),
                "final_answer": final_state["llm_answer"]
            }
            write_log_to_csv(log_record, cfg["log"]["csv_file"])
            print(f"📝 本轮日志已保存到 {cfg['log']['csv_file']}")
            print("\n----------------------------------------\n")


    except Exception as e:
        print(f"\n❌ 程序异常：{type(e).__name__}: {str(e)}")
