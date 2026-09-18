from dotenv import load_dotenv
import os
import yaml
import csv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAIError
from langchain_community.callbacks import OpenAICallbackHandler

# ---------------稳定获取当前脚本所在的文件夹路径---------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # -------------------------- 1. 加载 .env 密钥 --------------------------
    try:
        env_path = os.path.join(BASE_DIR, ".env")
        load_dotenv(env_path)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 为空，请检查 .env 文件")
    except FileNotFoundError:
        print(f"【错误】找不到 .env 文件，路径：{env_path}")
        return
    except Exception as e:
        print(f"【读取.env失败】{str(e)}")
        return

    # -------------------------- 2. 加载 yaml 配置文件 --------------------------
    cfg_path = os.path.join(BASE_DIR, "config.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # 校验必填配置
        _ = cfg["llm"]["base_url"]
        _ = cfg["llm"]["model"]
        _ = cfg["llm"]["temperature"]
        _ = cfg["prompt"]["system_prompt"]
        _ = cfg["prompt"]["human_template"]
        _ = cfg["demo"]["default_topic"]

        enable_token_stats = cfg["demo"].get("enable_token_stats", True)
        enable_cost_calc = cfg["demo"].get("enable_cost_calc", True)
        enable_csv_log = cfg["demo"].get("enable_csv_log", True)

        price_in = cfg["demo"]["price_input_per_million"]
        price_out = cfg["demo"]["price_output_per_million"]
    except FileNotFoundError:
        print(f"【错误】找不到 config.yaml，路径：{cfg_path}")
        return
    except yaml.YAMLError:
        print("【错误】config.yaml 语法错误，请检查yaml格式！")
        return
    except KeyError as e:
        print(f"【错误】config.yaml 缺少配置项：{e}")
        return
    except Exception as e:
        print(f"【读取配置文件失败】{str(e)}")
        return

    # -------------------------- 3. 初始化LLM、Prompt、Chain、回调器 --------------------------
    try:
        callback_handler = OpenAICallbackHandler()
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=cfg["llm"]["base_url"],
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"]
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", cfg["prompt"]["system_prompt"]),
            ("human", cfg["prompt"]["human_template"])
        ])
        chain = prompt | llm | StrOutputParser()
    except Exception as e:
        print(f"【初始化LLM/Chain失败】{str(e)}")
        return

    # -------------------------- 4. 流式调用 + 异常捕获 + token统计 + 费用 + 日志 --------------------------
    try:
        topic = input("请输入提问话题(直接回车使用默认话题): ") or cfg["demo"]["default_topic"]
        print(f"\n正在调用模型，提问主题：{topic}\n")
        print("===== 模型返回结果 =====")

        stream = chain.stream(
            {"topic": topic},
            config={"callbacks": [callback_handler]}
        )
        for chunk in stream:
            print(chunk, end="", flush=True)
        print("\n")

        # 读取token数据
        prompt_tokens = callback_handler.prompt_tokens
        completion_tokens = callback_handler.completion_tokens
        total_tokens = callback_handler.total_tokens

        # 打印token统计
        if enable_token_stats:
            print("===== Token消耗统计 =====")
            print(f"输入token(prompt): {prompt_tokens}")
            print(f"输出token(completion): {completion_tokens}")
            print(f"总token(total): {total_tokens}")

        # 费用计算
        cost = 0.0
        if enable_cost_calc:
            cost_in = prompt_tokens * price_in / 1_000_000
            cost_out = completion_tokens * price_out / 1_000_000
            cost = cost_in + cost_out
            print("===== 本次调用费用 =====")
            print(f"输入费用：{cost_in:.6f} 元")
            print(f"输出费用：{cost_out:.6f} 元")
            print(f"合计花费：{cost:.6f} 元")

        # 写入CSV日志
        if enable_csv_log:
            log_file = os.path.join(BASE_DIR, "llm_log.csv")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [
                now,
                cfg["llm"]["model"],
                topic,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                round(cost,6)
            ]
            header = ["时间","模型","提问话题","输入token","输出token","总token","花费(元)"]

            file_exists = os.path.isfile(log_file)
            with open(log_file, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(row)
            print(f"\n✅ 日志已保存至：{log_file}")

    except OpenAIError as e:
        print(f"\n【DeepSeek接口调用失败】{str(e)}")
    except Exception as e:
        print(f"\n【流式执行异常】{str(e)}")

if __name__ == "__main__":
    main()
