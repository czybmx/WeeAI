import json
import requests
from datetime import datetime
from colorama import Fore, Style, init
from bs4 import BeautifulSoup
from googlesearch import search
import concurrent.futures
import re

# Initialize colorama for cross-platform colored output
init(autoreset=True)

url_generate = "http://localhost:11434/api/generate"
conversation_history = []
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"

print(f"{Fore.YELLOW}Stupid AI Chat Interface 超级人工智障系统{Style.RESET_ALL}")

def get_response_stream(url, data):
    with requests.post(url, json=data, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                try:
                    response_dict = json.loads(line.decode('utf-8'))
                    yield response_dict.get("response", "")
                except json.JSONDecodeError:
                    continue

def build_prompt(conversation_history):
    return "\n".join(f"{entry['role']}\n{entry['content']}" for entry in conversation_history)

def chat_with_model(model, system_prompt, user_input, temperature=0.5):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})
    
    prompt = build_prompt(conversation_history)
    
    data = {
        "model": model,
        "temperature": temperature,
        "role": "assistant",
        "system": system_prompt,
        "prompt": prompt,
        "token": 14000,
        "stream": True
    }
    
    response_generator = get_response_stream(url_generate, data)
    complete_response = ""
    
    for response_chunk in response_generator:
        print(f"{Fore.LIGHTCYAN_EX}{response_chunk}", end='', flush=True)
        complete_response += response_chunk

    conversation_history.append({"role": "assistant", "content": complete_response})
    print("\n\n\n")

    return complete_response

def google_search(query, num_results=3):
    print(f"正在进行Google搜索: {query}")
    try:
        results = list(search(query, num=num_results, stop=num_results))
        return results
    except Exception as e:
        print(f"Google搜索失败: {e}")
        return []

def fetch_webpage_content(url):
    try:
        response = requests.get(url, headers={'User-Agent': user_agent})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:4096]
    except Exception as e:
        print(f"获取 {url} 时出错: {e}")
        return ""

def perform_searches(query):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        google_future = executor.submit(google_search, query)
        google_results = google_future.result()

        if not google_results:
            return "Google搜索没有结果"

        webpage_futures = [executor.submit(fetch_webpage_content, url) for url in google_results[:3]]
        webpage_contents = [future.result() for future in concurrent.futures.as_completed(webpage_futures)]

    search_results = f"Google 结果:\n" + "\n".join(google_results)
    search_results += f"\n\n网页内容:\n" + "\n\n".join(webpage_contents)

    return search_results

def main():
    while True:
        user_input = input(f"INPUT: ")

        if user_input.lower() in ['exit', 'quit']:
            break

        response = chat_with_model(
            "qwen2-7b-instruct-q8_0:latest",
            "你只会问问题，出题目，以无知者角度询问",
            f"请按照标题【{user_input}】进行出题，不需要回答标题，不需要做多余的解释"
        )
        print("\n\n")
        print(f"{Fore.GREEN}########################################################################################")

        netask = chat_with_model(
            "qwen2-7b-instruct-q8_0:latest",
            "请识别该【对话指令】是否需要利用网络查询?(YES/NO)",
            f"""
对话指令:
'{response}'
'''
//请你识别以上【对话指令】是否需要利用网络查询?(只回复YES/NO，不需要解释)//
"""
        )
        print("\n\n")
        print(f"{Fore.GREEN}########################################################################################")

        ingoogle_keyword = chat_with_model(
            "qwen2-7b-instruct-q8_0:latest",
            "你是一个关键词检测器",
            f"""
Infomation From Network:
\\{response}\\
'''
User：'{user_input}'
'''
请按照用户要求内容提取【一个】关键句用于网络搜索，不需要解释，不需要注明，也不需要回答用户的问题。直接返回这个短语。（注明：说直接点就是让你在谷歌搜索框输入的词）
"""
        )

        if netask.strip().lower() == "yes":
            print(f"【Searching with keyword: \"{ingoogle_keyword}\"...】\n")
            search_results = perform_searches(ingoogle_keyword)
            search_results_output = f"(智能搜索引擎：{ingoogle_keyword} 结果：{search_results})\n"
        else:
            search_results_output = "(没有找到合适的关键词进行搜索)\n"

        print("\n\n")
        print(f"{Fore.GREEN}########################################################################################")

        reply1 = chat_with_model(
            "qwen2-7b-instruct-q8_0:latest",
            f"你是一位万物解答师，专门解答任何疑问，有任何问题可以尝试参考来自网络相关的信息:\n{search_results_output}",
            f"请解答全部题目并解释答案原理:\n{response}"
        )
        print("\n\n")
        print(f"{Fore.GREEN}########################################################################################")
        
        reply2 = chat_with_model(
            "qwen2-7b-instruct-q8_0:latest",
            "你是一位万物专家，根据用户问题优化来自大脑的回应与输出",
            f"""
大脑：\n{reply1}
'''
用户：\n{user_input}
"""
        )

if __name__ == "__main__":
    main()
