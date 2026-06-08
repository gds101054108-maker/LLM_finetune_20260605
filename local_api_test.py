import json
import requests
import os
import re
# 配置项（仅需修改此处）
API_BASE="" #找培训人员要
MODEL_NAME="qwen3.7-max"# 可替换为上面列表中支持的任意模型
API_KEY = os.getenv("LOCAL_API_KEY", "")
TEST_DATA_PATH = "data/intent_test.json"
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# JSON提取容错函数（保持不变）
def extract_json(text):
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1].strip()
        else:
            return None
    # 补全缺失的括号和引号
    while json_str.count('{') > json_str.count('}'):
        json_str += '}'
    if json_str.count('"') % 2 != 0:
        json_str += '"'
    try:
        return json.loads(json_str)
    except:
        return None

# 构造API请求（保持不变）
def chat_completion(messages):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.01, # 分类任务用极低温度，保证结果稳定可复现
        "max_tokens": 512,
        "enable_thinking": False, # 显式关闭思考模式，强制直接输出结果，提升响应速度
        "stream": False,

    }
    response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")
        return None

# 加载测试集（适配Alpaca格式，去掉外层test_cases取数）
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

total = 0
correct = 0
error_cases = []

# 逐样本测试（去掉原有多轮上下文拼接逻辑，直接用样本内置的上下文和输入）
for sample_idx, sample in enumerate(test_data):
    # 解析样本的真实标签
    true_output = json.loads(sample["output"])
    true_intent = true_output["predicted_intent"]
    current_round = true_output["round"]
    customer_input = true_output["customer_input"]
    
    # 直接用样本内置的系统提示词和输入构造请求（和训练时的prompt完全一致）
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": sample["input"]}
    ]
    
    # 调用API
    response = chat_completion(messages)
    print(sample_idx,response)
    if not response:
        continue
    # 解析预测结果
    result = extract_json(response)
    if not result or "predicted_intent" not in result:
        error_cases.append({
            "sample_idx": sample_idx,
            "round": current_round,
            "customer_input": customer_input,
            "true_intent": true_intent,
            "response": response,
            "error_reason": "JSON解析失败"
        })
        total += 1
        continue
    # 对比结果
    predicted_intent = result["predicted_intent"]
    total += 1
    if predicted_intent == true_intent:
        correct += 1
    else:
        error_cases.append({
            "sample_idx": sample_idx,
            "round": current_round,
            "customer_input": customer_input,
            "true_intent": true_intent,
            "predicted_intent": predicted_intent,
            "error_reason": "意图识别错误"
        })

# 输出统计结果
print(f"测试完成！总样本数：{total}，正确数：{correct}，准确率：{correct/total*100:.2f}%")
# 输出错误案例
print(f"错误案例数：{len(error_cases)}，详细错误信息已保存到error_cases.json")
with open("error_cases.json", "w", encoding="utf-8") as f:
    json.dump(error_cases, f, ensure_ascii=False, indent=2)
