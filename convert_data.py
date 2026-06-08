import json
import random

# 加载系统提示词，去掉置信度字段说明
with open("intent_recognition_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()
# 移除输出要求里的confidence字段
system_prompt = system_prompt.replace(',\n  "confidence": "<置信度，0-1之间的小数，保留两位小数>"', '')
system_prompt_instruction = system_prompt.split("## 对话上下文")[0].strip()

def convert_to_alpaca(raw_data_path, output_path):
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    alpaca_data = []
    # 处理每个对话案例
    for case in raw_data["test_cases"]:
        case_id = case["case_id"]
        turns = case["dialogue"]
        # 维护当前对话上下文，逐轮累加
        current_context = []
        
        for idx, turn in enumerate(turns):
            current_round = idx + 1
            customer_input = turn["customer_input"]
            true_intent = turn["intent"]
            candidate_responses = turn["candidate_responses"]
            
            # 构造截止到当前轮的对话上下文：历史对话 + 当前客户输入
            dialogue_context_for_sample = current_context.copy()
            dialogue_context_for_sample.append({
                "role": "customer",
                "content": customer_input
            })
            
            # 拼接对话上下文字符串
            dialogue_context_str = "\n".join([
                f"{item['role']}: {item['content']}" 
                for item in dialogue_context_for_sample
            ])
            
            # 构造Alpaca格式三个字段：instruction=系统提示词，input=对话上下文+当前识别任务，output=识别结果
            input_content = f"""## 对话上下文（按顺序排列）
{dialogue_context_str}
## 当前待识别客户输入
轮次：{current_round}
客户输入：{customer_input}
请输出识别结果："""
            
            output_content = json.dumps({
                "round": current_round,
                "customer_input": customer_input,
                "predicted_intent": true_intent
            }, ensure_ascii=False)
            
            # 添加到训练样本
            alpaca_data.append({
                "instruction": system_prompt_instruction,
                "input": input_content,
                "output": output_content
            })
            
            # 随机选一个候选回复作为AI的回答，加到上下文里供下一轮使用
            selected_response = random.choice(candidate_responses)
            current_context.append({
                "role": "ai",
                "content": selected_response
            })
            current_context.append({
                "role": "customer",
                "content": customer_input
            })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成：共 {len(raw_data['test_cases'])} 个对话，生成 {len(alpaca_data)} 个训练样本，已保存到 {output_path}")

# 转换训练集和测试集
if __name__ == "__main__":
    #convert_to_alpaca("simulation_train_cases_100_final.json", "intent_train.json")
    convert_to_alpaca("simulation_test_cases.json", "intent_test.json")
