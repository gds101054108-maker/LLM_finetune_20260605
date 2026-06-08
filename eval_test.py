import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm # 可选，显示进度条，需要先pip install tqdm


import re
# 预处理函数：自动提取JSON内容
def extract_json(text):
    # 优先匹配被```json包裹的内容
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 没匹配到就找第一个{和最后一个}之间的内容
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1].strip()
    return text



# ---------------------- 固定配置（和训练时完全一致，不要修改） ----------------------
# 和训练时用的系统提示词完全一致，必须加上才能保证推理效果和训练一致
system_prompt = """# 人保外呼场景意图识别任务Prompt
## 任务说明
你现在需要完成人保车险外呼场景的客户意图识别任务：
1. 参考**完整对话上下文**（包括之前的AI坐席回复和客户历史输入），对当前客户的最新输入进行意图分类
2. 必须从下方【意图列表】中选择最匹配的唯一意图，不允许自定义新的意图
3. 意图识别需要结合上下文判断，不能仅看当前客户输入的字面意思
4. 【重要】输出必须严格按照指定的JSON格式，**不允许输出任何解释性内容、说明、思考过程，只输出纯JSON**
5. 如果需要输出思考内容，必须把JSON放在所有内容的最后面，且用```json和```包裹
## 意图列表（共14类）
| 意图ID | 意图描述
|--------|--------
| greeting_respond | 客户对AI坐席的问候/开场做出回应
| insurance_expiry_inquiry | 客户询问自己的车险保单到期时间
| premium_consult | 客户咨询续保保费金额、保费构成明细
| claim_process_inquiry | 客户咨询理赔申请流程、已有理赔案件的进度
| policy_info_verify | 客户核实保单上的个人信息、车辆信息是否正确
| discount_consult | 客户询问当前续保有没有优惠活动、折扣福利
| refuse_insurance_renewal | 客户明确表示不想在人保续保、拒绝续保邀约
| demand_registration | 客户提出需要登记特殊需求、提供具体信息、确认业务办理
| complaint_feedback | 客户反馈过往服务不满、提出投诉建议
| call_back_request | 客户表示现在不方便沟通，要求后续再回电联系
| address_change_request | 客户要求修改保单上的联系地址、收件地址
| end_conversation | 客户主动提出要结束本次通话
| uncertain_inquiry | 客户表述模糊不清、提出疑问，需要AI进一步澄清需求
| payment_method_consult | 客户咨询保费缴纳的方式、支持的缴费渠道
## 输出要求
输出为纯JSON格式，结构如下：
```json
{
  "round": "<当前对话轮次，数字>",
  "customer_input": "<客户当前输入的原文，和输入完全一致>",
  "predicted_intent": "<你预测的意图ID，必须严格匹配上面列表中的意图ID，不能写错字>"
}
```
【再次强调】不要输出任何除了JSON之外的内容，不要加任何解释、说明、开场白！
"""
# --------------------------------------------------------------------------------------

# ---------------------- 模型加载配置（根据你的模型类型修改即可） ----------------------
# 配置1：加载LoRA合并后的完整模型（推荐）
#model_path = "./output/qwen3.5_0.8b_intent_merged"
model_path="/home/user/dsq/LLaMA-Factory/Qwen/Qwen3.5-0.8B"
#model_path="./output/qwen3.5_0.8b_intent_full/checkpoint-125"
#model_path="./output/qwen3.5_0.8b_intent_freeze"
#model_path="./output/qwen3.5_0.8b_intent_gptq"
model_kwargs = dict(
    device_map="auto"
)

# 配置2：加载AWQ量化后的模型（需要先安装auto-gptq依赖）
# model_path = "./output/qwen3.5_0.8b_intent_awq"
# model_kwargs = dict(
#     device_map="auto",
#     load_in_4bit=True
# )
# --------------------------------------------------------------------------------------

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

# 加载测试集
with open("data/intent_test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

total = 0
correct = 0
intent_stats = defaultdict(lambda: {"total": 0, "correct": 0})
bad_cases = []

# 批量推理（加进度条方便看进度）
for sample in tqdm(test_data, desc="评估中"):
    total += 1
    user_input = sample["input"] # Alpaca格式直接取input字段
    true_output = json.loads(sample["output"])
    true_intent = true_output["predicted_intent"]
    intent_stats[true_intent]["total"] += 1
    
    try:
        # 构造对话格式（必须加system角色，和训练输入格式完全一致）
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 推理生成
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.01,
                do_sample=False
            )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
        # 解析结果
        print(response) 
        pred_output = extract_json(response)
        
        pred_output = json.loads(pred_output)

        pred_intent = pred_output["predicted_intent"]
        
        if pred_intent == true_intent:
            correct += 1
            intent_stats[true_intent]["correct"] += 1
        else:
            bad_cases.append({
                "input": user_input,
                "true_intent": true_intent,
                "pred_intent": pred_intent
            })
            
    except Exception as e:
        print('**********',e)
        bad_cases.append({
            "input": user_input,
            "true_intent": true_intent,
            "error": str(e)
        })

# 输出评估结果
print(f"\n=== 整体评估结果 ===\n")
print(f"总样本数：{total}")
print(f"正确识别数：{correct}")
print(f"整体准确率：{correct/total:.2%}\n")
print(f"=== 各意图识别准确率 ===\n")
for intent, stats in intent_stats.items():
    acc = stats["correct"]/stats["total"] if stats["total"]>0 else 0
    print(f"{intent}: {acc:.2%} ({stats['correct']}/{stats['total']})")

# 保存badcase
with open("bad_cases.json", "w", encoding="utf-8") as f:
    json.dump(bad_cases, f, ensure_ascii=False, indent=2)
print(f"\nBadcase已保存到bad_cases.json，共{len(bad_cases)}条")
