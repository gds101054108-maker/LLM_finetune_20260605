# 大模型优化指南·LlamaFactory微调实战（xx车险意图识别场景）
本项目是面向大模型微调入门的全流程实操培训教程，基于LlamaFactory框架+Qwen3.5系列模型，以xx车险外呼意图识别为真实业务场景，覆盖**环境安装→数据准备→微调训练→效果评估→量化部署**全流程，所有代码、配置、数据集开箱即用，可快速复现完整微调流程。

---

## 📁 目录结构说明
```
├── 📂 lingClaw_demo/                    # LingClaw智能体生成仿真数据参考示例
│   ├── lingclaw提示词.txt               # xx车险外呼场景对话生成开箱即用提示词
│   ├── lingclaw响应结果.txt             # LingClaw生成仿真数据的输出样例
│   └── picc_intent_simulation_data.json # LingClaw生成的标准仿真数据集样例
├── 📂 优秀学员学习考核资料参考/          # 培训考核配套参考资料
│   ├── 培训记录20260604.pdf             # 线下培训课件记录
│   ├── 真实业务微调.zip                  # 真实业务场景微调参考资源
│   └── 真实业务模型准确率评估.xlsx       # 模型效果评估模板
├── augment_train_data.py                # 训练数据扩增脚本：支持100条原始样本扩增到10000条
├── convert_data.py                      # 数据格式转换脚本：原始JSON转LlamaFactory支持的Alpaca格式
├── dataset_info.json                    # LlamaFactory数据集注册配置文件
├── eval_test.py                         # 本地模型测试集批量评估脚本：自动统计准确率、输出错误案例
├── get_local_models.py                  # 本地/内网API可用模型列表查询脚本
├── intent_recognition_prompt.txt        # xx车险外呼意图识别任务专属系统提示词（14类意图）
├── intent_test.json                     # 转换完成的标准测试集（Alpaca格式，可直接用于训练）
├── local_api_test.py                    # 内网API批量测试脚本：无需本地部署模型，直接调用公司大模型资源测试
├── simulation_test_cases.json           # 原始仿真测试集（10条对话）
├── simulation_train_cases_100_final.json# 原始仿真训练集（100条标注对话）
├── 大模型优化指南20260606.md             # 主教程文档：全流程从入门到实操详细说明
└── .gitignore                           # Git忽略配置
```

---

## ✨ 核心特性
1. **全流程可复现**：所有步骤均有完整文档、配套脚本和示例数据，零基础也能跑通完整微调流程
2. **仿真业务场景**：基于xx车险外呼业务需求，14类标准意图，贴近实际开发场景
3. **多微调方案支持**：内置LoRA/Freeze/全参数微调/DeepSpeed分布式训练等多种方案现成配置
4. **开箱即用工具链**：数据转换、扩增、训练、评估全流程脚本齐全，无需额外开发
5. **明确考核标准**：配套实操考核要求，方便培训验收和技能验证

---

## 🚀 快速开始
### 1. 环境安装
参考《大模型优化指南20260605.md》中「2. LlamaFactory环境安装与验证」章节，完成Python、CUDA、LlamaFactory环境安装，验证命令：
```bash
# 验证LlamaFactory安装成功
lmf version
# 验证CUDA可用
python -c "import torch;print(torch.cuda.is_available())"
```

### 2. 数据准备
- 直接使用项目自带的仿真数据集：`simulation_train_cases_100_final.json`（训练集）、`simulation_test_cases.json`（测试集）
- 自定义数据集：按照原始数据格式准备后，使用`convert_data.py`转换为Alpaca格式，使用`augment_train_data.py`扩增样本量
- 数据集注册：将转换好的JSON放到LlamaFactory的`data`目录下，修改`dataset_info.json`添加自定义数据集配置

### 3. 模型微调
#### 推荐快速入门方案：LoRA微调（低显存、快速度）
参考教程「3.4.1 LoRA微调配置」创建训练配置文件，执行训练命令：
```bash
# 单卡训练示例
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ./train_lora_intent.yaml
# 多卡DeepSpeed训练（推荐，显存占用更低）
CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 llamafactory-cli train ./train_lora_intent_deepspeed.yaml
```

### 4. 效果评估
#### 方式1：本地加载模型评估
```bash
python eval_test.py
```
#### 方式2：调用内网API评估（无需本地部署模型）
```bash
python local_api_test.py
```
运行完成后会自动输出测试集准确率，错误案例会保存到`error_cases.json`方便分析优化。

---

## 📋 实操考核标准（全6项必完成）
参考教程「4. 大模型微调实操考核标准」章节，所有考核项完成即为达标：
1. ✅ LlamaFactory训练环境安装验证
2. ✅ 多场景模型性能测试（内网API+本地基座模型）
3. ✅ 现有仿真数据集LoRA微调实操
4. ✅ LingClaw仿真数据集生成（≥200条训练集、≥20条测试集）
5. ✅ 自定义仿真数据集LoRA微调
6. （可选）真实业务场景微调优化

---

## 📝 注意事项
1. 所有脚本中的路径需要根据自己本地环境修改，避免路径不存在报错
2. 训练前必须先在`dataset_info.json`中注册自己的数据集，否则LlamaFactory无法识别
3. 内网api_base找培训人员要
4. 生成训练/测试集时要保证两个数据集的意图类别完全一致，避免评估结果失真
5. 显存不足时可通过减小`per_device_train_batch_size`、增大`gradient_accumulation_steps`、开启`gradient_checkpointing`降低显存需求

---

## 📚 参考资料
- LlamaFactory官方GitHub：https://github.com/hiyouga/LLaMA-Factory
- Qwen3.5模型官方文档：https://qwen.readthedocs.io/


