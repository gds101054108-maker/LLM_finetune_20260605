import requests
import os
os.environ['PYTHONUTF8'] = '1'

API_URL = "http://10.0.1.133:18080/v1/models"
LOCAL_API_KEY = "sk-bf-4b61e8a0-fcc7-43f2-9d29-e5ef1b609bf4"

API_URL="http://10.0.1.133:3030/v1/models"
LOCAL_API_KEY=""

def get_models():
    headers = {
        "Authorization": f"Bearer {LOCAL_API_KEY}"
    }
    response = requests.get(API_URL, headers=headers, timeout=60)
    if response.status_code == 200:
        models = response.json()
        print("本地接口部署的模型列表：")
        print("="*60)
        for model in models.get('data', []):
            print(f"- {model['id']}")
        print("="*60)
        return models
    else:
        print(f"获取模型列表失败，错误码：{response.status_code}, 错误信息：{response.text}")
        return None

if __name__ == "__main__":
    get_models()
