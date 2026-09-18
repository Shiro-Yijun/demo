from dotenv import load_dotenv
import os

load_dotenv(r"D:\Badou learning\Homework\Demos\.env")
key = os.getenv("DEEPSEEK_API_KEY")
print(key)
