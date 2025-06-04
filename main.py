# pip install openai-agents
# pip install python-dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print(gemini_api_key)
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name = 'Translator',
    instructions= 
    """You are a translator agent. generate translation of urdu into english.
    یک جدید ٹیکنالوجی ہے جو انسان کی سوچنے، سمجھنے اور فیصلہ کرنے کی صلاحیت کو مشینوں میں منتقل کرتی ہے۔ یہ نظام خود سیکھ سکتا ہے، مسائل کا حل نکال سکتا ہے اور مختلف شعبوں جیسے طب، تعلیم، صنعت، اور روزمرہ زندگی میں انسان کی مدد کرتا ہے۔ آج کے دور میں مصنوعی ذہانت ترقی کا اہم ذریعہ بن چکی ہے ",
"""
)

response = Runner.run_sync(
    agent,
    input = "translate given paragraph in english",
    run_config = config
    )
print(response.final_output)
