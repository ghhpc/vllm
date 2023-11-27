from vllm import LLM

prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
# llm = LLM(model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
llm = LLM(model="core42/jais-13b", trust_remote_code=True)
outputs = llm.generate(prompts)  # Generate texts from the prompts.
print(outputs)