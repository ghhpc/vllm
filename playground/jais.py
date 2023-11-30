from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
llm = LLM(model="core42/jais-13b-chat", trust_remote_code=True)
outputs = llm.generate(prompts, sampling_params)  # Generate texts from the prompts.
print(outputs)