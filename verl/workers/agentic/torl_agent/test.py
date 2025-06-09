import time
import os
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import Callable
from tqdm import tqdm
import asyncio
from tqdm.asyncio import tqdm as async_tqdm

def generate_test_codes(n=1000):
    """Generate n simple arithmetic operations"""
    codes = []
    for _ in range(n):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(['+', '-', '*', '/'])
        code = f"print({a} {op} {b})"
        codes.append(code)
    return codes

def send_request(tool_input):
    try:
        # Replace with your actual sandbox URL
        url = "http://0.0.0.0:8080/run_code"  # You'll need to replace this with actual URL
        response = requests.post(url, json=tool_input, timeout=10)
        return response.json()
    except:
        print("sandbox timeout")
        return {"run_result": {"stdout": "Error", "stderr": "TimeoutError"}}

def code_interpreter_single_call(code, timeout=20):
    """Single code interpreter call like in torlact.py"""
    tool_input = {'code': code, 'language': 'python'}
    try:
        result = send_request(tool_input)
    except:
        result = {"run_result": {"stdout": "Error", "stderr": "TimeoutError"}}
    
    def postproc(output):
        try:
            if str(output['run_result']['return_code']) == '0' or len(str(output['run_result']['stdout'])) != 0:
                return output['run_result']['stdout'], "Done"
            else:
                return output['run_result']['stdout'], output['run_result']['stderr'].strip()
        except Exception:
            return "Error", "UnknownError"
            
    return postproc(result)

def sequential_code_interpreter(codes):
    """Sequential implementation"""
    results = []
    for code in tqdm(codes, desc="Sequential"):
        results.append(code_interpreter_single_call(code))
    return results

def parallel_code_interpreter(codes):
    """Parallel implementation using ThreadPoolExecutor"""
    results = [None] * len(codes)
    with ThreadPoolExecutor(max_workers=max(min(len(codes), os.cpu_count(), 64), 1)) as executor:
        futures = {executor.submit(code_interpreter_single_call, code): i for i, code in enumerate(codes)}
        for future in tqdm(as_completed(futures), total=len(codes), desc="Parallel"):
            index = futures[future]
            try:
                results[index] = future.result()
            except:
                results[index] = ("Error", "UnknownError")
    return results

async def call_sync_from_async(fn: Callable, *args, **kwargs):
    """
    Shorthand for running a function in the default background thread pool executor
    and awaiting the result. The nature of synchronous code is that the future
    returned by this function is not cancellable
    """
    loop = asyncio.get_event_loop()
    coro = loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    result = await coro
    return result

class Agent:
    def __init__(self, code, index):
        self.code = code
        self.index = index
    
    async def run(self):
        result = await call_sync_from_async(code_interpreter_single_call, self.code)
        return self.index, result

async def process_queue(queue, num_workers=64):
    """Process the queue of agents"""
    results = [None] * queue.qsize()  # Pre-allocate results list
    pbar = tqdm(total=queue.qsize(), desc="Agents+Queue")
    
    async def worker():
        while True:
            try:
                agent = await queue.get()
                index, result = await agent.run()
                results[index] = result  # Store result at correct index
                pbar.update(1)
                queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in worker: {e}")
                pbar.update(1)
                queue.task_done()
    
    # Create workers
    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
    
    # Wait for queue to be processed
    await queue.join()
    
    # Cancel workers
    for w in workers:
        w.cancel()
    
    pbar.close()
    return results

async def run_async_test(codes):
    """Run the async test using multiple agents and a queue"""
    # Create queue and add agents
    queue = asyncio.Queue()
    for i, code in enumerate(codes):
        agent = Agent(code, i)  # Pass index to agent
        await queue.put(agent)
    
    # Process queue and get results
    results = await process_queue(queue)
    return results

def run_async_main(codes):
    """Helper to run async code in main"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_async_test(codes))

def main():
    # Generate test codes
    print("Generating test codes...")
    test_codes = generate_test_codes(1000)
    
    # Test sequential implementation
    print("\nTesting sequential implementation...")
    start_time = time.time()
    sequential_results = sequential_code_interpreter(test_codes)
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f} seconds")
    
    # Test parallel implementation with ThreadPoolExecutor
    print("\nTesting parallel implementation (ThreadPoolExecutor)...")
    start_time = time.time()
    parallel_results = parallel_code_interpreter(test_codes)
    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.2f} seconds")
    
    # Test implementation using agents and queue (like in torlact.py)
    print("\nTesting implementation with agents and queue (like in torlact.py)...")
    start_time = time.time()
    async_results = run_async_main(test_codes)
    async_time = time.time() - start_time
    print(f"Agents+Queue time: {async_time:.2f} seconds")
    
    # Compare results
    print("\nResults comparison:")
    print(f"Sequential vs Parallel results match: {sequential_results == parallel_results}")
    print(f"Sequential vs Agents+Queue results match: {sequential_results == async_results}")
    print(f"Parallel vs Agents+Queue results match: {parallel_results == async_results}")
    print(f"\nSpeed improvements:")
    print(f"ThreadPoolExecutor is {sequential_time / parallel_time:.2f}x faster than sequential")
    print(f"Agents+Queue is {sequential_time / async_time:.2f}x faster than sequential")
    print(f"Agents+Queue is {parallel_time / async_time:.2f}x faster than ThreadPoolExecutor")

if __name__ == "__main__":
    main()
