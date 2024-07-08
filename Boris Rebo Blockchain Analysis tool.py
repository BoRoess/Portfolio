#!/usr/bin/env python
# coding: utf-8

# 

# ## Demo library
# 
# Info about this transaction [b036fd0dbbdc26b454aa56104b8e2f1cf7a223c371a03b3f38f02a0fc3e73d39](https://www.blockchain.com/explorer/transactions/btc/b036fd0dbbdc26b454aa56104b8e2f1cf7a223c371a03b3f38f02a0fc3e73d39)

# In[ ]:





# ## Exercise 1 - Bitcoin block analysis
# 
# You'll analyse the [block 750,000](https://blockchain.com/btc/block/750000) of Bitcoin Blockchain 
# 
# 1. **Count the number of transactions containing:**
#    - 1 to 5 outputs
#    - 6 to 25 outputs
#    - 26 to 100 outputs
#    - 101 and more outputs (we will call them batched transactions)
# 
# 2. **Compute the median and average fee of batched transactions.**
# 
#    If you don't know know how to calculate the fees, I recommend the video below
# 
# 4. **Identify the most expensive transaction (highest sum of amounts of the inputs) and display:**
#    - Txid (transaction id)
#    - Count of inputs/outputs of this transaction
#    - Sum of inputs/outputs of this transaction

# In[ ]:


get_ipython().run_cell_magic('html', '', '<iframe width="560" height="315" src="https://www.youtube.com/embed/0_5wb5agLqE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\n')


# In[ ]:


# Import necessary libraries
import asyncio
from chainspect.redis import RedisChainspectorConnector

# Connector for the Redis instances
connector = RedisChainspectorConnector.live()

# The address to connect to
block = 750000

async with connector as cs:
    # List all the transactions for the block
    txs = await cs.block_height_txs(block)
    # TODO


# ## Exercise 2 - Identifying Transaction Paths to a Target Address
# 
# ### Objective
# Develop a function that identifies which outputs from block `N` can reach a fixed address `X` through transaction flows within a specified maximum number of hops (steps). The function should then store and return the addresses of these initial outputs.
# 
# ### Hints
# 
# 1. **List all transaction outputs from a block:**
#    - Retrieve and list all the transaction outputs in the specified block.
# 
# 2. **Trace transaction paths:**
#    - For each transaction output, trace the transaction paths forward (next transaction) to determine if it can connect to the fixed address `X`.
#    - Follow the flow of transactions to see where the money goes from each output.
# 
# 3. **Limit traversal:**
#    - Continue tracing the transaction paths up to a maximum of `y` hops (steps from one output to another).
# 
# 4. **Identify and store reachable outputs:**
#    - For each output, determine if a path exists to the fixed address `X` within  `y` hops.
#    - If such a path exists, store the address of the initial output.
# 
# 5. **Return initial output addresses:**
#    - Return a list of addresses of the initial outputs that can reach address `X` within the  `y`-hop limit.
# 
# ### Success metric
# 
# The tests outlined below should be successful. If you obtain different results, please provide an explanation.

# In[ ]:


# Import necessary libraries
import asyncio
from chainspect.redis import RedisChainspectorConnector

async def get_forward_tree_address_from_block(cs, address_to_reach, block, max_distance=3) -> list[str]:
    """
    Returns a list of addresses of the initial outputs that can reach a specific address within the y-hop limit.
    
    Parameters
    ----------
    cs : Chainspector
        An instance of a `Chainspector`.
    address_to_reach : str
        Address to be used as the end of the path.
    block : int
        Block number
    max_distance : int
        Limits the distance to search. If no path within `max_distance` from
        the list of inputs in the 'block' to `address_to_reach`, the result will be an empty list.
    
    Returns
    -------
    list[str]
        A list of all addresses that have a path between 'address_to_reach' and 
        all the addresses in the selected 'block'.
    """
    list_of_addresses = []
    # Add the logic to find the list of addresses here

    return list_of_addresses


async def test_get_forward_tree_address(cs):
    """
    Test function to verify that get_forward_tree_address works correctly.
    """
    # Test 1
    address_to_connect = '18mddZvpUTLqb1twgokF5HtVbb45YJFhdB'
    expected_addresses = ['1HfrzKE9K8cHQ1Le6SgARp8uoba5gruAx9', '1BvccStvq5piUqd7AByST9sqWWfpHjz3Et']
    block = 120001
    max_distance = 2
    result = await get_forward_tree_address_from_block(cs, address_to_connect, block, max_distance)
    assert set(result) == set(expected_addresses), f"Expected {expected_addresses}, but got {result}"
    print("Test 1 successful !")
    
    # Test 2
    address_to_connect = '1CDysWzQ5Z4hMLhsj4AKAEFwrgXRC8DqRN'
    expected_addresses = ['18eGJJUZeKoCHb7CXQdhJhQrKPhHUsVbfE', '1MtWU6RkJbXJkFv6Dw37Q7ukekQAXMjeQe',
                          '1PQjzVgHnz7T6Lr8zgRLop8K5qcL5xQMfz', '1Tp3NNN7c3xMaQuew87ecpZ4S3bnLaesX',
                          '1PCZT941y7t12BMWeQYTuSZL5XC5235e6a']
    block = 160004
    max_distance = 3
    result = await get_forward_tree_address_from_block(cs, address_to_connect, block, max_distance)
    assert set(result) == set(expected_addresses), f"Expected {expected_addresses}, but got {result}"
    print("Test 2 successful !")

if __name__ == "__main__":
    connector = RedisChainspectorConnector.live()
    async with connector as cs:
        await test_get_forward_tree_address(cs)


# ## If you want to play more, floor is yours !
# 
# You have free rein to show us your analytical, scientific and creativity skills. From descriptive statistics, graphs, forecasting, benchmarking, reporting or others it is up to you!

# In[3]:


#Assignment 1 
import asyncio
import pandas as pd
from statistics import median
from chainspect.redis import RedisChainspectorConnector
import time
import pydantic

# Connector for the Redis instances
connector = RedisChainspectorConnector.live()

# The block to analyze
block_height = 750000

async def fetch_transaction_details(cs, tx: str):
    inputs = await cs.tx_inputs(tx)
    outputs = await cs.tx_outputs(tx)
    input_values = await asyncio.gather(*[cs.output_value(inp) for inp in inputs])
    output_values = await asyncio.gather(*[cs.output_value(out) for out in outputs])
    input_sum = sum(input_values)
    output_sum = sum(output_values)
    fee = input_sum - output_sum
    return {
        'txid': tx,
        'input_count': len(inputs),
        'output_count': len(outputs),
        'input_sum': input_sum,
        'output_sum': output_sum,
        'fee': fee
    }

async def analyze_block_transactions(cs, block_height: int):
    # Fetch transactions for the block
    txs = await cs.block_height_txs(block_height)
    
    # Fetch transaction details concurrently
    transactions = await asyncio.gather(*[fetch_transaction_details(cs, tx) for tx in txs])
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(transactions)

    # 1. Count the number of transactions containing specific ranges of outputs
    count_1_to_5 = df[(df['output_count'] >= 1) & (df['output_count'] <= 5)].shape[0]
    count_6_to_25 = df[(df['output_count'] >= 6) & (df['output_count'] <= 25)].shape[0]
    count_26_to_100 = df[(df['output_count'] >= 26) & (df['output_count'] <= 100)].shape[0]
    count_101_and_more = df[df['output_count'] >= 101].shape[0]
    
    # 2. Compute the median and average fee of batched transactions (101 and more outputs)
    batched_tx_fees = df[df['output_count'] >= 101]['fee']
    median_fee = median(batched_tx_fees) if not batched_tx_fees.empty else 0
    average_fee = batched_tx_fees.mean() if not batched_tx_fees.empty else 0

    # 3. Identify the most expensive transaction
    most_expensive_tx = df.loc[df['input_sum'].idxmax()]

    return {
        'count_1_to_5': count_1_to_5,
        'count_6_to_25': count_6_to_25,
        'count_26_to_100': count_26_to_100,
        'count_101_and_more': count_101_and_more,
        'median_fee': median_fee,
        'average_fee': average_fee,
        'most_expensive_tx': {
            'txid': most_expensive_tx['txid'],
            'input_count': most_expensive_tx['input_count'],
            'output_count': most_expensive_tx['output_count'],
            'input_sum': most_expensive_tx['input_sum'],
            'output_sum': most_expensive_tx['output_sum']
        }
    }

async def main():
    start_time = time.time()
    async with connector as cs:
        analysis_result = await analyze_block_transactions(cs, block_height)
        print(analysis_result)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())


# In[127]:


#Final Answer Task 2
from typing import List, Set
import asyncio
import pandas as pd
from chainspect.redis.redis_chainspector import RedisChainspector

async def fetch_block_transactions(cs: RedisChainspector, block: int) -> pd.DataFrame:
    txids = await cs.block_height_txs(block)
    transactions = []
    for txid in txids:
        inputs = await cs.tx_inputs(txid)
        outputs = await cs.tx_outputs(txid)
        for inp in inputs:
            transactions.append({'source': inp, 'destination': txid})
        for out in outputs:
            transactions.append({'source': txid, 'destination': out})
    return pd.DataFrame(transactions)

async def trace_transaction_outputs(chainspector: RedisChainspector, output_id: str, target_address: str, max_hops: int, current_hops: int = 0, visited: Set[str] = None) -> bool:
    if current_hops > max_hops-1:
        return False
    
    if visited is None:
        visited = set()

    if output_id in visited:
        return False

    visited.add(output_id)

    next_txs = await chainspector.output_next_tx(output_id)
    
    for next_tx in next_txs:
        outputs = await chainspector.tx_outputs(next_tx)
        for out in outputs:
            try:
                out_address = await chainspector.output_address(out)
            except ValueError:
                continue  # If the output address is not found, continue with the next output
            if out_address == target_address:
                return True
            if await trace_transaction_outputs(chainspector, out, target_address, max_hops, current_hops + 1, visited):
                return True

    return False

async def trace_bitcoin_transactions(chainspector: RedisChainspector, block_height: int, target_address: str, max_hops: int) -> List[str]:
    txids = await chainspector.block_height_txs(block_height)
    initial_outputs = set()
    
    for txid in txids:
        outputs = await chainspector.tx_outputs(txid)
        for output in outputs:
            try:
                initial_output_address = await chainspector.output_address(output)
            except ValueError:
                continue  # If the initial output address is not found, continue with the next output
            result = await trace_transaction_outputs(chainspector, output, target_address, max_hops)
            if result:
                initial_outputs.add(initial_output_address)
    
    return list(initial_outputs)

async def main():
    connector = RedisChainspectorConnector.live()
    async with connector as cs:
        # Example parameters
        block_height = 160004 # Block height to analyze
        target_address = '1CDysWzQ5Z4hMLhsj4AKAEFwrgXRC8DqRN'  # Target address to reach
        max_hops = 3  # Maximum number of hops

        initial_outputs = await trace_bitcoin_transactions(cs, block_height, target_address, max_hops)
        print("Initial Outputs that can reach the target address:", initial_outputs)

# Run the main function
asyncio.run(main())


# In[ ]:




