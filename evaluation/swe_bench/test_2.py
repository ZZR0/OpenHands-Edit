import asyncio


# 模拟异步任务1
async def task1():
    print('任务1：开始')
    await asyncio.sleep(2)  # 模拟耗时操作
    print('任务1：完成')
    return '任务1结果'


# 模拟异步任务2，依赖任务1的结果
async def task2(result_from_task1):
    print(f'任务2：接收到 {result_from_task1}')
    await asyncio.sleep(1)  # 模拟耗时操作
    print('任务2：完成')
    return '任务2结果'


# 模拟异步任务3，依赖任务2的结果
async def task3(result_from_task2):
    print(f'任务3：接收到 {result_from_task2}')
    await asyncio.sleep(1)  # 模拟耗时操作
    print('任务3：完成')


# 模拟一个与任务链无关的异步任务（并发任务）
async def unrelated_task():
    for i in range(30):
        print(f'无关任务：正在执行第 {i + 1} 次循环')
        await asyncio.sleep(0.5)  # 模拟耗时操作


# 主函数
async def main():
    # 启动任务链
    task_chain = asyncio.create_task(task1())  # 任务1
    unrelated = asyncio.create_task(unrelated_task())  # 无关任务并发执行

    # 等待任务链完成
    result1 = await task_chain
    result2 = await task2(result1)
    await task3(result2)

    # 等待无关任务完成
    await unrelated


# 运行主函数
asyncio.run(main())
