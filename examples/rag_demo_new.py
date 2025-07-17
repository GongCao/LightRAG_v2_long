import os
import asyncio
import argparse
from lightrag import LightRAG, QueryParam
from lightrag.llm import QwenModel, req_emb
from lightrag.utils import EmbeddingFunc
import numpy as np
import pandas as pd
import json



WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

qwen_model = QwenModel()
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await qwen_model.get_response(
        prompt,
        history_messages=history_messages,
        system_prompt=system_prompt,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    texts = json.dumps(texts, ensure_ascii=False)
    return await req_emb(texts)


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def main(query: str):
    try:
        print("begin")
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,
                func=embedding_func,
            ),
        )
        # df = pd.read_excel('data.xlsx')
        # answers = df['Content'].tolist()
        # print(f"Answers list length: {len(answers)}")
        # await rag.ainsert(answers)
        
        df = pd.read_excel('data.xlsx')
        answers = df['Content'].tolist()
        questions = df['Title'].tolist()
        print(len(questions))
        print(len(answers))
        doc = ""
        for question, answer in zip(questions, answers):
          tmp = f" (Question: [{question}], Answer: [{answer}]);"
          doc = doc + tmp
        await rag.ainsert(doc)
        # 假设你的配置如下

        # yuque_client = YuqueClient(config)

        # 获取文档内容

        # 列出文档
        # doc_list = yuque_client.list_doc()
        #
        # for doc in doc_list:
        #     print(doc)
        #     doc_content = yuque_client.get_doc_content(doc_id=doc.get('id'))
        #     # 移动文档
        #     await rag.ainsert(doc_content)
        #     yuque_client.move_doc(node_uuid='your_node_uuid')


        # # Perform naive search
        # print(
        #     await rag.aquery(
        #         "哪些地方受到故障影响？", param=QueryParam(mode="naive")
        #     )
        # )
        #
        # # Perform local search
        # print(
        #     await rag.aquery(
        #         "哪些地方受到故障影响？", param=QueryParam(mode="local")
        #     )
        # )
        #
        # # Perform global search
        # print(
        #     await rag.aquery(
        #         "哪些地方受到故障影响？",
        #         param=QueryParam(mode="global"),
        #     )
        # )

        # Perform hybrid search
        print(
            await rag.aquery(
                query,
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query for LightRAG")
    parser.add_argument('query', type=str, help="Query to be run against LightRAG")
    args = parser.parse_args()

    asyncio.run(main(args.query))
    # asyncio.run(main())

