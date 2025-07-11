import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import qwen_model, req_emb
from lightrag.utils import EmbeddingFunc
import numpy as np

import requests
from bs4 import BeautifulSoup






class YuqueClient:
    def __init__(self, config):
        self.config = config
        self.YUQUE_API_URL_TEMPLATE = "https://yuque-api.antfin-inc.com/api/v2/repos/{}/{}{}"

    def get_doc_content(self, doc_id):
        try:
            response = requests.get(self.build_request(f"/docs/{doc_id}"), headers=self.build_headers())
            response.raise_for_status()
            response_json = response.json()
            data_json = response_json['data']
            body_html = data_json['body']
            print("原始文档")
            print(body_html)
            text = self.clean_html(body_html)
            print("处理的文档")
            print(text)
            return text
        except Exception as e:
            raise RuntimeError("Query Yuque doc detail failed!") from e

    def list_doc(self):
        try:
            response = requests.get(self.build_request("/toc"), headers=self.build_headers())
            response.raise_for_status()
            response_json = response.json()
            doc_infos = response_json['data']
            filtered_docs = [
                doc for doc in doc_infos
                if doc['type'].lower() == "doc" and doc['parent_uuid'] == self.config['source_uuid']
            ]
            return filtered_docs
        except Exception as e:
            print(e)
            raise RuntimeError("Query Yuque doc list failed!") from e

    def move_doc(self, node_uuid):
        data = {
            "action": "prependChild",
            "node_uuid": node_uuid,
            "target_uuid": self.config['archive_uuid']
        }
        try:
            response = requests.put(self.build_request("/toc"), json=data, headers=self.build_headers())
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError("Move Yuque doc failed!") from e

    def build_request(self, path):
        return self.YUQUE_API_URL_TEMPLATE.format(
            self.config['group_login'],
            self.config['book_slug'],
            path
        )

    def build_headers(self):
        return  {
            'X-Auth-Token': self.config.get('token'),
            'Content-Type': 'application/json',
        }

    def clean_html(self, html):
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()



WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await qwen_model(
        prompt,
        history_messages=history_messages,
        system_prompt=system_prompt,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
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


async def main():
    try:
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

        # 假设你的配置如下
        # config = {
        #     'group_login': 'zhc8k7',
        #     'book_slug': 'gsbi8g',
        #     'token': 's7PMvyM3PZH9zOIU4wWKY7x8EeW8FF3meXYkKdkQ',
        #     'source_uuid': 'c7j2coekbIrMLYMW',
        #     'archive_uuid': '6opEhZ7RpbXfvEZb'
        # }

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
                "有哪些退票的故障？",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())

