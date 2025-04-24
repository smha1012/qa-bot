from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
import streamlit as st
from retrievers import init_retriever
from states import GraphState
from rag import create_rag_chain
from nodes import *
import ormsgpack


DB_INDEX = "LANGCHAIN_DB_INDEX"


def safe_serialize(value):
    try:
        ormsgpack.packb(value)
        return True
    except Exception as e:
        st.warning(f"⚠️ 직렬화 불가 객체 감지됨: {type(value)} → 제거됨")
        return False
        
def create_graph():
    retriever = init_retriever()
    # 문서 검색 체인 생성
    rag_chain = create_rag_chain()

    # 그래프 상태 초기화
    workflow = StateGraph(GraphState)

    # 노드 정의
    workflow.add_node("query_expand", QueryRewriteNode())  # 질문 재작성
    workflow.add_node("query_rewrite", QueryRewriteNode())  # 질문 재작성
    workflow.add_node("web_search", WebSearchNode())  # 웹 검색
    workflow.add_node("retrieve", RetrieveNode(retriever))  # 문서 검색
    workflow.add_node("grade_documents", FilteringDocumentsNode())  # 문서 평가
    workflow.add_node(
        "general_answer", GeneralAnswerNode(ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0))
    )  # 일반 답변 생성
    workflow.add_node("rag_answer", RagAnswerNode(rag_chain))  # RAG 답변 생성

    # 엣지 추가
    workflow.add_conditional_edges(
        START,
        RouteQuestionNode(),
        {
            "query_expansion": "query_expand",  # 웹 검색으로 라우팅
            "general_answer": "general_answer",  # 벡터스토어로 라우팅
        },
    )

    workflow.add_edge("query_expand", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_web_search_node,
        {
            "web_search": "web_search",  # 웹 검색 필요
            "rag_answer": "rag_answer",  # RAG 답변 생성 가능
        },
    )

    workflow.add_edge("query_rewrite", "rag_answer")

    workflow.add_conditional_edges(
        "rag_answer",
        AnswerGroundednessCheckNode(),
        {
            "relevant": END,
            "not relevant": "web_search",
            "not grounded": "query_rewrite",
        },
    )

    workflow.add_edge("web_search", "rag_answer")

    # 그래프 컴파일
    app = workflow.compile(checkpointer=MemorySaver())
    return app


def stream_graph(
    app,
    query: str,
    streamlit_container,
    thread_id: str,
):
    config = RunnableConfig(recursion_limit=9, configurable={"thread_id": thread_id})

    # AgentState 객체를 활용하여 질문을 입력합니다.
    inputs = GraphState(question=query, generation="", documents=[])

    # app.stream을 통해 입력된 메시지에 대한 출력을 스트리밍합니다.
    actions = {
        "retrieve": "🔍 문서를 조회하는 중입니다.",
        "grade_documents": "👀 조회한 문서 중 중요한 내용을 추려내는 중입니다.",
        "rag_answer": "🔥 문서를 기반으로 답변을 생성하는 중입니다.",
        "general_answer": "🔥 문서를 기반으로 답변을 생성하는 중입니다.",
        "web_search": "🛜 웹 검색을 진행하는 중입니다.",
    }

    try:
        # streamlit_container
        with streamlit_container.status(
            "😊 열심히 생각중 입니다...", expanded=True
        ) as status:
            st.write("🧑‍💻 질문의 의도를 분석하는 중입니다.")
            for output in app.stream(inputs, config=config):
                # 출력된 결과에서 키와 값을 순회합니다.
                for key, value in output.items():
                    if not safe_serialize(value):
                        continue  # 직렬화 불가 객체 건너뛰기
                    # 노드의 이름과 해당 노드에서 나온 출력을 출력합니다.
                    if key in actions:
                        st.write(actions[key])
                # 출력 값을 예쁘게 출력합니다.
            status.update(label="답변 완료", state="complete", expanded=False)
    except GraphRecursionError as e:
        print(f"Recursion limit reached: {e}")
    #return app.get_state(config={"configurable": {"thread_id": thread_id}}).values

    state = app.get_state(config={"configurable": {"thread_id": thread_id}}).values

    # 안전 필터링
    safe_state = {
        k: v for k, v in state.items()
        if isinstance(v, (str, int, float, list, dict, type(None)))
    }
    
    return safe_state
